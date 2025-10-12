#include "run_graph_gpu.h"

#include <cstdint>
#include <chrono>
#include <thread>
#include <memory>
#include <mutex>
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/status_macros.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/gpu/gpu_service.h"
#include "mediapipe/util/resource_util.h"

// Include the new lifecycle management headers
#include "mediapipe/framework/deps/model_lifecycle_manager.h"
#include "mediapipe/framework/deps/scoped_model_resources.h"
#include "mediapipe/framework/deps/scoped_options_registry.h"

// CUDA and OpenCV CUDA support
#include <opencv2/core/cuda.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

// const char kLandmarksOutputStream[] = "hand_landmarks";
// const char kHandednessOutputStream[] = "handedness";
const char kInputStream[] = "input_video";


class MPPGraphRunner {
    public:
        MPPGraphRunner(std::string task) : task_(task) {
            // Create a unique model ID for this instance
            model_id_ = task + "_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
            
            // Initialize lifecycle management
            lifecycle_manager_ = std::make_unique<mediapipe::ScopedModelManager>(model_id_);
            
            // Create scoped resource managers
            resource_cache_ = mediapipe::CreateScopedModelResourcesCache(model_id_);
            options_registry_ = mediapipe::CreateScopedOptionsRegistry(model_id_);
            
            // Register with lifecycle manager
            resource_cache_->RegisterWithLifecycleManager(lifecycle_manager_->operator->());
            options_registry_->RegisterWithLifecycleManager(lifecycle_manager_->operator->());
            
            if (task_ == "hand_landmarks") {
                kOutputStream_ = "hand_landmarks";
            } else if (task_ == "face_mesh") {
                kOutputStream_ = "multi_face_landmarks";
            } else {
                ABSL_LOG(ERROR) << "Unsupported task: " << task_;
            }
        }

        ~MPPGraphRunner() {
            ABSL_LOG(INFO) << "Destroying MPPGraphRunner for model: " << model_id_;
            
            // Clean up OpenGL texture properly
            if (reusable_gl_texture_ != 0) {
                ABSL_LOG(INFO) << "Cleaning up GL texture handle: " << reusable_gl_texture_;
                if (gpu_helper_.Initialized()) {
                    try {
                        gpu_helper_.RunInGlContext([this]() -> absl::Status {
                            glDeleteTextures(1, &reusable_gl_texture_);
                            return absl::OkStatus();
                        });
                    } catch (...) {
                        ABSL_LOG(WARNING) << "Failed to delete OpenGL texture in destructor";
                    }
                }
                reusable_gl_texture_ = 0;
            }
            
            // Reset smart pointers - let them clean up automatically
            poller_landmarks_.reset();
            lifecycle_manager_.reset();
            resource_cache_.reset();
            options_registry_.reset();
            
            // Let the graph destructor handle the rest
        }

        absl::Status InitMPPGraph(std::string calculator_graph_config_file) {
            // Add initialization lock to prevent race conditions
            static std::mutex init_mutex;
            std::lock_guard<std::mutex> lock(init_mutex);
            
            std::string calculator_graph_config_contents;
            MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
                calculator_graph_config_file,
                &calculator_graph_config_contents));
            
            mediapipe::CalculatorGraphConfig config =
                mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

            // Initialize the calculator graph
            MP_RETURN_IF_ERROR(graph_.Initialize(config));

            // Create GPU resources for this instance
            MP_ASSIGN_OR_RETURN(gpu_resources_, mediapipe::GpuResources::Create());
            ABSL_LOG(INFO) << "GPU resources created for model: " << model_id_;
            
            // Set the GPU resources for this instance
            MP_RETURN_IF_ERROR(graph_.SetGpuResources(gpu_resources_));
            
            // Initialize GPU helper with instance resources
            gpu_helper_.InitializeForTest(gpu_resources_.get());

            // Initialize output stream poller with non-blocking mode
            MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks_tmp,
                            graph_.AddOutputStreamPoller(kOutputStream_));
            poller_landmarks_ = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_landmarks_tmp));

            // Start the graph
            MP_RETURN_IF_ERROR(graph_.StartRun({}));
            
            return absl::OkStatus();
        }

        absl::Status processFrame(const cv::cuda::GpuMat &camera_frame, size_t frame_timestamp_us, std::vector<mediapipe::NormalizedLandmarkList> &landmarks) {
            MP_RETURN_IF_ERROR(
                gpu_helper_.RunInGlContext([&]() -> absl::Status {
                // Create or resize GL texture only when needed
                if (reusable_gl_texture_ == 0 || 
                    texture_width_ != camera_frame.cols || 
                    texture_height_ != camera_frame.rows) {
                    
                    if (reusable_gl_texture_ != 0) {
                        glDeleteTextures(1, &reusable_gl_texture_);
                    }
                    
                    glGenTextures(1, &reusable_gl_texture_);
                    glBindTexture(GL_TEXTURE_2D, reusable_gl_texture_);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 
                                camera_frame.cols, camera_frame.rows,
                                0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                    
                    texture_width_ = camera_frame.cols;
                    texture_height_ = camera_frame.rows;
                } else {
                    glBindTexture(GL_TEXTURE_2D, reusable_gl_texture_);
                }
                
                // Use CUDA-OpenGL interop (zero-copy GPU-to-GPU transfer)
                cudaGraphicsResource_t cuda_resource;
                cudaError_t cuda_status = cudaGraphicsGLRegisterImage(&cuda_resource, reusable_gl_texture_,
                                                GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
                if (cuda_status != cudaSuccess) {
                    return absl::InternalError("Failed to register GL texture with CUDA");
                }
                
                cuda_status = cudaGraphicsMapResources(1, &cuda_resource);
                if (cuda_status != cudaSuccess) {
                    cudaGraphicsUnregisterResource(cuda_resource);
                    return absl::InternalError("Failed to map CUDA resources");
                }
                
                cudaArray_t cuda_array;
                cuda_status = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);
                if (cuda_status != cudaSuccess) {
                    cudaGraphicsUnmapResources(1, &cuda_resource);
                    cudaGraphicsUnregisterResource(cuda_resource);
                    return absl::InternalError("Failed to get mapped CUDA array");
                }
                
                cuda_status = cudaMemcpy2DToArray(cuda_array, 0, 0,
                                    camera_frame.data, camera_frame.step,
                                    camera_frame.cols * camera_frame.elemSize(), 
                                    camera_frame.rows,
                                    cudaMemcpyDeviceToDevice);
                if (cuda_status != cudaSuccess) {
                    cudaGraphicsUnmapResources(1, &cuda_resource);
                    cudaGraphicsUnregisterResource(cuda_resource);
                    return absl::InternalError("Failed to copy data to CUDA array");
                }
                
                // Unmap and unregister CUDA resource immediately after use
                cuda_status = cudaGraphicsUnmapResources(1, &cuda_resource);
                if (cuda_status != cudaSuccess) {
                    ABSL_LOG(WARNING) << "Failed to unmap CUDA resources: " << cuda_status;
                }
                cuda_status = cudaGraphicsUnregisterResource(cuda_resource);
                if (cuda_status != cudaSuccess) {
                    ABSL_LOG(WARNING) << "Failed to unregister CUDA resource: " << cuda_status;
                }
                
                // Wrap GL texture as GpuBuffer
                auto texture_buffer = mediapipe::GlTextureBuffer::Wrap(
                    GL_TEXTURE_2D, reusable_gl_texture_,
                    camera_frame.cols, camera_frame.rows,
                    mediapipe::GpuBufferFormat::kBGRA32,
                    gpu_helper_.GetSharedGlContext(),
                    [](std::shared_ptr<mediapipe::GlSyncPoint> sync) {}
                );
                
                auto gpu_buffer = mediapipe::GpuBuffer(std::move(texture_buffer));
                
                // Send to graph
                MP_RETURN_IF_ERROR(graph_.AddPacketToInputStream(
                    kInputStream, 
                    mediapipe::MakePacket<mediapipe::GpuBuffer>(gpu_buffer)
                        .At(mediapipe::Timestamp(frame_timestamp_us))
                ));
                
                return absl::OkStatus();
                })
            );
            
            // Wait for the graph to process this specific timestamp
            MP_RETURN_IF_ERROR(graph_.WaitUntilIdle());

            // Check the queue - if there's a packet, get it; if not, no face detected
            mediapipe::Packet packet_landmarks;
            int queue_size = poller_landmarks_->QueueSize();
            
            if (queue_size > 0) {
            bool has_packet = poller_landmarks_->Next(&packet_landmarks);
            if (has_packet && !packet_landmarks.IsEmpty()) {
                landmarks = packet_landmarks.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            } else {
                landmarks.clear();
            }
            } else {
            // No packet in queue means no face detected
            landmarks.clear();
            }
            
            return absl::OkStatus();
        }

        // Add explicit cleanup method
        void cleanup() {
            ABSL_LOG(INFO) << "Starting explicit cleanup for model: " << model_id_;
            
            // Close input stream
            if (graph_.HasInputStream(kInputStream)) {
                auto status = graph_.CloseInputStream(kInputStream);
                if (!status.ok()) {
                    ABSL_LOG(ERROR) << "Failed to close input stream during cleanup: " << status.message();
                }
            }
            
            // Reset poller
            poller_landmarks_.reset();
            
            // Wait for completion and clean up GPU context
            auto status = graph_.WaitUntilDone();
            if (!status.ok()) {
                ABSL_LOG(ERROR) << "Graph cleanup error: " << status.message();
            }
            
            // Clean up OpenGL texture
            if (reusable_gl_texture_ != 0 && gpu_helper_.Initialized()) {
                gpu_helper_.RunInGlContext([this]() -> absl::Status {
                    glDeleteTextures(1, &reusable_gl_texture_);
                    return absl::OkStatus();
                });
                reusable_gl_texture_ = 0;
            }
            
            if (gpu_helper_.Initialized()) {
                gpu_helper_.RunInGlContext([]() -> absl::Status {
                    glFinish();
                    return absl::OkStatus();
                });
            }
            
            // Clean up instance GPU resources
            if (gpu_resources_) {
                gpu_resources_.reset();
                ABSL_LOG(INFO) << "GPU resources cleaned up for model: " << model_id_;
            }
            
            ABSL_LOG(INFO) << "Explicit cleanup completed for model: " << model_id_;
        }

        // Reset tracking state by restarting the graph
        absl::Status resetTracking() {
            ABSL_LOG(INFO) << "Resetting tracking state for model: " << model_id_;
            
            // Close all input streams to stop processing
            MP_RETURN_IF_ERROR(graph_.CloseAllPacketSources());
            
            // Wait for graph to finish processing current packets
            MP_RETURN_IF_ERROR(graph_.WaitUntilDone());
            
            // Restart the graph - this clears all calculator state including tracking
            MP_RETURN_IF_ERROR(graph_.StartRun({}));
            
            ABSL_LOG(INFO) << "Tracking state reset completed for model: " << model_id_;
            return absl::OkStatus();
        }

    private:
        mediapipe::GlCalculatorHelper gpu_helper_;
        mediapipe::CalculatorGraph graph_;
        std::shared_ptr<mediapipe::GpuResources> gpu_resources_;

        std::string task_;
        std::string kOutputStream_ = "";
        std::unique_ptr<mediapipe::OutputStreamPoller> poller_landmarks_;
        
        // Lifecycle management members
        std::string model_id_;
        std::unique_ptr<mediapipe::ScopedModelManager> lifecycle_manager_;
        std::unique_ptr<mediapipe::ScopedModelResourcesCache> resource_cache_;
        std::unique_ptr<mediapipe::ScopedOptionsRegistry> options_registry_;
        
        // CUDA-OpenGL interop members
        GLuint reusable_gl_texture_ = 0;
        int texture_width_ = 0;
        int texture_height_ = 0;
};


HandTrackingGraphRunner::~HandTrackingGraphRunner() {
    if (runnerVoid != nullptr) {
        MPPGraphRunner* runner = static_cast<MPPGraphRunner*>(runnerVoid);
        // Explicit cleanup before destruction
        runner->cleanup();
        delete runner;
        runnerVoid = nullptr;
    }
}

bool HandTrackingGraphRunner::initGraph(const std::string& calculator_graph_config_file) {
    // Create the runner on the heap to ensure it persists
    MPPGraphRunner* mpp_graph_runner = new MPPGraphRunner("hand_landmarks");
    absl::Status status = mpp_graph_runner->InitMPPGraph(calculator_graph_config_file);
    if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to initialize graph: " << status.message();
        delete mpp_graph_runner;
        return false;
    }
    runnerVoid = mpp_graph_runner; // Store the runner pointer
    return true;
}

bool HandTrackingGraphRunner::processFrame(const cv::cuda::GpuMat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks) {
    MPPGraphRunner &runner = *(MPPGraphRunner *)runnerVoid;
    std::vector<mediapipe::NormalizedLandmarkList> landmarks_tmp;
    absl::Status status = runner.processFrame(camera_frame, frame_timestamp_us, landmarks_tmp);
    if (!status.ok()) {
        std::cout << "Failed to process the frame: " << status.message() << std::endl;
        return false;
    }

    landmarks.resize(landmarks_tmp.size());
    for (int i = 0; i < landmarks_tmp.size(); i++) {
        landmarks[i].landmarks.resize(landmarks_tmp[i].landmark_size());
        // landmarks[i].presence.resize(landmarks_tmp[i].landmark_size());
        landmarks[i].visibility.resize(landmarks_tmp[i].landmark_size());
        for (int j = 0; j < landmarks_tmp[i].landmark_size(); j++) {
            landmarks[i].landmarks[j].x = landmarks_tmp[i].landmark(j).x();
            landmarks[i].landmarks[j].y = landmarks_tmp[i].landmark(j).y();
            landmarks[i].landmarks[j].z = landmarks_tmp[i].landmark(j).z();
            // landmarks[i].presence[j] = landmarks_tmp[i].landmark(j).presence();
            landmarks[i].visibility[j] = landmarks_tmp[i].landmark(j).visibility();
        }
    }

	return status.ok();
}

void HandTrackingGraphRunner::cleanup() {
    if (runnerVoid != nullptr) {
        static_cast<MPPGraphRunner*>(runnerVoid)->cleanup();
    }
}

bool HandTrackingGraphRunner::resetTracking() {
    if (runnerVoid != nullptr) {
        MPPGraphRunner* runner = static_cast<MPPGraphRunner*>(runnerVoid);
        absl::Status status = runner->resetTracking();
        if (!status.ok()) {
            std::cout << "Failed to reset tracking: " << status.message() << std::endl;
            return false;
        }
        return true;
    }
    return false;
}

FacemeshGraphRunner::~FacemeshGraphRunner() {
    if (runnerVoid != nullptr) {
        MPPGraphRunner* runner = static_cast<MPPGraphRunner*>(runnerVoid);
        // Explicit cleanup before destruction
        runner->cleanup();
        delete runner;
        runnerVoid = nullptr;
    }
}

bool FacemeshGraphRunner::initGraph(const std::string& calculator_graph_config_file) {
    // Create the runner on the heap to ensure it persists
    MPPGraphRunner* mpp_graph_runner = new MPPGraphRunner("face_mesh");
    absl::Status status = mpp_graph_runner->InitMPPGraph(calculator_graph_config_file);
    if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to initialize graph: " << status.message();
        delete mpp_graph_runner;
        return false;
    }
    runnerVoid = mpp_graph_runner; // Store the runner pointer
    return true;
}

bool FacemeshGraphRunner::processFrame(const cv::cuda::GpuMat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks) {
    MPPGraphRunner &runner = *(MPPGraphRunner *)runnerVoid;
    std::vector<mediapipe::NormalizedLandmarkList> landmarks_tmp;
    absl::Status status = runner.processFrame(camera_frame, frame_timestamp_us, landmarks_tmp);
    if (!status.ok()) {
        std::cout << "Failed to process the frame: " << status.message() << std::endl;
        return false;
    }

    landmarks.resize(landmarks_tmp.size());
    for (int i = 0; i < landmarks_tmp.size(); i++) {
        landmarks[i].landmarks.resize(landmarks_tmp[i].landmark_size());
        // landmarks[i].presence.resize(landmarks_tmp[i].landmark_size());
        landmarks[i].visibility.resize(landmarks_tmp[i].landmark_size());
        for (int j = 0; j < landmarks_tmp[i].landmark_size(); j++) {
            landmarks[i].landmarks[j].x = landmarks_tmp[i].landmark(j).x();
            landmarks[i].landmarks[j].y = landmarks_tmp[i].landmark(j).y();
            landmarks[i].landmarks[j].z = landmarks_tmp[i].landmark(j).z();
            // landmarks[i].presence[j] = landmarks_tmp[i].landmark(j).presence();
            landmarks[i].visibility[j] = landmarks_tmp[i].landmark(j).visibility();
        }
    }

    return status.ok();
}

void FacemeshGraphRunner::cleanup() {
    if (runnerVoid != nullptr) {
        static_cast<MPPGraphRunner*>(runnerVoid)->cleanup();
    }
}

bool FacemeshGraphRunner::resetTracking() {
    if (runnerVoid != nullptr) {
        MPPGraphRunner* runner = static_cast<MPPGraphRunner*>(runnerVoid);
        absl::Status status = runner->resetTracking();
        if (!status.ok()) {
            std::cout << "Failed to reset tracking: " << status.message() << std::endl;
            return false;
        }
        return true;
    }
    return false;
}