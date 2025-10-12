#include "run_graph_gpu.h"

#include <cstdint>
#include <chrono>
#include <thread>
#include <memory>
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
            // Proper cleanup sequence to prevent memory leaks
            ABSL_LOG(INFO) << "Destroying MPPGraphRunner for model: " << model_id_;
            
            // 1. Close input stream first
            if (graph_.HasInputStream(kInputStream)) {
                auto status = graph_.CloseInputStream(kInputStream);
                if (!status.ok()) {
                    ABSL_LOG(ERROR) << "Failed to close input stream: " << status.message();
                }
            }
            
            // 2. Reset poller before shutting down graph
            poller_landmarks_.reset();
            
            // 3. Wait for graph to finish processing and then shut it down
            auto status = graph_.WaitUntilDone();
            if (!status.ok()) {
                ABSL_LOG(ERROR) << "Graph did not finish properly: " << status.message();
            }
            
            // 4. Explicitly clean up GPU resources in the correct context
            if (gpu_helper_.Initialized()) {
                gpu_helper_.RunInGlContext([this]() -> absl::Status {
                    // Force cleanup of any remaining GPU resources
                    glFinish(); // Wait for all GPU operations to complete
                    return absl::OkStatus();
                });
            }
            
            // 5. Clean up lifecycle-managed resources
            // This will automatically clean up all registered resources, caches, and registrations
            lifecycle_manager_.reset();  // This triggers all cleanup functions
            resource_cache_.reset();
            options_registry_.reset();
            
            ABSL_LOG(INFO) << "MPPGraphRunner cleanup completed for model: " << model_id_;
            
            // 6. The graph destructor will handle the rest of the cleanup
        }

        absl::Status InitMPPGraph(std::string calculator_graph_config_file) {
            std::string calculator_graph_config_contents;
            MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
                calculator_graph_config_file,
                &calculator_graph_config_contents));
            // ABSL_LOG(INFO) << "Get calculator graph config contents: "
            //         << calculator_graph_config_contents;
            mediapipe::CalculatorGraphConfig config =
                mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

            // ABSL_LOG(INFO) << "Initialize the calculator graph.";
            MP_RETURN_IF_ERROR(graph_.Initialize(config));

            // ABSL_LOG(INFO) << "Initialize the GPU.";
            MP_ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
            MP_RETURN_IF_ERROR(graph_.SetGpuResources(std::move(gpu_resources)));
            
            // Initialize GPU helper
            gpu_helper_.InitializeForTest(graph_.GetGpuResources().get());

            // ABSL_LOG(INFO) << "Initialize output stream poller.";

            MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks_tmp,
                            graph_.AddOutputStreamPoller(kOutputStream_));
            poller_landmarks_ = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_landmarks_tmp));


            MP_RETURN_IF_ERROR(graph_.StartRun({}));
            // ABSL_LOG(INFO) << "Graph initialized successfully.";
            return absl::OkStatus();
        }

        absl::Status processFrame(const cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<mediapipe::NormalizedLandmarkList> &landmarks) {
            auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
                mediapipe::ImageFormat::SRGBA, camera_frame.cols, camera_frame.rows,
                mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
            cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
            camera_frame.copyTo(input_frame_mat);

            MP_RETURN_IF_ERROR(
                gpu_helper_.RunInGlContext([this, &input_frame, &frame_timestamp_us]() -> absl::Status {
                    // Convert ImageFrame to GpuBuffer - using the efficient MediaPipe pattern
                    auto texture = gpu_helper_.CreateSourceTexture(*input_frame.get());
                    auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
                    
                    // Send GPU image packet into the graph.
                    MP_RETURN_IF_ERROR(graph_.AddPacketToInputStream(
                        kInputStream, mediapipe::Adopt(gpu_frame.release())
                                        .At(mediapipe::Timestamp(frame_timestamp_us))));
                    
                    // Release texture after packet is sent - this was the key fix
                    texture.Release();
                    // Force GPU to complete operations before returning
                    glFlush();
                    glFinish(); // Wait for GPU operations to complete
                    return absl::OkStatus();
                })
            );

            // Wait for the graph to process this specific timestamp
            MP_RETURN_IF_ERROR(graph_.WaitUntilIdle());

            // Check the queue - if there's a packet, get it; if not, no hands detected
            mediapipe::Packet packet_landmarks;
            int queue_size = poller_landmarks_->QueueSize();
            
            if (queue_size > 0) {
                bool has_packet = poller_landmarks_->Next(&packet_landmarks);
                if (has_packet && !packet_landmarks.IsEmpty()) {
                    landmarks = packet_landmarks.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
                } else {
                    landmarks.clear();
                    // ABSL_LOG(WARNING) << "No landmarks found in the packet.";
                }
            } else {
                // No packet in queue means no hands detected
                landmarks.clear();
            }

            // Force cleanup of any temporary GPU resources after processing
            if (gpu_helper_.Initialized()) {
                gpu_helper_.RunInGlContext([]() -> absl::Status {
                    glFlush();
                    return absl::OkStatus();
                });
            }

            return absl::OkStatus();
        }

        // Add explicit cleanup method
        void cleanup() {
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
            
            if (gpu_helper_.Initialized()) {
                gpu_helper_.RunInGlContext([]() -> absl::Status {
                    glFinish();
                    return absl::OkStatus();
                });
            }
        }

    private:
        mediapipe::CalculatorGraph graph_;
        mediapipe::GlCalculatorHelper gpu_helper_;

        std::string task_;
        std::string kOutputStream_ = "";
        std::unique_ptr<mediapipe::OutputStreamPoller> poller_landmarks_;
        
        // Lifecycle management members
        std::string model_id_;
        std::unique_ptr<mediapipe::ScopedModelManager> lifecycle_manager_;
        std::unique_ptr<mediapipe::ScopedModelResourcesCache> resource_cache_;
        std::unique_ptr<mediapipe::ScopedOptionsRegistry> options_registry_;
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

bool HandTrackingGraphRunner::processFrame(const cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks) {
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

bool FacemeshGraphRunner::processFrame(const cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks) {
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