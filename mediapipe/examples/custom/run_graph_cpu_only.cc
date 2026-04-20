#include "run_graph_cpu_only.h"

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
#include "mediapipe/util/resource_util.h"

// Include the lifecycle management headers
#include "mediapipe/framework/deps/model_lifecycle_manager.h"
#include "mediapipe/framework/deps/scoped_model_resources.h"
#include "mediapipe/framework/deps/scoped_options_registry.h"

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
            } else if (task_ == "pose_landmarks") {
                kOutputStream_ = "pose_landmarks";
            } else {
                ABSL_LOG(ERROR) << "Unsupported task: " << task_;
            }
        }

        ~MPPGraphRunner() {
            ABSL_LOG(INFO) << "Destroying MPPGraphRunner for model: " << model_id_;

            // Reset smart pointers - let them clean up automatically
            poller_landmarks_.reset();
            lifecycle_manager_.reset();
            resource_cache_.reset();
            options_registry_.reset();
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

            // Initialize the calculator graph (CPU-only, no GPU resources)
            MP_RETURN_IF_ERROR(graph_.Initialize(config));

            // Initialize output stream poller
            MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks_tmp,
                            graph_.AddOutputStreamPoller(kOutputStream_));
            poller_landmarks_ = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_landmarks_tmp));

            // Start the graph
            MP_RETURN_IF_ERROR(graph_.StartRun({}));

            return absl::OkStatus();
        }

        absl::Status processFrame(const cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<mediapipe::NormalizedLandmarkList> &landmarks) {
            // Wrap cv::Mat into ImageFrame
            auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
                mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
                mediapipe::ImageFrame::kDefaultAlignmentBoundary);
            cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
            camera_frame.copyTo(input_frame_mat);

            // Send to graph
            MP_RETURN_IF_ERROR(graph_.AddPacketToInputStream(
                kInputStream,
                mediapipe::Adopt(input_frame.release())
                    .At(mediapipe::Timestamp(frame_timestamp_us))
            ));

            // Wait for the graph to process this specific timestamp
            MP_RETURN_IF_ERROR(graph_.WaitUntilIdle());

            // Check the queue - if there's a packet, get it; if not, no detection
            mediapipe::Packet packet_landmarks;
            int queue_size = poller_landmarks_->QueueSize();

            if (queue_size > 0) {
                bool has_packet = poller_landmarks_->Next(&packet_landmarks);
                if (has_packet && !packet_landmarks.IsEmpty()) {
                    if (task_ == "pose_landmarks") {
                        landmarks = { packet_landmarks.Get<mediapipe::NormalizedLandmarkList>() };
                    } else {
                        landmarks = packet_landmarks.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
                    }
                } else {
                    landmarks.clear();
                }
            } else {
                // No packet in queue means no detection
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

            // Wait for completion
            auto status = graph_.WaitUntilDone();
            if (!status.ok()) {
                ABSL_LOG(ERROR) << "Graph cleanup error: " << status.message();
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
        mediapipe::CalculatorGraph graph_;

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
    for (int i = 0; i < (int)landmarks_tmp.size(); i++) {
        landmarks[i].landmarks.resize(landmarks_tmp[i].landmark_size());
        landmarks[i].visibility.resize(landmarks_tmp[i].landmark_size());
        for (int j = 0; j < landmarks_tmp[i].landmark_size(); j++) {
            landmarks[i].landmarks[j].x = landmarks_tmp[i].landmark(j).x();
            landmarks[i].landmarks[j].y = landmarks_tmp[i].landmark(j).y();
            landmarks[i].landmarks[j].z = landmarks_tmp[i].landmark(j).z();
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
        landmarks[i].visibility.resize(landmarks_tmp[i].landmark_size());
        for (int j = 0; j < landmarks_tmp[i].landmark_size(); j++) {
            landmarks[i].landmarks[j].x = landmarks_tmp[i].landmark(j).x();
            landmarks[i].landmarks[j].y = landmarks_tmp[i].landmark(j).y();
            landmarks[i].landmarks[j].z = landmarks_tmp[i].landmark(j).z();
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

PoseGraphRunner::~PoseGraphRunner() {
    if (runnerVoid != nullptr) {
        MPPGraphRunner* runner = static_cast<MPPGraphRunner*>(runnerVoid);
        runner->cleanup();
        delete runner;
        runnerVoid = nullptr;
    }
}

bool PoseGraphRunner::initGraph(const std::string& calculator_graph_config_file) {
    MPPGraphRunner* mpp_graph_runner = new MPPGraphRunner("pose_landmarks");
    absl::Status status = mpp_graph_runner->InitMPPGraph(calculator_graph_config_file);
    if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to initialize graph: " << status.message();
        delete mpp_graph_runner;
        return false;
    }
    runnerVoid = mpp_graph_runner;
    return true;
}

bool PoseGraphRunner::processFrame(const cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks) {
    MPPGraphRunner &runner = *(MPPGraphRunner *)runnerVoid;
    std::vector<mediapipe::NormalizedLandmarkList> landmarks_tmp;
    absl::Status status = runner.processFrame(camera_frame, frame_timestamp_us, landmarks_tmp);
    if (!status.ok()) {
        std::cout << "Failed to process the frame: " << status.message() << std::endl;
        return false;
    }

    landmarks.resize(landmarks_tmp.size());
    for (int i = 0; i < (int)landmarks_tmp.size(); i++) {
        landmarks[i].landmarks.resize(landmarks_tmp[i].landmark_size());
        landmarks[i].visibility.resize(landmarks_tmp[i].landmark_size());
        for (int j = 0; j < landmarks_tmp[i].landmark_size(); j++) {
            landmarks[i].landmarks[j].x = landmarks_tmp[i].landmark(j).x();
            landmarks[i].landmarks[j].y = landmarks_tmp[i].landmark(j).y();
            landmarks[i].landmarks[j].z = landmarks_tmp[i].landmark(j).z();
            landmarks[i].visibility[j] = landmarks_tmp[i].landmark(j).visibility();
        }
    }

    return status.ok();
}

void PoseGraphRunner::cleanup() {
    if (runnerVoid != nullptr) {
        static_cast<MPPGraphRunner*>(runnerVoid)->cleanup();
    }
}

bool PoseGraphRunner::resetTracking() {
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
