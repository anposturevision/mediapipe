#include "run_graph_gpu.h"

#include <cstdint>
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

// const char kLandmarksOutputStream[] = "hand_landmarks";
// const char kHandednessOutputStream[] = "handedness";
const char kInputStream[] = "input_video";

class MPPGraphRunner {
    public:
        MPPGraphRunner(std::string task) : task_(task) {
            if (task_ == "hand_landmarks") {
                kOutputStream_ = "hand_landmarks";
            } else if (task_ == "face_mesh") {
                kOutputStream_ = "multi_face_landmarks";
            } else {
                ABSL_LOG(ERROR) << "Unsupported task: " << task_;
            }
        }

        absl::Status InitMPPGraph(std::string calculator_graph_config_file) {
            std::string calculator_graph_config_contents;
            MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
                calculator_graph_config_file,
                &calculator_graph_config_contents));
            ABSL_LOG(INFO) << "Get calculator graph config contents: "
                    << calculator_graph_config_contents;
            mediapipe::CalculatorGraphConfig config =
                mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

            ABSL_LOG(INFO) << "Initialize the calculator graph.";
            MP_RETURN_IF_ERROR(graph_.Initialize(config));

            ABSL_LOG(INFO) << "Initialize the GPU.";
            MP_ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
            MP_RETURN_IF_ERROR(graph_.SetGpuResources(std::move(gpu_resources)));
            gpu_helper_.InitializeForTest(graph_.GetGpuResources().get());

            ABSL_LOG(INFO) << "Initialize output stream poller.";

            MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks_tmp,
                            graph_.AddOutputStreamPoller(kOutputStream_));
            poller_landmarks_ = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_landmarks_tmp));


            MP_RETURN_IF_ERROR(graph_.StartRun({}));
            ABSL_LOG(INFO) << "Graph initialized successfully.";
            return absl::OkStatus();
        }

        absl::Status ProcessFrame(cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<mediapipe::NormalizedLandmarkList> &landmarks) {
            cv::cvtColor(camera_frame, camera_frame, cv::COLOR_BGR2RGBA);
            auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
                mediapipe::ImageFormat::SRGBA, camera_frame.cols, camera_frame.rows,
                mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
            cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
            camera_frame.copyTo(input_frame_mat);

            MP_RETURN_IF_ERROR(
                gpu_helper_.RunInGlContext([this, &input_frame, &frame_timestamp_us]() -> absl::Status {
                    // Convert ImageFrame to GpuBuffer.
                    auto texture = gpu_helper_.CreateSourceTexture(*input_frame.get());
                    auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
                    glFlush();
                    texture.Release();
                    // Send GPU image packet into the graph.
                    MP_RETURN_IF_ERROR(graph_.AddPacketToInputStream(
                        kInputStream, mediapipe::Adopt(gpu_frame.release())
                                        .At(mediapipe::Timestamp(frame_timestamp_us))));
                    return absl::OkStatus();
                })
            );

            mediapipe::Packet packet_landmarks;
            poller_landmarks_->Next(&packet_landmarks);
            landmarks = packet_landmarks.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

            return absl::OkStatus();
        }

    private:
        mediapipe::CalculatorGraph graph_;
        mediapipe::GlCalculatorHelper gpu_helper_;

        std::string task_;
        std::string kOutputStream_ = "";
        std::unique_ptr<mediapipe::OutputStreamPoller> poller_landmarks_;
};

bool HandTrackingGraphRunner::initGraph(const std::string& calculator_graph_config_file) {
    MPPGraphRunner mpp_graph_runner("hand_landmarks");
    absl::Status status = mpp_graph_runner.InitMPPGraph(calculator_graph_config_file);
    if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to initialize graph: " << status.message();
        return false;
    }
    runnerVoid = &mpp_graph_runner; // Store the runner in a void pointer
    return true;
}

bool HandTrackingGraphRunner::ProcessFrame(cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks) {
    MPPGraphRunner &runner = *(MPPGraphRunner *)runnerVoid;
    std::vector<mediapipe::NormalizedLandmarkList> landmarks_tmp;
    absl::Status status = runner.ProcessFrame(camera_frame, frame_timestamp_us, landmarks_tmp);
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

bool FacemeshGraphRunner::initGraph(const std::string& calculator_graph_config_file) {
    MPPGraphRunner mpp_graph_runner("face_mesh");
    absl::Status status = mpp_graph_runner.InitMPPGraph(calculator_graph_config_file);
    if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to initialize graph: " << status.message();
        return false;
    }
    runnerVoid = &mpp_graph_runner; // Store the runner in a void pointer
    return true;
}

bool FacemeshGraphRunner::ProcessFrame(cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks) {
    MPPGraphRunner &runner = *(MPPGraphRunner *)runnerVoid;
    std::vector<mediapipe::NormalizedLandmarkList> landmarks_tmp;
    absl::Status status = runner.ProcessFrame(camera_frame, frame_timestamp_us, landmarks_tmp);
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