#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/modules/face_geometry/protos/face_geometry.pb.h"

#ifdef _WIN32
#define MP_FACEMESH_EXPORT extern "C" __declspec(dllexport)
#else
#define MP_FACEMESH_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct MpFacemeshResult {
    float* landmarks_out;
    int landmarks_capacity;
    int num_landmarks;
    float* transform_matrices_out;
    int transform_matrices_capacity;
    int num_transform_matrices;
};

MP_FACEMESH_EXPORT int mp_facemesh_process_frame_full(
    void* handle,
    const uint8_t* rgba_data,
    int width,
    int height,
    int stride,
    MpFacemeshResult* result);

namespace {

constexpr int kLandmarkValues = 4;
constexpr int kMaxFaceMeshLandmarks = 478;
constexpr size_t kFrameTimestampStepUs = 33333;
constexpr char kInputStream[] = "input_video";
constexpr char kLandmarksOutputStream[] = "multi_face_landmarks";
constexpr char kGeometryOutputStream[] = "multi_face_geometry";

using FaceTransformMatrix = std::array<float, 16>;

struct LandmarkList {
    std::vector<cv::Point3f> landmarks;
    std::vector<float> visibility;
};

class CpuFacemeshRunner {
public:
    bool Init(const char* graph_config_path) {
        std::string graph_config_contents;
        auto status = mediapipe::file::GetContents(
            graph_config_path, &graph_config_contents);
        if (!status.ok()) {
            return false;
        }

        mediapipe::CalculatorGraphConfig config =
            mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
                graph_config_contents);

        status = graph_.Initialize(config);
        if (!status.ok()) {
            return false;
        }

        auto landmarks_poller_or =
            graph_.AddOutputStreamPoller(kLandmarksOutputStream);
        if (!landmarks_poller_or.ok()) {
            return false;
        }
        landmarks_poller_ = std::make_unique<mediapipe::OutputStreamPoller>(
            std::move(landmarks_poller_or.value()));

        if (!HasOutputStream(config, kGeometryOutputStream)) {
            return false;
        }
        auto geometry_poller_or =
            graph_.AddOutputStreamPoller(kGeometryOutputStream);
        if (!geometry_poller_or.ok()) {
            return false;
        }
        geometry_poller_ = std::make_unique<mediapipe::OutputStreamPoller>(
            std::move(geometry_poller_or.value()));

        status = graph_.StartRun({});
        return status.ok();
    }

    bool ProcessFrame(
        const uint8_t* rgba_data,
        int width,
        int height,
        int stride,
        std::vector<LandmarkList>& faces,
        std::vector<FaceTransformMatrix>& transform_matrices) {
        cv::Mat rgba_view(
            height,
            width,
            CV_8UC4,
            const_cast<uint8_t*>(rgba_data),
            static_cast<size_t>(stride));
        cv::Mat rgb_frame;
        cv::cvtColor(rgba_view, rgb_frame, cv::COLOR_RGBA2RGB);

        auto input_frame = std::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB,
            width,
            height,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        rgb_frame.copyTo(input_frame_mat);

        const auto timestamp = mediapipe::Timestamp(next_timestamp_us_);
        auto status = graph_.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(timestamp));
        if (!status.ok()) {
            return false;
        }
        status = graph_.WaitUntilIdle();
        if (!status.ok()) {
            return false;
        }
        next_timestamp_us_ += kFrameTimestampStepUs;

        faces.clear();
        if (landmarks_poller_ != nullptr && landmarks_poller_->QueueSize() > 0) {
            mediapipe::Packet landmarks_packet;
            if (landmarks_poller_->Next(&landmarks_packet) &&
                !landmarks_packet.IsEmpty()) {
                const auto& landmarks_tmp =
                    landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
                faces.resize(landmarks_tmp.size());
                for (int i = 0; i < static_cast<int>(landmarks_tmp.size()); ++i) {
                    faces[i].landmarks.resize(landmarks_tmp[i].landmark_size());
                    faces[i].visibility.resize(landmarks_tmp[i].landmark_size());
                    for (int j = 0; j < landmarks_tmp[i].landmark_size(); ++j) {
                        faces[i].landmarks[j].x = landmarks_tmp[i].landmark(j).x();
                        faces[i].landmarks[j].y = landmarks_tmp[i].landmark(j).y();
                        faces[i].landmarks[j].z = landmarks_tmp[i].landmark(j).z();
                        faces[i].visibility[j] = landmarks_tmp[i].landmark(j).visibility();
                    }
                }
            }
        }

        transform_matrices.clear();
        if (geometry_poller_ != nullptr && geometry_poller_->QueueSize() > 0) {
            mediapipe::Packet geometry_packet;
            if (geometry_poller_->Next(&geometry_packet) &&
                !geometry_packet.IsEmpty()) {
                const auto& geometries =
                    geometry_packet.Get<std::vector<mediapipe::face_geometry::FaceGeometry>>();
                transform_matrices.reserve(geometries.size());
                for (const auto& geometry : geometries) {
                    FaceTransformMatrix matrix{};
                    if (geometry.has_pose_transform_matrix()) {
                        const auto& matrix_data = geometry.pose_transform_matrix();
                        const bool column_major =
                            matrix_data.layout() == mediapipe::MatrixData::COLUMN_MAJOR;
                        if (matrix_data.rows() == 4 &&
                            matrix_data.cols() == 4 &&
                            matrix_data.packed_data_size() == 16) {
                            for (int row = 0; row < 4; ++row) {
                                for (int col = 0; col < 4; ++col) {
                                    const int src = column_major
                                        ? col * 4 + row
                                        : row * 4 + col;
                                    matrix[row * 4 + col] = matrix_data.packed_data(src);
                                }
                            }
                        }
                    }
                    transform_matrices.push_back(matrix);
                }
            }
        }

        return true;
    }

private:
    static bool HasOutputStream(
        const mediapipe::CalculatorGraphConfig& config,
        const std::string& stream_name) {
        for (const auto& stream : config.output_stream()) {
            if (stream == stream_name ||
                stream.find(":" + stream_name) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    mediapipe::CalculatorGraph graph_;
    std::unique_ptr<mediapipe::OutputStreamPoller> landmarks_poller_;
    std::unique_ptr<mediapipe::OutputStreamPoller> geometry_poller_;
    size_t next_timestamp_us_ = 0;
};

struct MpFacemeshHandle {
    CpuFacemeshRunner runner;
};

}  // namespace

MP_FACEMESH_EXPORT void* mp_facemesh_init(const char* graph_config_path) {
    if (graph_config_path == nullptr || graph_config_path[0] == '\0') {
        return nullptr;
    }

    try {
        auto handle = std::make_unique<MpFacemeshHandle>();
        if (!handle->runner.Init(graph_config_path)) {
            return nullptr;
        }
        return handle.release();
    } catch (const std::exception&) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

MP_FACEMESH_EXPORT int mp_facemesh_process_frame_full(
    void* handle,
    const uint8_t* rgba_data,
    int width,
    int height,
    int stride,
    MpFacemeshResult* result) {
    if (result != nullptr) {
        result->num_landmarks = 0;
        result->num_transform_matrices = 0;
    }

    if (handle == nullptr || rgba_data == nullptr || result == nullptr ||
        result->landmarks_out == nullptr || result->landmarks_capacity <= 0 ||
        result->transform_matrices_out == nullptr ||
        result->transform_matrices_capacity <= 0 ||
        width <= 0 || height <= 0 || stride < width * 4) {
        return 0;
    }

    try {
        auto* facemesh = static_cast<MpFacemeshHandle*>(handle);
        std::vector<LandmarkList> faces;
        std::vector<FaceTransformMatrix> transform_matrices;
        if (!facemesh->runner.ProcessFrame(
                rgba_data,
                width,
                height,
                stride,
                faces,
                transform_matrices)) {
            return 0;
        }

        if (faces.empty()) {
            return 1;
        }

        const LandmarkList& face = faces.front();
        const int count = std::min<int>(
            static_cast<int>(face.landmarks.size()),
            std::min(result->landmarks_capacity, kMaxFaceMeshLandmarks));

        for (int i = 0; i < count; ++i) {
            result->landmarks_out[i * kLandmarkValues + 0] = face.landmarks[i].x;
            result->landmarks_out[i * kLandmarkValues + 1] = face.landmarks[i].y;
            result->landmarks_out[i * kLandmarkValues + 2] = face.landmarks[i].z;
            result->landmarks_out[i * kLandmarkValues + 3] =
                i < static_cast<int>(face.visibility.size())
                    ? face.visibility[i]
                    : 0.0f;
        }

        result->num_landmarks = count;

        const int matrix_count = std::min<int>(
            static_cast<int>(transform_matrices.size()),
            result->transform_matrices_capacity);
        for (int i = 0; i < matrix_count; ++i) {
            std::copy(
                transform_matrices[i].begin(),
                transform_matrices[i].end(),
                result->transform_matrices_out + i * 16);
        }
        result->num_transform_matrices = matrix_count;
        if (count > 0 && matrix_count == 0) {
            return 0;
        }
        return 1;
    } catch (const std::exception&) {
        return 0;
    } catch (...) {
        return 0;
    }
}

MP_FACEMESH_EXPORT void mp_facemesh_cleanup(void* handle) {
    delete static_cast<MpFacemeshHandle*>(handle);
}
