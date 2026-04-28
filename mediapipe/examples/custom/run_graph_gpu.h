#ifndef RUN_GRAPH_GPU_H
#define RUN_GRAPH_GPU_H

#include <array>
#include <cstdlib>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

// Forward declaration
class MPPGraphRunner;

struct LandmarkList {
    std::vector<cv::Point3f> landmarks;
    std::vector<float> visibility;
};

// Row-major 4x4 face pose transform matrix produced by MediaPipe's
// face_geometry pipeline. One per detected face.
using FaceTransformMatrix = std::array<float, 16>;

class GraphRunner {
    public:
        virtual ~GraphRunner() = default;
        virtual bool initGraph(const std::string& calculator_graph_config_file) = 0;
        virtual bool processFrame(const cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks) = 0;
        virtual void cleanup() = 0;  // Add explicit cleanup method
};

class HandTrackingGraphRunner : public GraphRunner {
    public:
        HandTrackingGraphRunner() = default;
        ~HandTrackingGraphRunner();

        bool initGraph(const std::string& calculator_graph_config_file);
        bool processFrame(const cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks);
        void cleanup() override;  // Explicit cleanup method

    private:
        void* runnerVoid = nullptr;
};

class FacemeshGraphRunner : public GraphRunner {
    public:
        FacemeshGraphRunner() = default;
        ~FacemeshGraphRunner();

        bool initGraph(const std::string& calculator_graph_config_file);
        bool processFrame(const cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks);

        // Same as processFrame above, but also returns one 4x4 row-major pose
        // transform matrix per detected face from the face_geometry subgraph.
        // `transform_matrices.size()` matches `landmarks.size()` on success.
        // If the graph was loaded without face_geometry wired in, the vector
        // is left empty.
        bool processFrame(const cv::Mat &camera_frame, size_t frame_timestamp_us,
                          std::vector<LandmarkList> &landmarks,
                          std::vector<FaceTransformMatrix> &transform_matrices);
        void cleanup() override;

    private:
        void* runnerVoid = nullptr;
};

class PoseGraphRunner : public GraphRunner {
    public:
        PoseGraphRunner() = default;
        ~PoseGraphRunner();

        bool initGraph(const std::string& calculator_graph_config_file);
        bool processFrame(const cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks);
        void cleanup() override;

    private:
        void* runnerVoid = nullptr;
};

#endif // RUN_GRAPH_GPU_H

