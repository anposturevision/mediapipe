#ifndef RUN_GRAPH_GPU_H
#define RUN_GRAPH_GPU_H

#include <cstdlib>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

// Forward declaration
class MPPGraphRunner;

struct LandmarkList {
    std::vector<cv::Point3f> landmarks;
    std::vector<float> visibility;
};

class GraphRunner {
    public:
        virtual ~GraphRunner() = default;
        virtual bool initGraph(const std::string& calculator_graph_config_file) = 0;
        virtual bool processFrame(const cv::cuda::GpuMat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks) = 0;
        virtual void cleanup() = 0;  // Add explicit cleanup method
        virtual bool resetTracking() = 0;  // Add reset tracking method
};

class HandTrackingGraphRunner : public GraphRunner {
    public:
        HandTrackingGraphRunner() = default;
        ~HandTrackingGraphRunner();

        bool initGraph(const std::string& calculator_graph_config_file);
        bool processFrame(const cv::cuda::GpuMat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks);
        void cleanup() override;  // Explicit cleanup method
        bool resetTracking() override;  // Reset tracking method

    private:
        void* runnerVoid = nullptr;
};

class FacemeshGraphRunner : public GraphRunner {
    public:
        FacemeshGraphRunner() = default;
        ~FacemeshGraphRunner();

        bool initGraph(const std::string& calculator_graph_config_file);
        bool processFrame(const cv::cuda::GpuMat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks);
        void cleanup() override;  // Explicit cleanup method
        bool resetTracking() override;  // Reset tracking method

    private:
        void* runnerVoid = nullptr;
};

#endif // RUN_GRAPH_GPU_H

