#include <cstdlib>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

struct LandmarkList {
    std::vector<cv::Point3f> landmarks;
    std::vector<float> visibility;
};

class GraphRunner {
    public:
        virtual ~GraphRunner() = default;
        virtual bool initGraph(const std::string& calculator_graph_config_file) = 0;
        virtual bool ProcessFrame(cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks) = 0;
};

class HandTrackingGraphRunner : public GraphRunner {
    public:
        HandTrackingGraphRunner() = default;
        ~HandTrackingGraphRunner() = default;

        bool initGraph(const std::string& calculator_graph_config_file);
        bool ProcessFrame(cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks);

    private:
        void* runnerVoid;
};

class FacemeshGraphRunner : public GraphRunner {
    public:
        FacemeshGraphRunner() = default;
        ~FacemeshGraphRunner() = default;

        bool initGraph(const std::string& calculator_graph_config_file);
        bool ProcessFrame(cv::Mat &camera_frame, size_t frame_timestamp_us, std::vector<LandmarkList> &landmarks);

    private:
        void* runnerVoid;
};

