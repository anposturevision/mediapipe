// Example usage of the hand landmark extraction
#include "mediapipe/examples/custom/run_graph_gpu.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include <iostream>

int main() {
    HandTrackingGraphRunner hand_tracker;
    std::string graph_config = "/build/mediaipe/mediapipe/graphs/hand_tracking/hand_tracking_lib.pbtxt";
    if (!hand_tracker.initGraph(graph_config)) {
        std::cerr << "Failed to initialize the hand tracking graph." << std::endl;
        return -1;
    }
    
    
    return 0;
}
