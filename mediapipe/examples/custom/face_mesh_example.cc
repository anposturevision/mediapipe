#include "mediapipe/examples/custom/run_graph_gpu.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include <iostream>

int main() {
    FacemeshGraphRunner face_mesh;
    std::string graph_config = "/build/mediaipe/mediapipe/graphs/face_mesh/face_mesh_gpu_lib.pbtxt";
    if (!face_mesh.initGraph(graph_config)) {
        std::cerr << "Failed to initialize the face mesh graph." << std::endl;
        return -1;
    }
    
    
    return 0;
}