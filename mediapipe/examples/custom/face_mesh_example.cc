#include "mediapipe/examples/custom/run_graph_gpu.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <csignal>
#include <cstdlib>

void signal_handler(int signal) {
    std::cerr << "Received signal " << signal << " - exiting gracefully" << std::endl;
    exit(0);
}

int main() {
    // Set up signal handler to catch segfaults
    signal(SIGSEGV, signal_handler);
    signal(SIGABRT, signal_handler);
    
    try {
        FacemeshGraphRunner face_mesh;
        std::string graph_config = "/mediapipe/mediapipe/graphs/face_mesh/face_mesh_gpu_lib.pbtxt";
        
        std::cout << "Initializing face mesh graph..." << std::endl;
        if (!face_mesh.initGraph(graph_config)) {
            std::cerr << "Failed to initialize the face mesh graph." << std::endl;
            return -1;
        }
        std::cout << "Face mesh graph initialized successfully." << std::endl;

        cv::Mat camera_frame = cv::imread("/mediapipe/mediapipe/examples/custom/far_face.jpg");
        if (camera_frame.empty()) {
            std::cerr << "Failed to read the image file." << std::endl;
            return -1;
        }
        std::cout << "Image loaded successfully. Size: " << camera_frame.cols << "x" << camera_frame.rows << std::endl;

        // Convert to GPU memory for CUDA processing
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(camera_frame);
        std::cout << "Image uploaded to GPU memory." << std::endl;

        // Performance evaluation parameters
        const int num_iterations = 100;  // Number of times to run inference
        const int warmup_iterations = 10; // Warmup runs (not counted in timing)
        std::vector<double> inference_times;
        inference_times.reserve(num_iterations);
        
        // Use monotonically increasing timestamps
        size_t current_timestamp_us = 0;
        
        std::cout << "Starting performance evaluation..." << std::endl;
        std::cout << "Warmup iterations: " << warmup_iterations << std::endl;
        std::cout << "Timing iterations: " << num_iterations << std::endl;
        
        // Warmup runs
        for (int i = 0; i < warmup_iterations; i++) {
            std::vector<LandmarkList> landmarks;
            if (!face_mesh.processFrame(gpu_frame, current_timestamp_us, landmarks)) {
                std::cerr << "Failed to process frame during warmup iteration " << i << std::endl;
                return -1;
            }
            if (i == 0) {
                std::cout << "Warmup: Found " << landmarks.size() << " faces in first frame." << std::endl;
            }
            current_timestamp_us += 33333;  // Increment by ~30 FPS equivalent (33.333ms)
        }
        std::cout << "Warmup completed." << std::endl;
        
        // Timed inference runs
        for (int i = 0; i < num_iterations; i++) {
            std::vector<LandmarkList> landmarks;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            if (!face_mesh.processFrame(gpu_frame, current_timestamp_us, landmarks)) {
                std::cerr << "Failed to process frame during iteration " << i << std::endl;
                return -1;
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double inference_time_ms = duration.count() / 1000.0;
            inference_times.push_back(inference_time_ms);
            
            current_timestamp_us += 33333;  // Increment timestamp for next frame
            
            if ((i + 1) % 10 == 0) {
                std::cout << "Completed " << (i + 1) << "/" << num_iterations << " iterations." << std::endl;
            }
        }
        
        // Calculate and display performance statistics
        double total_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0);
        double avg_time = total_time / num_iterations;
        
        std::sort(inference_times.begin(), inference_times.end());
        double min_time = inference_times.front();
        double max_time = inference_times.back();
        double median_time = inference_times[num_iterations / 2];
        
        // Calculate percentiles
        double p95_time = inference_times[static_cast<int>(num_iterations * 0.95)];
        double p99_time = inference_times[static_cast<int>(num_iterations * 0.99)];
        
        std::cout << "\n=== Performance Results ===" << std::endl;
        std::cout << "Number of iterations: " << num_iterations << std::endl;
        std::cout << "Average inference time: " << avg_time << " ms" << std::endl;
        std::cout << "Median inference time: " << median_time << " ms" << std::endl;
        std::cout << "Min inference time: " << min_time << " ms" << std::endl;
        std::cout << "Max inference time: " << max_time << " ms" << std::endl;
        std::cout << "95th percentile: " << p95_time << " ms" << std::endl;
        std::cout << "99th percentile: " << p99_time << " ms" << std::endl;
        std::cout << "Throughput: " << (1000.0 / avg_time) << " FPS" << std::endl;
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double time : inference_times) {
            variance += (time - avg_time) * (time - avg_time);
        }
        variance /= num_iterations;
        double std_dev = std::sqrt(variance);
        std::cout << "Standard deviation: " << std_dev << " ms" << std::endl;

        // Show results from the last frame for verification
        std::vector<LandmarkList> final_landmarks;
        
        std::cout << "\n=== Final Frame Results ===" << std::endl;
        if (!face_mesh.processFrame(gpu_frame, current_timestamp_us, final_landmarks)) {
            std::cerr << "Failed to process final verification frame." << std::endl;
            return -1;
        }
        
        std::cout << "Found " << final_landmarks.size() << " faces in final frame." << std::endl;
        
        // Print some landmark information from final frame
        for (size_t i = 0; i < final_landmarks.size(); i++) {
            std::cout << "Face " << i << " has " << final_landmarks[i].landmarks.size() << " landmarks" << std::endl;
            if (!final_landmarks[i].landmarks.empty()) {
                std::cout << "  First landmark: (" << final_landmarks[i].landmarks[0].x 
                         << ", " << final_landmarks[i].landmarks[0].y << ", " << final_landmarks[i].landmarks[0].z << ")" << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}