#include "mediapipe/examples/custom/run_graph_gpu.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace {

// Returns true if any element of the matrix is non-zero. The wrapper fills
// transform_matrices[i] with all zeros when face_geometry produced nothing
// for that face on that frame.
bool matrix_is_populated(const FaceTransformMatrix& m) {
    for (float v : m) {
        if (v != 0.0f) return true;
    }
    return false;
}

// Decomposes a row-major 4x4 rigid transform's top-left 3x3 rotation into
// (pitch, yaw, roll) Euler angles in degrees. Convention:
//   pitch = X-rotation, yaw = Y-rotation, roll = Z-rotation
// Mirrors the math used in test_camera_angle.py for direct comparison.
void decompose_euler_deg(const FaceTransformMatrix& m,
                         double& pitch_deg, double& yaw_deg, double& roll_deg) {
    // Row-major 3x3 rotation: R[r][c] = m[r*4 + c]
    const double r00 = m[0],  r01 = m[1],  r02 = m[2];
    const double r10 = m[4],  r11 = m[5];
    const double r20 = m[8],  r21 = m[9],  r22 = m[10];

    const double sy = std::sqrt(r00 * r00 + r10 * r10);
    double pitch, yaw, roll;
    if (sy > 1e-6) {
        pitch = std::atan2(r21, r22);
        yaw   = std::atan2(-r20, sy);
        roll  = std::atan2(r10, r00);
    } else {
        // Gimbal lock — yaw near ±90°
        pitch = std::atan2(-r01 /*=R[1][2] when col-major*/ , r11);
        yaw   = std::atan2(-r20, sy);
        roll  = 0.0;
        (void)r02;
    }
    constexpr double kRad2Deg = 180.0 / M_PI;
    pitch_deg = pitch * kRad2Deg;
    yaw_deg   = yaw   * kRad2Deg;
    roll_deg  = roll  * kRad2Deg;
}

void print_matrix(const FaceTransformMatrix& m) {
    std::cout << std::fixed << std::setprecision(4);
    for (int r = 0; r < 4; ++r) {
        std::cout << "  [ ";
        for (int c = 0; c < 4; ++c) {
            std::cout << std::setw(10) << m[r * 4 + c];
            if (c < 3) std::cout << ", ";
        }
        std::cout << " ]" << std::endl;
    }
    std::cout.unsetf(std::ios::fixed);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        // Optional CLI args:
        //   argv[1] = path to the .pbtxt graph (default: upstream copy)
        //   argv[2] = path to the test image  (default: bundled far_face.jpg)
        // Pass the ai_inference graph file here to exercise the face_geometry
        // pipeline without modifying the upstream vanilla graph.
        std::string graph_config = (argc > 1)
            ? argv[1]
            : "/mediapipe/mediapipe/graphs/face_mesh/face_mesh_gpu_lib.pbtxt";
        std::string image_path = (argc > 2)
            ? argv[2]
            : "/mediapipe/mediapipe/examples/custom/far_face.jpg";

        std::cout << "Graph config: " << graph_config << std::endl;
        std::cout << "Test image:   " << image_path << std::endl;

        FacemeshGraphRunner face_mesh;
        std::cout << "Initializing face mesh graph..." << std::endl;
        if (!face_mesh.initGraph(graph_config)) {
            std::cerr << "Failed to initialize the face mesh graph." << std::endl;
            return -1;
        }
        std::cout << "Face mesh graph initialized successfully." << std::endl;

        cv::Mat camera_frame = cv::imread(image_path);
        if (camera_frame.empty()) {
            std::cerr << "Failed to read the image file." << std::endl;
            return -1;
        }
        std::cout << "Image loaded successfully. Size: " << camera_frame.cols << "x" << camera_frame.rows << std::endl;

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
        
        // Warmup runs (matrix-aware overload — exercises the face_geometry pipeline)
        for (int i = 0; i < warmup_iterations; i++) {
            std::vector<LandmarkList> landmarks;
            std::vector<FaceTransformMatrix> matrices;
            if (!face_mesh.processFrame(camera_frame, current_timestamp_us, landmarks, matrices)) {
                std::cerr << "Failed to process frame during warmup iteration " << i << std::endl;
                return -1;
            }
            if (i == 0) {
                std::cout << "Warmup: Found " << landmarks.size() << " face(s) and "
                          << matrices.size() << " geometry matri" << (matrices.size() == 1 ? "x" : "ces")
                          << " in first frame." << std::endl;
            }
            current_timestamp_us += 33333;  // ~30 FPS spacing
        }
        std::cout << "Warmup completed." << std::endl;

        // Timed inference runs — also count how often face_geometry yielded a
        // populated matrix, so we can sanity-check the geometry stream stays
        // in sync with the landmarks stream over many frames.
        int frames_with_geometry = 0;
        for (int i = 0; i < num_iterations; i++) {
            std::vector<LandmarkList> landmarks;
            std::vector<FaceTransformMatrix> matrices;

            auto start_time = std::chrono::high_resolution_clock::now();

            if (!face_mesh.processFrame(camera_frame, current_timestamp_us, landmarks, matrices)) {
                std::cerr << "Failed to process frame during iteration " << i << std::endl;
                return -1;
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double inference_time_ms = duration.count() / 1000.0;
            inference_times.push_back(inference_time_ms);

            if (!matrices.empty() && matrix_is_populated(matrices[0])) {
                frames_with_geometry++;
            }

            current_timestamp_us += 33333;

            if ((i + 1) % 10 == 0) {
                std::cout << "Completed " << (i + 1) << "/" << num_iterations
                          << " iterations  (geometry hits so far: "
                          << frames_with_geometry << ")" << std::endl;
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

        std::cout << "\nGeometry hit rate: " << frames_with_geometry
                  << " / " << num_iterations
                  << " (" << (100.0 * frames_with_geometry / num_iterations) << "%)" << std::endl;

        // Show results from the last frame for verification
        std::vector<LandmarkList> final_landmarks;
        std::vector<FaceTransformMatrix> final_matrices;

        std::cout << "\n=== Final Frame Results ===" << std::endl;
        if (!face_mesh.processFrame(camera_frame, current_timestamp_us, final_landmarks, final_matrices)) {
            std::cerr << "Failed to process final verification frame." << std::endl;
            return -1;
        }

        std::cout << "Found " << final_landmarks.size() << " faces in final frame." << std::endl;
        std::cout << "Found " << final_matrices.size() << " transform matri"
                  << (final_matrices.size() == 1 ? "x" : "ces") << " in final frame." << std::endl;

        // Print some landmark information from final frame
        for (size_t i = 0; i < final_landmarks.size(); i++) {
            std::cout << "Face " << i << " has " << final_landmarks[i].landmarks.size() << " landmarks" << std::endl;
            if (!final_landmarks[i].landmarks.empty()) {
                std::cout << "  First landmark: (" << final_landmarks[i].landmarks[0].x
                         << ", " << final_landmarks[i].landmarks[0].y << ", " << final_landmarks[i].landmarks[0].z << ")" << std::endl;
            }
        }

        // Print the face_geometry pose transform matrix and its decomposition.
        std::cout << "\n=== Face Geometry Results ===" << std::endl;
        if (final_matrices.empty()) {
            std::cout << "No transform matrix returned. Possible causes:" << std::endl;
            std::cout << "  * graph wasn't built with face_geometry wired in" << std::endl;
            std::cout << "  * face was not detected on the final frame" << std::endl;
            std::cout << "  * geometry_pipeline_metadata_landmarks.binarypb missing at" << std::endl;
            std::cout << "    mediapipe/modules/face_geometry/data/geometry_pipeline_metadata_landmarks.binarypb" << std::endl;
        } else {
            for (size_t i = 0; i < final_matrices.size(); i++) {
                const auto& m = final_matrices[i];
                std::cout << "Face " << i << " pose transform matrix (row-major 4x4):" << std::endl;
                print_matrix(m);

                if (matrix_is_populated(m)) {
                    double pitch_deg = 0, yaw_deg = 0, roll_deg = 0;
                    decompose_euler_deg(m, pitch_deg, yaw_deg, roll_deg);
                    std::cout << std::fixed << std::setprecision(2);
                    std::cout << "  Decomposed Euler angles (degrees):" << std::endl;
                    std::cout << "    pitch (X): " << std::setw(8) << pitch_deg << std::endl;
                    std::cout << "    yaw   (Y): " << std::setw(8) << yaw_deg   << std::endl;
                    std::cout << "    roll  (Z): " << std::setw(8) << roll_deg  << std::endl;
                    std::cout << "  Translation (camera-space, MediaPipe units ≈ cm):" << std::endl;
                    std::cout << "    tx: " << std::setw(8) << m[3]  << std::endl;
                    std::cout << "    ty: " << std::setw(8) << m[7]  << std::endl;
                    std::cout << "    tz: " << std::setw(8) << m[11] << std::endl;
                    std::cout.unsetf(std::ios::fixed);
                } else {
                    std::cout << "  Matrix is all zeros — face_geometry produced no result for this face." << std::endl;
                }
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