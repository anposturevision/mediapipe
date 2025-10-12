# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = [
    "//visibility:public",
])

cc_library(
    name = "run_graph_gpu_lib",
    srcs = ["run_graph_gpu.cc"],
    hdrs = ["run_graph_gpu.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/log:absl_log",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/deps:no_destructor",
        "//mediapipe/framework/deps:model_lifecycle_manager",
        "//mediapipe/framework/deps:scoped_model_resources",
        "//mediapipe/framework/deps:scoped_options_registry",
        "@cuda_system//:cuda_runtime",
        "@cuda_system//:cuda_gl_interop",
    ],
    alwayslink = 1,
)

# Shared library with all dependencies included
cc_binary(
    name = "librun_graph_gpu.so",
    linkshared = 1,
    deps = [
        ":run_graph_gpu_lib",
        "//mediapipe/graphs/hand_tracking:mobile_calculators",
        "//mediapipe/graphs/face_mesh:mobile_calculators",
        "@cuda_system//:cuda_runtime",
        "@cuda_system//:cuda_gl_interop",
    ],
    linkopts = [
        "-L/usr/local/cuda-12.5/targets/x86_64-linux/lib",
        "-lcudart",
        "-lcuda",
    ],
)

# Example binary that uses the hand tracking library
cc_binary(
    name = "hand_landmark_example",
    srcs = ["hand_landmark_simple_example.cc"],
    data = [
        "//mediapipe/graphs/hand_tracking:hand_tracking_lib.pbtxt",
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    deps = [
        ":run_graph_gpu_lib",
        "//mediapipe/graphs/hand_tracking:mobile_calculators",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "@com_google_absl//absl/log:absl_log",
    ],
)

# Example binary that uses the face mesh library
cc_binary(
    name = "face_mesh_example",
    srcs = ["face_mesh_example.cc"],
    data = [
        "//mediapipe/graphs/face_mesh:face_mesh_gpu_lib.pbtxt",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_detection:face_detection_full_range.tflite",
    ],
    deps = [
        ":run_graph_gpu_lib",
        "//mediapipe/graphs/face_mesh:mobile_calculators",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "@com_google_absl//absl/log:absl_log",
        "@cuda_system//:cuda_runtime",
        "@cuda_system//:cuda_gl_interop",
    ],
    linkopts = [
        "-L/usr/local/cuda-12.5/targets/x86_64-linux/lib",
        "-lcudart",
        "-lcuda",
    ],
)
