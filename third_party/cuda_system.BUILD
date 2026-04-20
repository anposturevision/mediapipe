# Copyright 2024 The MediaPipe Authors.
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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda_runtime",
    srcs = [
        "targets/x86_64-linux/lib/libcudart.so.12.5.82",
    ],
    hdrs = glob([
        "targets/x86_64-linux/include/**/*.h",
        "targets/x86_64-linux/include/**/*.hpp",
    ]),
    includes = ["targets/x86_64-linux/include"],
    linkopts = [
        "-Ltargets/x86_64-linux/lib",
        "-lcudart",
    ],
)

cc_library(
    name = "cuda_gl_interop",
    hdrs = glob([
        "targets/x86_64-linux/include/cuda_gl_interop.h",
    ]),
    includes = ["targets/x86_64-linux/include"],
    linkopts = [
        "-Ltargets/x86_64-linux/lib",
    ],
    deps = [":cuda_runtime"],
)

cc_library(
    name = "cuda",
    srcs = [
        "targets/x86_64-linux/lib/stubs/libcuda.so",
    ],
    hdrs = glob([
        "targets/x86_64-linux/include/cuda.h",
    ]),
    includes = ["targets/x86_64-linux/include"],
    linkopts = [
        "-Ltargets/x86_64-linux/lib",
        "-lcuda",
    ],
)
