# Description:
#   OpenCV prebuilt libraries for video/image processing

licenses(["notice"])  # BSD license

# Prebuilt OpenCV libraries
cc_library(
    name = "opencv",
    hdrs = glob([
        "include/opencv4/opencv2/**/*.h*",
    ]),
    includes = [
        "include/opencv4",
    ],
    srcs = glob([
        "lib/libopencv_core.so*",
        "lib/libopencv_imgproc.so*",
        "lib/libopencv_imgcodecs.so*", 
        "lib/libopencv_highgui.so*",
        "lib/libopencv_calib3d.so*",
        "lib/libopencv_features2d.so*",
        "lib/libopencv_video.so*",
        "lib/libopencv_videoio.so*",
    ]),
    visibility = ["//visibility:public"],
)
