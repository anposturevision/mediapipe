#!/bin/bash

# MediaPipe Build and Package Script
# This script builds the necessary MediaPipe components and copies essential files

set -e  # Exit on any error

# Configuration
BUILD_FLAGS="-c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 --define OPENCV=prebuilt_folder"
PACKAGE_DIR="/tmp/mediapipe_package"

echo "=========================================="
echo "MediaPipe Build and Package Script"
echo "=========================================="
echo "Package destination: $PACKAGE_DIR"
echo "Protobuf path: $PROTOBUF_ROOT"
echo "Build flags: $BUILD_FLAGS"
echo "=========================================="

# Function to print status
print_status() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Clean and create package directory
print_status "Creating package directory..."
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"/{lib,include/mediapipe/{framework/{port,tool,formats,deps},gpu,util}}

# Step 1: Verify system protobuf installation
print_status "Verifying system protobuf installation..."
if [ ! -d "$PROTOBUF_INCLUDE" ] || [ ! -d "$PROTOBUF_LIB" ]; then
    print_status "❌ ERROR: Protobuf not found at $PROTOBUF_ROOT"
    print_status "Please set PROTOBUF_ROOT to your protobuf installation path"
    exit 1
fi

if [ -f "$PROTOBUF_INCLUDE/google/protobuf/message.h" ]; then
    print_status "✓ Found protobuf headers at $PROTOBUF_INCLUDE"
else
    print_status "❌ ERROR: Protobuf headers not found"
    exit 1
fi

if ls "$PROTOBUF_LIB"/libprotobuf.* >/dev/null 2>&1; then
    print_status "✓ Found protobuf libraries at $PROTOBUF_LIB"
else
    print_status "❌ ERROR: Protobuf libraries not found"
    exit 1
fi

# Step 2: Build MediaPipe with system protobuf (skip protobuf targets)
print_status "Building MediaPipe with system protobuf..."
bazel build $BUILD_FLAGS \
    //mediapipe/framework:calculator_framework \
    //mediapipe/framework/formats:image_frame \
    //mediapipe/framework/formats:image_frame_opencv

# Step 2.1: Build GPU components
print_status "Building GPU components..."
bazel build $BUILD_FLAGS \
    //mediapipe/gpu:gl_calculator_helper \
    //mediapipe/gpu:gpu_buffer \
    //mediapipe/gpu:gpu_shared_data_internal \
    //mediapipe/gpu:gpu_service || echo "Warning: Some GPU components failed to build"

# Step 3: Build your custom library
print_status "Building custom run_graph_gpu_lib..."
bazel build $BUILD_FLAGS //mediapipe/examples/custom:run_graph_gpu_lib

# Step 4: Copy built library
print_status "Copying built library..."
cp bazel-bin/mediapipe/examples/custom/librun_graph_gpu_lib.so "$PACKAGE_DIR/lib/"

# Step 4.1: Copy protobuf libraries that contain the implementations
print_status "Copying protobuf libraries..."
find bazel-bin/mediapipe/framework/formats -name "*image_format*proto*.so" -exec cp {} "$PACKAGE_DIR/lib/" \; 2>/dev/null || true
find bazel-bin/mediapipe/framework -name "*calculator*proto*.so" -exec cp {} "$PACKAGE_DIR/lib/" \; 2>/dev/null || true

# Step 5: Copy main header
print_status "Copying main header..."
cp mediapipe/examples/custom/run_graph_gpu.h "$PACKAGE_DIR/include/"

# Step 6: Copy MediaPipe headers
print_status "Copying MediaPipe headers..."

# Copy framework headers
find mediapipe/framework -maxdepth 1 -name "*.h" -exec cp {} "$PACKAGE_DIR/include/mediapipe/framework/" \; 2>/dev/null || true
cp mediapipe/framework/port/*.h "$PACKAGE_DIR/include/mediapipe/framework/port/" 2>/dev/null || true
cp mediapipe/framework/tool/*.h "$PACKAGE_DIR/include/mediapipe/framework/tool/" 2>/dev/null || true
cp mediapipe/framework/formats/*.h "$PACKAGE_DIR/include/mediapipe/framework/formats/" 2>/dev/null || true
cp mediapipe/framework/deps/*.h "$PACKAGE_DIR/include/mediapipe/framework/deps/" 2>/dev/null || true
cp mediapipe/gpu/*.h "$PACKAGE_DIR/include/mediapipe/gpu/" 2>/dev/null || true
cp mediapipe/util/*.h "$PACKAGE_DIR/include/mediapipe/util/" 2>/dev/null || true

# Step 6.1: Copy system protobuf headers
print_status "Copying system protobuf headers..."
mkdir -p "$PACKAGE_DIR/include/google"
if [ -d "$PROTOBUF_INCLUDE/google/protobuf" ]; then
    cp -r "$PROTOBUF_INCLUDE/google/protobuf" "$PACKAGE_DIR/include/google/"
    print_status "✓ System protobuf headers copied"
fi

# Step 6.2: Copy system protobuf libraries
print_status "Copying system protobuf libraries..."
find "$PROTOBUF_LIB" -name "libprotobuf*.so*" -exec cp {} "$PACKAGE_DIR/lib/" \; 2>/dev/null || true
find "$PROTOBUF_LIB" -name "libprotobuf*.a" -exec cp {} "$PACKAGE_DIR/lib/" \; 2>/dev/null || true

# Copy additional framework subdirectory headers
print_status "Copying additional framework headers..."
for subdir in stream_handler packet_generator status_handler; do
    if [ -d "mediapipe/framework/$subdir" ]; then
        mkdir -p "$PACKAGE_DIR/include/mediapipe/framework/$subdir"
        cp mediapipe/framework/$subdir/*.h "$PACKAGE_DIR/include/mediapipe/framework/$subdir/" 2>/dev/null || true
    fi
done

# Step 7: Copy any generated MediaPipe-specific protobuf headers (if any exist)
print_status "Copying MediaPipe-specific generated headers..."
# Only copy MediaPipe-specific generated headers, not core protobuf
find bazel-bin/mediapipe -name "*.pb.h" -type f | while read -r header; do
    # Skip if it's a core protobuf file (we're using system protobuf)
    if [[ "$header" != *"/google/protobuf/"* ]]; then
        rel_path="${header#bazel-bin/mediapipe/}"
        target_dir="$PACKAGE_DIR/include/mediapipe/$(dirname "$rel_path")"
        mkdir -p "$target_dir"
        cp "$header" "$target_dir/" 2>/dev/null || true
    fi
done

# Step 8: Summary
print_status "Build and packaging complete!"
echo ""
echo "Package location: $PACKAGE_DIR"
echo "Library: $(ls -la $PACKAGE_DIR/lib/)"
echo "Headers: $(find $PACKAGE_DIR/include -name "*.h" | wc -l) files"
echo "Protobuf headers: $(find $PACKAGE_DIR/include -name "*.pb.h" | wc -l) files"
echo "System protobuf: $PROTOBUF_ROOT"
echo ""

# Step 8.5: Verify critical headers are present
print_status "Verifying critical headers..."
critical_headers=(
    "google/protobuf/message.h"
    "run_graph_gpu.h"
    "mediapipe/framework/calculator_framework.h"
)

for header in "${critical_headers[@]}"; do
    if [ -f "$PACKAGE_DIR/include/$header" ]; then
        print_status "✓ $header - Found"
    else
        print_status "✗ $header - Missing"
    fi
done

echo ""
print_status "Note: This package uses system protobuf from $PROTOBUF_ROOT"
print_status "Make sure your target system has compatible protobuf installed"
print_status "Done!"
