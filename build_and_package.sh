
apt update && apt install -y apt-transport-https curl gnupg && \
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor -o /usr/share/keyrings/bazel-archive-keyring.gpg && \
echo "deb [signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
apt update && apt install -y bazel bazel-6.5.0

bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11  mediapipe/examples/custom:librun_graph_gpu.so