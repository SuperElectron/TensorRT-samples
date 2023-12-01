# syntax = docker/dockerfile:1.2
ARG ARCH=x86_64
ARG ARCH_BIN=6.1

# Build stage for aarch64 (arm64)
FROM nvcr.io/nvidia/tensorrt:23.10-py3

MAINTAINER "Matthew McCann <matmccann@gmail.com>"
ENV DEBIAN_FRONTEND noninteractive

RUN python3 -m pip install ultralytics

# Install apt dependencies
RUN apt-get update -yqq && apt-get install -y \
    libopencv-dev \
    build-essential \
    cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libatlas-base-dev gfortran python3-dev \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev \
    libtiff-dev libdc1394-22-dev

# Clone OpenCV and OpenCV-contrib repositories
RUN cd /tmp && git clone https://github.com/opencv/opencv.git
RUN cd /tmp && git clone https://github.com/opencv/opencv_contrib.git

# Navigate to the OpenCV build directory
RUN mkdir -p /tmp/opencv/build && \
    cd /tmp/opencv/build && \
    cmake .. \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_EXAMPLES=OFF \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D CUDA_ARCH_BIN="${ARCH_BIN}" \
      -D CUDA_ARCH_PTX="${ARCH_BIN}"

# Compile and install OpenCV \
RUN cd /tmp/opencv/build && \
    make -j$(nproc) && \
    make install
# Add OpenCV shared libraries to the library path
RUN sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
RUN ldconfig

# Clean up
RUN rm -rf /tmp/opencv /tmp/opencv_contrib

CMD "bash"