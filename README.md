![docker](https://img.shields.io/badge/docker-19+-white)
![C++](https://img.shields.io/badge/C++-17+-cyan)

![TensorRT](https://img.shields.io/badge/tensorRT-8.6.1-green)
![CudNN](https://img.shields.io/badge/CudNN-8.9.1-orange)


# TensorRT C++ Samples

Real-time inference using TensorRT.
- convert onnx to trt on target hardware
- run yolo models on target hardware (and it automatically creates the trt engine file)
- run engine file on target hardware


## Getting Started


1. Download a [yolo model](https://github.com/ultralytics/ultralytics)
2. Update the Makefile with your ARCH_BIN (see #reference for details)
3. Start the docker container.
```bash
make build
make run
```

3. Build the code

- place your images in `src/images`

```bash
# inside the docker container
mkdir build && cd build
cmake ..
make -j -l4

```

- alternatively, if you want to build an individual module alone, you can follow these steps

```bash
# go to module of interest
$ cd /src/engine
# create build directory
$ mkdir build && cd build
# build the project
$ cmake .. && make -j
```

__modules__

the main CMakeLists.txt builds these folders:

```text
converter ----> converts yolo model to tensorRT serialized engine file (trt engine file)
engine    ----> runs a trt engine file
yolo      ----> runs a yolo model (converts to trt engine and runs)
```

When you build in the main directory here is what the outputs look like

```bash
/src/build# tree -L 2 -I 'CMakeFiles'
.
|-- CMakeCache.txt
|-- Makefile
|-- cmake_install.cmake
|-- converter
|   |-- Makefile
|   |-- cmake_install.cmake
|   `-- onnx2trt                <<<<<<<<<< convert onnx 2 trt
|-- engine
|   |-- Makefile
|   |-- cmake_install.cmake
|   `-- engine                  <<<<<<<<<< run serialized engine file
`-- yolo
    |-- Makefile
    |-- cmake_install.cmake
    |-- detectImage             <<<<<<<<<< run object detection on an image with yolo model
    |-- detectWebcam            <<<<<<<<<< run object detection on an webcam with yolo model
    |-- libyolo.so
    `-- profile                 <<<<<<<<<< calculate yolo model execution time when doing detection on image
```


# References

__arch_bin__

The Dockerfile has an `ARG` `ARCH_BIN` that is used to build openCV wth cuda support.
You can check [nvidia docs](https://developer.nvidia.com/cuda-gpus) to match your gpu and set ARCH_BIN in the Makefile


```bash
# here we have GeForce GTX 1050. The docs label it as ARCH_BIN=6.1
$ nvidia-smi
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1050        Off | 00000000:01:00.0 Off |                  N/A |
  ...
  ...
+---------------------------------------------------------------------------------------+
```

__version check__

-  check your versions (inside docker container)
```bash

# TensorRT version
$ find / -name NvInferVersion.h -type f
/usr/include/x86_64-linux-gnu/NvInferVersion.h

# this displays TensorRT version 8.6.1
$ cat /usr/include/x86_64-linux-gnu/NvInferVersion.h | grep NV_TENSORRT | head -n 3
#define NV_TENSORRT_MAJOR 8 //!< TensorRT major version.
#define NV_TENSORRT_MINOR 6 //!< TensorRT minor version.
#define NV_TENSORRT_PATCH 1 //!< TensorRT patch version.

# this displays cudNN version 8.9.1
$ cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2 | head -n 3
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 9
#define CUDNN_PATCHLEVEL 1
```

__tao converter__

```bash
# make run puts you inside the docker container

# before running this, check the README.txt in /src/scripts/tao-converter and install any dependencies and set paths
/tmp/tao-converter# export MODEL_PATH=~/path/to/folder
/tmp/tao-converter# export MODEL=replace_with_model_name
/tmp/tao-converter# export KEY=replace_with_nvidia_key
/tmp/tao-converter# ./tao-converter -k "${KEY}" -t fp16 -e "${MODEL_PATH}/${MODEL}.engine" -o output "${MODEL_PATH}/${MODEL}.etlt"

[INFO] ----------------------------------------------------------------
[INFO] Input filename:   /tmp/filer9wcjU
[INFO] ONNX IR version:  0.0.7
[INFO] Opset version:    13
[INFO] Producer name:    pytorch
[INFO] Producer version: 1.10
```

__trtexec__

- ask for help
```bash
$ /usr/src/tensorrt/bin/trtexec --help
```

- profile model speed

```bash
# load in a onnx file
$ export MODEL_PATH=/path/to/folder
$ export ONNX_NAME=model.onnx
$ export TRT_NAME=model.engine
$ /usr/src/tensorrt/bin/trtexec --onnx="${MODEL_PATH}/${ONNX_NAME}" --iterations=5 --workspace=4096
# load in a trt engine file
$ /usr/src/tensorrt/bin/trtexec --loadEngine="${MODEL_PATH}/${TRT_NAME}" --fp16 --batch=1 --iterations=50 --workspace=4096
# save logs to a file
$ /usr/src/tensorrt/bin/trtexec --loadEngine="${MODEL_PATH}/${TRT_NAME}" --fp16 --batch=1 --iterations=50 --workspace=4096 > stats.log 
```

- model conversion

```bash
$ export MODEL_PATH=/path/to/folder
$ export MODEL_NAME=model
# convert the model to FP16 (if supported on hardware)
$ /usr/src/tensorrt/bin/trtexec --onnx="${MODEL_PATH}/${MODEL_NAME}.onnx" --saveEngine="${MODEL_PATH}/${MODEL_NAME}_fp16.engine" --useCudaGraph --fp16 > "${MODEL_NAME}_fp16.log" 
  
```

