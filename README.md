![docker](https://img.shields.io/badge/docker-19+-white)
![C++](https://img.shields.io/badge/C++-17+-cyan)

![TensorRT](https://img.shields.io/badge/tensorRT-8.6.1-green)
![CudNN](https://img.shields.io/badge/CudNN-8.9.1-orange)


# TensorRT C++ Samples

- real-time inference using TensorRT 

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
root@mat-XPS-15-9560:/src/scripts/tao-converter# export MODEL=~/path/to/folder
root@mat-XPS-15-9560:/src/scripts/tao-converter# ./tao-converter -k ess -t fp16 -e $MODEL/ess.engine -o output $MODEL/ess.etlt

[INFO] ----------------------------------------------------------------
[INFO] Input filename:   /tmp/filer9wcjU
[INFO] ONNX IR version:  0.0.7
[INFO] Opset version:    13
[INFO] Producer name:    pytorch
[INFO] Producer version: 1.10
```