cmake_minimum_required(VERSION 3.16)

project(ONNXtoTensorRT)

# Set C++ version and optimization level
include(/src/cmake/ccache.cmake)
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# Set the path to TensorRT includes and libraries
if (NOT TensorRT_DIR)
    set(TensorRT_DIR /opt/tensorrt)
endif()

# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)


# Add the executable and library
add_executable(${PROJECT_NAME} onnx2trt.cpp)
set(LIBRARY_NAME ${PROJECT_NAME}_LIB)
add_library(${LIBRARY_NAME} SHARED onnx2trt.cpp)
target_include_directories(${LIBRARY_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME} PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
target_include_directories(${LIBRARY_NAME} PUBLIC .)

# Link against TensorRT libraries
target_include_directories(${PROJECT_NAME} PRIVATE ${TENSORRT_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PRIVATE ${TENSORRT_LIB_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvonnxparser)

# Link against OpenCV
target_include_directories(${PROJECT_NAME} PRIVATE ${OPENCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE opencv_core opencv_highgui opencv_imgcodecs)

# Add any additional libraries if needed
# target_link_libraries(${PROJECT_NAME} PRIVATE ...)
include(${CMAKE_MODULE_PATH}/nvds.cmake)

# Set the output directory for the executable
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
