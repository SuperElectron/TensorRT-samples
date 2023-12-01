cmake_minimum_required(VERSION 3.24)
project(tensorRTsamples)

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "/src/cmake" ${CMAKE_MODULE_PATH})
# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")
set(TensorRT_DIR /opt/tensorrt)

## Build the YoloV8 library
add_library(YoloV8_TRT SHARED ${CMAKE_SOURCE_DIR}/yolov8.cpp)

# include openCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(YoloV8_TRT PUBLIC ${OpenCV_LIBS})

# Use ccache to speed up rebuilds
include(/src/cmake/ccache.cmake)


# Build the TensorRT inference engine library
# TensorRT is found and linked by the tensorrt-cpp-api
add_subdirectory(${CMAKE_SOURCE_DIR}/tensorrt-cpp-api)


# Link to the library
#target_include_directories(YoloV8_TRT PUBLIC ${OpenCV_INCLUDE_DIRS} ${OpenCV_CUDA_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(YoloV8_TRT PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

target_link_libraries(YoloV8_TRT PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(YoloV8_TRT PUBLIC ${CMAKE_SOURCE_DIR}/tensorrt-cpp-api)

# Build and link the executables
add_executable(detect_object_image ${CMAKE_SOURCE_DIR}/object_detection_image.cpp)
target_link_libraries(detect_object_image YoloV8_TRT)

add_executable(benchmark ${CMAKE_SOURCE_DIR}/benchmark.cpp)
target_link_libraries(benchmark YoloV8_TRT)

add_executable(detect_object_video ${CMAKE_SOURCE_DIR}/object_detection_video_stream.cpp)
target_link_libraries(detect_object_video YoloV8_TRT)

add_executable(driver ${CMAKE_SOURCE_DIR}/driver.cpp)
target_link_libraries(driver tensorrt_cpp_api)
