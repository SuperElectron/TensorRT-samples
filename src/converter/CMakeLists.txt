###################################

cmake_minimum_required(VERSION 3.24)
project(onnx2trt)

###################################
# set project settings
set(CMAKE_MODULE_PATH "/src/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# Set the path to TensorRT includes and libraries
set(TensorRT_DIR /opt/tensorrt)
# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
# Use ccache to speed up rebuilds
include(${CMAKE_MODULE_PATH}/ccache.cmake)

###################################
# Add the executable and library
add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/onnx2trt.cpp)

###################################
# Include libraries

find_package(CUDA REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CUDA_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
)

target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvonnxparser)


### Set the output directory for the executable
#set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
