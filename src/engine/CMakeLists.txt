###################################

cmake_minimum_required(VERSION 3.24)
enable_language(C CXX)
project(engine)

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "/src/cmake" ${CMAKE_MODULE_PATH})
# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")
# include trt path
set(TensorRT_DIR /opt/tensorrt)
# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# Use ccache to speed up rebuilds
include(/src/cmake/ccache.cmake)

##################################################

# get this module's files
file(GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

file(GLOB SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${SRC} ${HEADERS})

###################################
# Include libraries

# this requires FindTensorRT.cmake
find_package(TensorRT REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${TensorRT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${TensorRT_LIBRARIES})

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
)
