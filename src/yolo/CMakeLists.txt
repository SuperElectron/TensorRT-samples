###################################

cmake_minimum_required(VERSION 3.24)
project(yolo)

###################################
# set project settings
set(CMAKE_MODULE_PATH "/src/cmake" ${CMAKE_MODULE_PATH})
# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")
set(TensorRT_DIR /opt/tensorrt)
# Use ccache to speed up rebuilds
include(${CMAKE_MODULE_PATH}/ccache.cmake)

###################################
## create a project library and add all shared files
add_library(${PROJECT_NAME} SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/yolov8.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/yolov8.h
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/engine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/engine.h
)

###################################
# include libs


# this requires FindTensorRT.cmake
find_package(TensorRT REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${TensorRT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${TensorRT_LIBRARIES})

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${${OpenCV_CUDA_INCLUDE_DIRS}}
        ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
)

###################################
# create and link executables to shared library
add_executable(detectImage ${CMAKE_CURRENT_SOURCE_DIR}/detectImage.cpp)
target_link_libraries(detectImage ${PROJECT_NAME})

add_executable(profile ${CMAKE_CURRENT_SOURCE_DIR}/profile.cpp)
target_link_libraries(profile ${PROJECT_NAME})

add_executable(detectWebcam ${CMAKE_CURRENT_SOURCE_DIR}/detectWebcam.cpp)
target_link_libraries(detectWebcam ${PROJECT_NAME})
