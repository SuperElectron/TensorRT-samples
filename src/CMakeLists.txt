# CMakeLists.txt in the src directory

cmake_minimum_required(VERSION 3.24)
project(MainProject)

# Add engine and yolo as subdirectories
add_subdirectory(engine)
add_subdirectory(yolo)
add_subdirectory(converter)