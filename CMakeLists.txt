cmake_minimum_required(VERSION 3.10)
project(ivt_sic_assits_robot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(ivt_sic_assits_robot running.cpp)
target_link_libraries(ivt_sic_assits_robot PRIVATE realsense2::realsense2 ${OpenCV_LIBS})
