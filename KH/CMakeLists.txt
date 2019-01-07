# kheo1772@gmail.com
cmake_minimum_required(VERSION 2.8.3)
project(stitching_images)

find_package(OpenCV REQUIRED)

include_directories(include)
configure_file(include/Config.hpp.in ${CMAKE_BINARY_DIR}/generated/Config.hpp)
include_directories(${CMAKE_BINARY_DIR}/generated/)

set(SOURCES src/main.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
