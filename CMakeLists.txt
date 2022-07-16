cmake_minimum_required(VERSION 3.0)
project(projectVale)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 20)

set(EXECUTABLE_NAME projectVale)

include_directories(src)

add_executable(${EXECUTABLE_NAME}
                src/main.cpp
                src/NeuralNetwork.h
                src/NeuralNetwork.cpp)

target_link_libraries(${EXECUTABLE_NAME} Eigen3::Eigen)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)