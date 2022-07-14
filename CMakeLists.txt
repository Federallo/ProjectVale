cmake_minimum_required(VERSION 3.0)
project(projectVale)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 20)

set(EXECUTABLE_NAME projectVale)

include_directories(src)

add_executable(${EXECUTABLE_NAME}
                src/main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)