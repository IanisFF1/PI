cmake_minimum_required(VERSION 3.14)
project(Proiect1)

set(CMAKE_CXX_STANDARD 17)


set(OpenCV_DIR "C:/Users/IanisFatFrumos/Downloads/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


link_directories(${OpenCV_DIR}/x64/vc16/lib)
set(OpenCV_LIBRARIES opencv_world4110)


add_executable(Proiect1
        main.cpp
        PlateDetection.cpp
        PlateDetection.h
)

target_link_libraries(Proiect1 ${OpenCV_LIBRARIES})


set_target_properties(Proiect1 PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Proiect1"
)
