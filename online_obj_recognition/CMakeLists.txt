cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)
 
project(PCL_openni_viewer)
 
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native")
set(CAFFE_INCLUDEDIR /home/sunting/Documents/program/3D_recognition/caffe-master/include /home/sunting/Documents/program/3D_recognition/caffe-master/distribute/include)
set(CAFFE_LINK_LIBRARAY /home/sunting/Documents/program/3D_recognition/caffe-master/build/lib)

find_package(PCL 1.7 REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)
 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(${PROTOBUF_INCLUDE_DIR})
include_directories(${OPENCV_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIR})
set(PCL_BUILD_TYPE Release)
 
include_directories(/usr/local/cuda-7.5/targets/x86_64-linux/include)
link_directories(/usr/lib/x86_64-linux-gnu /usr/local/hdf5/lib)

include_directories(${CAFFE_INCLUDEDIR})
link_directories(${CAFFE_LINK_LIBRARAY} ${HDF5_HL_LIBRARIES} ${PCL_LIBRARY_DIRS})

file(GLOB PCL_openni_viewer_SRC
    "src/*.h"
    "src/*.cpp"
    "src/*.hpp"
)
add_executable(openniViewer ${PCL_openni_viewer_SRC})
 
target_link_libraries(openniViewer ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBRARIES} ${PROTOBUF_LIBRARY} ${HDF5_HL_LIBRARIES} ${PCL_LIBRARIES} caffe glog)
