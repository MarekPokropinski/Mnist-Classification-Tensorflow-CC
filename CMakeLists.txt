# CMakeLists.txt
cmake_minimum_required(VERSION 3.21)
set (CMAKE_CXX_STANDARD 17)
project(TF_CLASSIFICATION)

find_package(TensorflowCC REQUIRED)


include_directories(src)

file(GLOB SOURCES "src/*.cpp")
add_executable(main ${SOURCES})

# Link the Tensorflow library.
target_link_libraries(main TensorflowCC::TensorflowCC)

# You may also link cuda if it is available.
find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(main ${CUDA_LIBRARIES})
endif()
