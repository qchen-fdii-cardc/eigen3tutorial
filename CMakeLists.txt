cmake_minimum_required(VERSION 3.10)
project(eigen_tutorial)

# Find Eigen package
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${EIGEN3_INCLUDE_DIR})

# extra 
add_executable(eigen3version eigen3version.cpp)

# Chapter 1 examples
add_executable(basic_matrix chp01/basic_matrix.cpp)
add_executable(matrix_arithmetic chp01/matrix_arithmetic.cpp)
add_executable(special_matrices chp01/special_matrices.cpp)

# Chapter 2 examples
add_executable(matrix_operations chp02/matrix_operations.cpp)
add_executable(matrix_decompositions chp02/matrix_decompositions.cpp)

# Chapter 3 examples
add_executable(vector_operations chp03/vector_operations.cpp)
add_executable(vector_advanced chp03/vector_advanced.cpp)

# Chapter 4 examples
add_executable(linear_equations chp04/linear_equations.cpp)
add_executable(least_squares chp04/least_squares.cpp)

# Chapter 5 examples
add_executable(eigenvalues chp05/eigenvalues.cpp)
add_executable(pca_example chp05/pca_example.cpp)

# Chapter 6 examples
add_executable(transforms chp06/transforms.cpp)
add_executable(rotations chp06/rotations.cpp)

# Chapter 7 examples
add_executable(sparse_matrix chp07/sparse_matrix.cpp)
add_executable(advanced_features chp07/advanced_features.cpp)

# Link Eigen to all targets
foreach(target
    eigen3version
    basic_matrix matrix_arithmetic special_matrices
    matrix_operations matrix_decompositions
    vector_operations vector_advanced
    linear_equations least_squares
    eigenvalues pca_example
    transforms rotations
    sparse_matrix advanced_features)
    target_link_libraries(${target} Eigen3::Eigen)
endforeach() 


message(STATUS "BUILD_TYPE: ${CMAKE_BUILD_TYPE}")