# ocr_image_processing/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(OCRProcessor CXX)

set(CMAKE_C_COMPILER /usr/bin/clang-20)
set(CMAKE_CXX_COMPILER /usr/bin/clang++-20)
# set(CMAKE_CXX_CLANG_TIDY "/usr/bin/clang-tidy-20;-checks=-*;-modernize=-*")

# --- Find PkgConfig (required to use pkg_check_modules) ---
find_package(PkgConfig REQUIRED)
if (NOT PKGCONFIG_FOUND)
    message(FATAL_ERROR "PkgConfig not found. Please install pkg-config.")
endif()

# --- Find Tesseract using PkgConfig ---
# This command uses the 'tesseract.pc' file installed by libtesseract-dev
pkg_check_modules(TESSERACT REQUIRED tesseract)
if (NOT TESSERACT_FOUND)
    message(FATAL_ERROR "Tesseract not found using pkg-config. Ensure libtesseract-dev is installed and pkg-config works correctly.")
endif()
message(STATUS "Tesseract found using pkg-config. Include dirs: ${TESSERACT_INCLUDE_DIRS}, Libraries: ${TESSERACT_LIBRARIES}")


# --- Find Leptonica using PkgConfig (UPDATED FOR 'lept.pc'!) ---
# The package name for pkg_check_modules is the base name of the .pc file (e.g., 'lept' for 'lept.pc')
pkg_check_modules(LEPTONICA REQUIRED lept) # <--- CHANGE IS HERE!
if (NOT LEPTONICA_FOUND)
    message(FATAL_ERROR "Leptonica not found using pkg-config. Ensure your 'lept.pc' file is in PKG_CONFIG_PATH and pkg-config works correctly.")
endif()
message(STATUS "Leptonica found using pkg-config. Include dirs: ${LEPTONICA_INCLUDE_DIRS}, Libraries: ${LEPTONICA_LIBRARIES}")

# --- Find OpenCV ---
# OpenCV typically installs its own CMake configuration file (OpenCVConfig.cmake),
# so find_package(OpenCV) usually works directly without pkg-config.
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install libopencv-dev.")
endif()
message(STATUS "OpenCV found. Version: ${OpenCV_VERSION}")


# Add executable
add_executable(ocr_processor src/main.cpp)

# Specify include directories for the compiler
# These are gathered from pkg-config for Tesseract and Leptonica, and from OpenCV's find_package
target_include_directories(ocr_processor PRIVATE
    ${TESSERACT_INCLUDE_DIRS}
    ${LEPTONICA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Link the libraries
target_link_libraries(ocr_processor PRIVATE
    ${TESSERACT_LIBRARIES}
    ${LEPTONICA_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

# Set C++ standard
set_property(TARGET ocr_processor PROPERTY CXX_STANDARD 20)
set_property(TARGET ocr_processor PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ocr_processor PROPERTY CXX_EXTENSIONS OFF)

