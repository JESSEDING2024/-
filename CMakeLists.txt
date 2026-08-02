cmake_minimum_required(VERSION 3.14)

# 【核心修改点】必须在 project() 之前引入 vcpkg 的工具链文件 (用于加载 OpenCV)
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(whiteboard_pipeline_stage1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ================= 1. 查找通过 vcpkg 安装的 OpenCV =================
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)
find_package(OpenCL REQUIRED)
# ================= 2. 源码级引入 MNN (关闭不需要的周边工具以加快编译) =================
# 强行关闭 MNN 的辅助工具，我们只需要核心推理库
set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_QUANTOOLS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_CONVERTER OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# 引入项目内的 MNN 源码目录
add_subdirectory(third_party/MNN)
# ================= 3. 声明你的主程序 =================
add_executable(whiteboard_pipeline
        src/main.cpp
        src/whiteboardenchance.cpp
        CL/image_process.cl
        src/whiteboardenchance.h
        CL/image_process.h.cpp

)

# ================= 4. 关键修复：暴露头文件并链接库 =================
# 修复 C1083 报错：让编译器知道去哪里找 MNN/Interpreter.hpp
target_include_directories(whiteboard_pipeline PRIVATE third_party/MNN/include)
target_link_libraries(whiteboard_pipeline PRIVATE OpenCL::OpenCL)
# 将 OpenCV 和 MNN 共同链接到你的执行文件上
target_link_libraries(whiteboard_pipeline PRIVATE
        ${OpenCV_LIBS}
        MNN
)

if(MSVC)
    target_compile_options(whiteboard_pipeline PRIVATE /W4 /permissive-)
else()
    target_compile_options(whiteboard_pipeline PRIVATE -Wall -Wextra -Wpedantic)
endif()

#由于clion，默认cl为txt文件
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/CL
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})