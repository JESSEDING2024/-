//
// Created by dingj on 8/1/2026.
//
#include "whiteboardenchance.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include <CL/cl.h> // 新增：包含 OpenCL 头文件
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <fstream> // 新增：用于读取 .cl 文件
#include <memory>
#include <mutex>
#include <vector>

// ... 保留原来的 sortCorners, Frame, LatestDoubleBuffer, FpsMeter, Options, captureLoop, inferenceLoop ...
// (为了节省字数，这里省略你已经写好的那些类的代码，直接放新增部分)
static void sortCorners(std::vector<cv::Point2f>& pts) {
    if (pts.size() != 4) return;

    // 按照 y 坐标排序，分出上下两组
    std::sort(pts.begin(), pts.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.y < b.y;
    });

    // 前两个是上方点，根据 x 坐标分出左上、右上
    std::vector<cv::Point> top = {pts[0], pts[1]};
    std::sort(top.begin(), top.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });
    cv::Point tl = top[0];
    cv::Point tr = top[1];

    // 后两个是下方点，根据 x 坐标分出左下、右下
    std::vector<cv::Point> bottom = {pts[2], pts[3]};
    std::sort(bottom.begin(), bottom.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });
    cv::Point bl = bottom[0];
    cv::Point br = bottom[1];

    pts = {tl, tr, br, bl};
}

// ---------------------------------------------------------
// 新增：OpenCL 硬件加速器类
// ---------------------------------------------------------
OpenCLEnhancer::OpenCLEnhancer() {
        // 1. 初始化 OpenCL 环境
        cl_platform_id platform;
        clGetPlatformIDs(1, &platform, NULL);
        cl_device_id device;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
        queue = clCreateCommandQueue(context, device, 0, NULL);

        // 2. 编译内核
        std::ifstream file("CL/image_process.cl");
        if (!file.is_open()) {
            std::cerr << "[Fault] Cant find  image_process.cl file！" << std::endl;
            exit(-1);
        }
        std::string sourceStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        const char* source = sourceStr.c_str();
        program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
        clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        kernel = clCreateKernel(program, "transform_and_brighten", NULL);

        // 3. 灵魂核心：分配零拷贝内存 (CL_MEM_ALLOC_HOST_PTR)
        // 注意：为了应对动态变化的输出尺寸，我们一次性分配足够大的显存，用的时候只用一部分
        cl_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, in_size, NULL, NULL);
        cl_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, out_size, NULL, NULL);
        cl_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 9 * sizeof(float), NULL, NULL);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_input);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_output);
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_matrix);
    }

OpenCLEnhancer::~OpenCLEnhancer() {
        if (cl_input) clReleaseMemObject(cl_input);
        if (cl_output) clReleaseMemObject(cl_output);
        if (cl_matrix) clReleaseMemObject(cl_matrix);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    // 替代原来的 enhanceWhiteboard 函数
cv::Mat OpenCLEnhancer::process(const cv::Mat& src_image, std::vector<cv::Point2f> corners) {
    if (corners.size() != 4 || src_image.empty()) return src_image.clone();

    // 1. 确保角点顺序正确
    sortCorners(corners);

    // 2. 计算目标尺寸，动态计算输出频率
    int out_width = std::max(cv::norm(corners[1] - corners[0]), cv::norm(corners[2] - corners[3]));
    int out_height = std::max(cv::norm(corners[3] - corners[0]), cv::norm(corners[2] - corners[1]));

    if (out_width <= 0 || out_height <= 0 || out_width > max_out_width || out_height > max_out_height) {
        // 如果尺寸异常，给个安全默认值
        out_width = 800;
        out_height = 600;
    }

    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(0, 0), cv::Point2f((float)out_width, 0),
        cv::Point2f((float)out_width, (float)out_height), cv::Point2f(0, (float)out_height)
    };

    // 获取透视变换矩阵 (这里用正向矩阵即可，因为 cv::warpPerspective 内部会自己处理逆向映射)
    cv::Mat matrix = cv::getPerspectiveTransform(corners, dstPoints);

    // ==========================================================
    // 核心更改：将 cv::Mat 转换为 cv::UMat，触发 GPU (OpenCL) 硬件加速
    // ==========================================================
    cv::UMat gpu_src, gpu_warped, gpu_gray, gpu_thresh;

    // 将 CPU 数据拷贝到 GPU
    src_image.copyTo(gpu_src);

    // 1. GPU 上的透视变换 (内部自动调用 OpenCL 优化)
    cv::warpPerspective(gpu_src, gpu_warped, matrix, cv::Size(out_width, out_height),cv::INTER_LANCZOS4);

    // 2. 在GPU上实现的灰度转换
    cv::cvtColor(gpu_warped, gpu_gray, cv::COLOR_BGR2GRAY);

    // 3. GPU 上的自适应阈值 (去阴影二值化，极度消耗算力，利用 GPU 加速效果最好)
    int blockSize = 31;
    double C = 10.0;
    cv::adaptiveThreshold(gpu_gray, gpu_thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, blockSize, C);

    // 4. 将结果转回三通道 (如果你的后续管线需要 BGR 格式)
    cv::UMat gpu_result;
    cv::cvtColor(gpu_thresh, gpu_result, cv::COLOR_GRAY2BGR);

    // 5. 将处理完的数据从 GPU 拉回 CPU (cv::Mat)
    cv::Mat final_bgr;
    gpu_result.copyTo(final_bgr);

    return final_bgr;
}
