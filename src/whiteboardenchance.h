//
// Created by dingj on 8/1/2026.
//

#ifndef WHITEBOARD_PIPELINE_STAGE1_WHITEBOARDENCHANCE_H
#define WHITEBOARD_PIPELINE_STAGE1_WHITEBOARDENCHANCE_H
#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// 如果你使用的是 OpenCL，通常需要包含这个：
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class OpenCLEnhancer {

private:

    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem cl_input = nullptr;
    cl_mem cl_output = nullptr;
    cl_mem cl_matrix = nullptr;

    int max_in_width = 1920;  // 预分配最大输入尺寸
    int max_in_height = 1080;
    int max_out_width = 1920; // 预分配最大输出尺寸
    int max_out_height = 1080;

    size_t in_size = max_in_width * max_in_height * 4;
    size_t out_size = max_out_width * max_out_height * 4;
public:

    OpenCLEnhancer();
    ~OpenCLEnhancer();
    cv::Mat process(const cv::Mat& src_image, std::vector<cv::Point2f> corners);


};


// 你的 class WhiteboardEnhancer { ... }; 定义在下面
#endif //WHITEBOARD_PIPELINE_STAGE1_WHITEBOARDENCHANCE_H



