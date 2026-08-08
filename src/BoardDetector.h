//
// Created by dingj on 2/8/2026.
//

#ifndef WHITEBOARD_PIPELINE_STAGE1_BOARDDETECTOR_H
#define WHITEBOARD_PIPELINE_STAGE1_BOARDDETECTOR_H
#include <iostream>
#include <ostream>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <NvInfer.h>
#include "string"
#include <memory>
#include <fstream>
#include <cuda_runtime_api.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>



#include <opencv2/opencv.hpp>
class TensortLogger:public nvinfer1::ILogger {
    //Severity severity表示当前日志的严重级别（枚举类型）
    //const char* msg：表示具体的日志文本内容
public:

    void log(Severity severity, const char* msg) noexcept override;
}globle_logger; //instantiation
/*
多态允许你用同一个“基类指针”，去指挥不同的“子类对象”干活。
// 声明一个基类的指针，但实际指向子类对象
BoardDetector* detector1 = new YoloDetector();
BoardDetector* detector2 = new TraditionalDetector();
// 编译器会在【运行的时候】自动判断底层到底是什么对象
// 下面这句会自动调用 YoloDetector 里的 detect()
detector1->detect();
// 下面这句会自动调用 TraditionalDetector 里的 detect()
detector2->detect();
*/
class BoardDetector {
    public:   // //pure virtual fucntion ,subclass must offer the real code
        virtual ~BoardDetector() =default;
        virtual bool init(const std::string& modelpath) = 0; //init the model
        virtual std::vector<cv::Point2f> detectCorners(const cv::Mat& frame) = 0;

};

class TensortDetector:public BoardDetector {

private:
         nvinfer1::ICudaEngine* engine_ = nullptr;
         nvinfer1::IExecutionContext* context_ = nullptr;
         nvinfer1::IRuntime* runtime_ = nullptr;
         size_t input_size = 0;
         size_t output_size = 0;
         void* device_buffer[2] = {nullptr,nullptr}; //ptr 数组
         float* host_input_buffer_ = nullptr;
         float* host_output_buffer_ = nullptr;
         cudaStream_t stream_ = nullptr;
         const int model_width = 640;
         const int model_height = 640;
         cv::Mat RGB_frame,float_frame;

public:
    // 定义存放结果的结构体

    bool init(const std:: string& modelpath) override;

    std::vector<cv::Point2f> detectCorners(const cv::Mat& frame) override;
 };



class MNNOpenCLDetector : public BoardDetector {
private:
    std::shared_ptr<MNN::Interpreter> net_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;

    // 【修改点1】：将预处理处理器提出来，作为类成员长期持有
    std::shared_ptr<MNN::CV::ImageProcess> pretreat_;

    const int model_w = 640;
    const int model_h = 640;

public:
    bool init(const std::string& modelPath) override;
    std::vector<cv::Point2f> detectCorners(const cv::Mat& frame) override;
    ~MNNOpenCLDetector() override {
        // pretreat_, net_ 都是智能指针，会自动释放
    }
};


std::unique_ptr<BoardDetector> detector_factory(std::string type);


#endif //WHITEBOARD_PIPELINE_STAGE1_BOARDDETECTOR_H
