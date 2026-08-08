//
// Created by dingj on 2/8/2026.
//
#include "BoardDetector.h"

//factory model
std::unique_ptr<BoardDetector> detector_factory(const std::string& type) {
      if ("tensortRT" == type) {
           return std::make_unique<TensortDetector>();
      }else if("MNN" == type){
           return std::make_unique<MNNOpenCLDetector>();
      }else {
          //return nullptr;
           return nullptr;
      }
}

void TensortLogger:: log(Severity severity, const char* msg) noexcept {
    if (severity <=Severity::kWARNING) std:: cerr<< msg << std::endl;
}

struct PoseResult {
    cv::Rect2f box;
    float score;
    std::vector<cv::Point2f> corners;
};

// 解析 YOLOv8-Pose 输出 Tensor 的核心逻辑,对预测框进行处理
// out_ptr 就是你从 GPU 拷回 CPU 的 host_output_buffer_
PoseResult ParseYOLOv8Pose(const float* out_ptr, int num_anchors = 8400, float conf_threshold = 0.5f) {
    PoseResult best_result;
    best_result.score = 0.0f;

    // 遍历所有 8400 个预测框
    for (int i = 0; i < num_anchors; ++i) {
        float score = out_ptr[i * 17 + 4]; // 第 5 个值是类别置信度 (白板的概率)

        // 过滤低置信度，并且只保留得分最高的那一个白板
        if (score > conf_threshold && score > best_result.score) {
            best_result.score = score;

            // 解析 Bounding Box (cx, cy, w, h)
            float cx = out_ptr[i * 17 + 0];
            float cy = out_ptr[i * 17 + 1];
            float w  = out_ptr[i * 17 + 2];
            float h  = out_ptr[i * 17 + 3];
            best_result.box = cv::Rect2f(cx - w / 2, cy - h / 2, w, h);

            // 解析 4 个角点坐标
            best_result.corners.clear();
            for (int k = 0; k < 4; ++k) {
                // 跳过前 5 个值 (box+score)，每个关键点占 3 个值 (x, y, conf)
                float kpt_x = out_ptr[i * 17 + 5 + k * 3 + 0];
                float kpt_y = out_ptr[i * 17 + 5 + k * 3 + 1];
                // float kpt_conf = out_ptr[i * 17 + 5 + k * 3 + 2]; // 关键点自身的置信度，这里先省略判断

                best_result.corners.push_back(cv::Point2f(kpt_x, kpt_y));
            }
        }
    }
    return best_result;
}
 bool TensortDetector::init(const std:: string& modelpath) {
         //1.读取编译好的AI模型加载到内存中数据流中
         std::ifstream modelFile(modelpath,std::ios::binary);                     //input file stream
         if (!modelFile.good()) {
             std::cerr << "Could not open model file " << modelpath << std::endl;
             return false;
         }
         //获取模型大小
         modelFile.seekg(0,modelFile.end);
         size_t modelFileSize = modelFile.tellg();
         modelFile.seekg(0,modelFile.beg);
         //2.define a char stream，序列化存储
         char* rtModelstream = new char[modelFileSize];
         modelFile.read(rtModelstream,modelFileSize);
         modelFile.close();
         //3.进入反序列化
         runtime_ = nvinfer1::createInferRuntime(globle_logger);
         engine_ =  runtime_->deserializeCudaEngine(rtModelstream,modelFileSize); //load on the GPU.engine is the real model run on the GPU
         delete rtModelstream;
         //4.创建上下文存储变量
         context_ = engine_->createExecutionContext();
         //5. 分配 CUDA 显存 (假设 1x3x640x640 输入，1x17x8400 输出)
         input_size = 1*3*640*640*sizeof(float); //输入的字节数
         output_size = 1*17*8400*sizeof(float);  //输出的字节数

         cudaMalloc(&device_buffer[0], input_size);
         cudaMalloc(&device_buffer[1], output_size); //指针的指针
          //创造内存cpu去接收GPU处理过的 ，创建内存buffer接收opencv处理过的
         host_output_buffer_ = new float[1*17*8400];
         host_input_buffer_ = new float[input_size];
         //创建异步流
         cudaStreamCreate(&stream_);

     }

std::vector<cv::Point2f> TensortDetector ::detectCorners(const cv::Mat& frame){
    if (frame.empty()) {
        return {};             //return an empty vector(the default initial value)
        //return std::vector<cv::Point2f>();
    }

         // ==========================================================
         // 1. 图像预处理 (LetterBox + BGR转RGB + 归一化 + HWC转CHW)
         // ==========================================================
        //将任意尺寸的原始图片，在不改变画面长宽比（不发生拉伸变形）的前提下，缩放并放入模型要求的固定输入尺寸中
        //计算缩放比例
         const float scale = std::min((float)model_width/frame.cols,(float)model_height/frame.rows);
        //缩小后的实际宽高
         int scaled_frame_w = std::round(frame.cols*scale);
         int scaled_frame_h = std::round(frame.rows*scale);
         //padding 适合模型输入需求(补齐上下左右空白)
         int pad_w = (model_width - scaled_frame_w )/2;
         int pad_h = (model_height - scaled_frame_h)/2;
         //创建cv对象，并且去压缩
         cv::Mat resized_frame;
         cv::resize(frame,resized_frame,cv::Size(scaled_frame_w,scaled_frame_h),0,0,cv::INTER_LINEAR);
         //创建一个全新的画布，并且将画布贴在中间
         cv::Mat padded_frame(model_height,model_width,CV_8UC3,cv::Scalar(0, 0, 0));
         //将缩放后的图像贴到画布正中间
         resized_frame.copyTo(padded_frame(cv::Rect(pad_w,pad_h,scaled_frame_w,scaled_frame_h)));
         //Opencv BGR-> Model RGB

         cv::cvtColor(padded_frame,RGB_frame,cv::COLOR_BGR2RGB);
         //改变通道的数值类型，并进行缩小
         RGB_frame.convertTo(float_frame,CV_32FC3,1.0f/255.0f);
         //frame HWC-> CHW
         std::vector<cv::Mat> chw_channels;
         for (int C= 0;C<3;C++) {
             chw_channels.emplace_back(cv::Mat(model_height, model_width, CV_32FC1,
                                          host_input_buffer_ + C * model_height * model_width));
         }
         cv::split(float_frame,chw_channels);
         // ==========================================================
         // 2. GPU 异步推理
         // ==========================================================
         //数据从CPU异步传输到GPU
         cudaMemcpyAsync(device_buffer[0],host_input_buffer_,input_size,cudaMemcpyHostToDevice,stream_);
         //利用异步流执行异步指令
         context_->setTensorAddress("frame_tensor", device_buffer[0]);
         context_->setTensorAddress("output_tensor", device_buffer[1]);
         context_->enqueueV3(stream_);
         //异步复制回CPU（内存）
         cudaMemcpyAsync(host_output_buffer_,device_buffer[1],output_size,cudaMemcpyDeviceToHost,stream_);
         //阻塞等待：确保推理和拷贝全部完成
         cudaStreamSynchronize(stream_); // 程序执行此行，CPU强行在这里等待
         // ==========================================================
         // 3. 解析结果与坐标逆映射
         // ==========================================================
         //存储最终还原好的坐标
         std::vector<cv::Point2f> final_corners{};
         PoseResult result = ParseYOLOv8Pose(host_output_buffer_, 8400, 0.6f); //锚点和置信度
         if (result.score< 0.6f || result.corners.size()!= 4) {  //置信度和大小判断
             return  final_corners;
         }
         // 逆映射：把 640x640 里的坐标，还原回 1080P (原始 frame) 的真实像素坐标
         for (const auto& pt : result.corners) {
             float original_x = (pt.x - pad_w) / scale;
             float original_y = (pt.y - pad_h) / scale;

             // 防止坐标越界跑到画面外面去
             original_x = std::max(0.0f, std::min(original_x, (float)frame.cols - 1.0f));
             original_y = std::max(0.0f, std::min(original_y, (float)frame.rows - 1.0f));

             final_corners.push_back(cv::Point2f(original_x, original_y));
         }
         return final_corners;
}


bool MNNOpenCLDetector::init(const std::string& modelPath){
        net_.reset(MNN::Interpreter::createFromFile(modelPath.c_str()));
        if (!net_) return false;

        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_OPENCL;
        config.numThread = 4;
        MNN::BackendConfig backendConfig;
        backendConfig.precision = MNN::BackendConfig::Precision_Low;
        config.backendConfig = &backendConfig;

        session_ = net_->createSession(config);
        input_tensor_ = net_->getSessionInput(session_, nullptr);

        // 【修改点2】：在 Init 阶段就配置并创建好 ImageProcess，绝不在循环里创建
        MNN::CV::ImageProcess::Config img_config;
        img_config.filterType = MNN::CV::BILINEAR;
        img_config.sourceFormat = MNN::CV::BGR;
        img_config.destFormat = MNN::CV::RGB;

        // 【修改点3】：修复内存泄漏。直接使用 const 栈数组赋值
        const float norm[3] = {1/255.0f, 1/255.0f, 1/255.0f};
        const float mean[3] = {0.0f, 0.0f, 0.0f}; // YOLO 不减均值
        ::memcpy(img_config.normal, norm, sizeof(norm));
        ::memcpy(img_config.mean, mean, sizeof(mean));

        pretreat_.reset(MNN::CV::ImageProcess::create(img_config));

        return true;
    }

    std::vector<cv::Point2f>MNNOpenCLDetector:: detectCorners(const cv::Mat& frame){
        if (frame.empty()) return {};

        // 【修改点4】：增加 Letterbox 处理，保证图像比例不失真
        float scale = std::min((float)model_w / frame.cols, (float)model_h / frame.rows);
        int new_w = std::round(frame.cols * scale);
        int new_h = std::round(frame.rows * scale);
        int pad_w = (model_w - new_w) / 2;
        int pad_h = (model_h - new_h) / 2;

        cv::Mat resized_img, padded_img(model_h, model_w, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::resize(frame, resized_img, cv::Size(new_w, new_h));
        resized_img.copyTo(padded_img(cv::Rect(pad_w, pad_h, new_w, new_h)));

        // 复用 Init 中创建好的 pretreat_ 进行高效转换 (底层会有 OpenCL 硬件加速)
        pretreat_->convert(padded_img.data, padded_img.cols, padded_img.rows, padded_img.step[0], input_tensor_);

        // 执行 GPU 推理
        net_->runSession(session_);

        // 获取输出 Tensor 并解析
        MNN::Tensor* output_tensor = net_->getSessionOutput(session_, nullptr);
        MNN::Tensor host_tensor(output_tensor, output_tensor->getDimensionType());
        output_tensor->copyToHostTensor(&host_tensor);

        float* out_ptr = host_tensor.host<float>();

        // 解析坐标...
        // PoseResult result = ParseYOLOv8Pose(out_ptr);

        // 【关键】：记得把 result.corners 里的点，用 scale, pad_w, pad_h 逆映射回 frame 的真实尺寸！

        return std::vector<cv::Point2f>();
    }

