#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include "whiteboardenchance.h"
// 辅助函数：对四个角点进行排序 (确保顺序为: 左上、右上、右下、左下)
void sortCorners(std::vector<cv::Point2f>& pts) {
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

// 透视变换与去阴影处理函数(已经替换为OpenClenchance)
cv::Mat enhanceWhiteboard(const cv::Mat& src_image, std::vector<cv::Point2f> corners) {
    if (corners.size() != 4) {
        return src_image.clone();
    }

    // 1. 确保角点顺序正确
    sortCorners(corners);

    // 2. 计算目标拉平后的宽高 (可以根据实际白板比例自定义，比如 1000x750 或 1200x900)
    // 这里通过计算上下、左右边长的最大值来决定输出尺寸，保证不失真
    int widthTop = cv::norm(corners[1] - corners[0]);
    int widthBottom = cv::norm(corners[2] - corners[3]);
    int maxWidth = std::max(widthTop, widthBottom);

    int heightLeft = cv::norm(corners[3] - corners[0]);
    int heightRight = cv::norm(corners[2] - corners[1]);
    int maxHeight = std::max(heightLeft, heightRight);

    // 如果检测到的尺寸异常，给个默认安全值
    if (maxWidth < 100 || maxHeight < 100) {
        maxWidth = 800;
        maxHeight = 600;
    }

    // 定义目标矩形的四个顶点
    std::vector<cv::Point2f> srcPoints = {
        (cv::Point2f)corners[0], // 左上
        (cv::Point2f)corners[1], // 右上
        (cv::Point2f)corners[2], // 右下
        (cv::Point2f)corners[3]  // 左下
    };

    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(0, 0),
        cv::Point2f((float)maxWidth, 0),
        cv::Point2f((float)maxWidth, (float)maxHeight),
        cv::Point2f(0, (float)maxHeight)
    };

    // 3. 计算单应性矩阵并进行透视变换（拉平）
    cv::Mat warpedImage;
    cv::Mat matrix = cv::getPerspectiveTransform(srcPoints, dstPoints);
    cv::warpPerspective(src_image, warpedImage, matrix, cv::Size(maxWidth, maxHeight));

    // 4. 去阴影与色彩增强（灰度化 + 局部自适应阈值）
    cv::Mat gray, thresh;
    cv::cvtColor(warpedImage, gray, cv::COLOR_BGR2GRAY);

    // 自适应阈值化（Adaptive Threshold）
    // Block Size 必须是奇数（例如 31 或 51），根据白板上的字迹粗细调整
    // C 是常数，用于微调过滤掉阴影的敏感度
    int blockSize = 31;
    double C = 10.0;
    cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, blockSize, C);

    // 此时 thresh 是黑白二值图（字是黑色 0，背景是白色 255）
    // 如果你希望输出带有原图色彩但去除了阴影的图像，可以通过除法或者掩码与原图结合
    // 这里直接返回精简干净的二值化文档图
    cv::Mat resultColor;
    cv::cvtColor(thresh, resultColor, cv::COLOR_GRAY2BGR);

    return resultColor;
}



using Clock = std::chrono::steady_clock;

// 帧数据结构
struct Frame {
    cv::Mat image;
    std::uint64_t sequence = 0;
    Clock::time_point capturedAt;
};

// 双缓冲机制：保证拿到的永远是最新的一帧
template<typename T>
class LatestDoubleBuffer {
public:
    void publish(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const int next = 1 - lastWritten_;
            slots_[next] = std::move(item);
            lastWritten_ = next;
            readyIndex_ = next;
            hasReady_ = true;
        }
        ready_.notify_one();
    }

    bool takeLatest(T& output, const std::atomic_bool& running) {
        std::unique_lock<std::mutex> lock(mutex_);
        //ready_.wait(lock, [&] { return hasReady_ || !running.load(); }); //[&] 访问这个作用域的所有函数和变量 ,判断值为真则 接着把持锁
        while (!(hasReady_ ||!running.load())) {
         ready_.wait(lock);
        }
        if (!hasReady_) {
            return false;
        }
        output = slots_[readyIndex_];
        hasReady_ = false;
        return true;
    }

    void wakeAll() { ready_.notify_all(); }

private:
    std::array<T, 2> slots_;
    int lastWritten_ = 1;
    int readyIndex_ = 0;
    bool hasReady_ = false;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
};

// FPS 计算器
class FpsMeter {
public:
    void tick() {
        ++count_;
        const auto now = Clock::now();
        const auto elapsed = std::chrono::duration<double>(now - started_).count();
        if (elapsed >= 1.0) {
            fps_.store(static_cast<double>(count_) / elapsed);
            count_ = 0;
            started_ = now;
        }
    }
    double fps() const { return fps_.load(); }

private:
    Clock::time_point started_ = Clock::now();
    std::uint32_t count_ = 0;
    std::atomic<double> fps_{0.0};
};

struct Options {
    int cameraIndex = 0;
    bool display = true;
    int captureBackend = cv::CAP_ANY;
};

// 1. 抓帧子线程
void captureLoop(const Options& options, LatestDoubleBuffer<Frame>& frameBuffer,
                 std::atomic_bool& running, FpsMeter& captureFps) {
    cv::VideoCapture camera(options.cameraIndex, options.captureBackend);
    if (!camera.isOpened()) {
        std::cerr << "Cannot open camera " << options.cameraIndex << ".\n";
        running = false;
        frameBuffer.wakeAll();
        return;
    }

    camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    camera.set(cv::CAP_PROP_FPS, 30);
    camera.set(cv::CAP_PROP_BUFFERSIZE, 1);

    std::uint64_t sequence = 0;
    while (running.load()) {
        cv::Mat image;
        if (!camera.read(image) || image.empty()) {
            std::cerr << "Camera read failed; stopping.\n";
            break;
        }
        frameBuffer.publish({std::move(image), ++sequence, Clock::now()});
        captureFps.tick();
    }
    running = false;
    frameBuffer.wakeAll();
}

// 推理结果结构体
struct InferenceResult {
    std::vector<cv::Point2f> corners;
    bool success = false;
    std::uint64_t sequence = 0;
};

// 2. 推理子线程：专门在后台默默计算 MNN，解放主线程！
void inferenceLoop(LatestDoubleBuffer<Frame>& frameBuffer,
                   LatestDoubleBuffer<InferenceResult>& resultBuffer,
                   std::atomic_bool& running,
                   std::shared_ptr<MNN::Interpreter> net,
                   MNN::Session* session) {

    MNN::Tensor* input_tensor = net->getSessionInput(session, nullptr);

    // 配置 MNN 图像预处理器
    MNN::CV::ImageProcess::Config img_config;
    img_config.filterType = MNN::CV::BILINEAR;
    img_config.sourceFormat = MNN::CV::BGR;
    img_config.destFormat = MNN::CV::RGB;
    float normal[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    ::memcpy(img_config.normal, normal, sizeof(normal));
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(img_config));

    Frame frame;
    while (frameBuffer.takeLatest(frame, running)) {
        const int MODEL_INPUT_WIDTH = 256;
        const int MODEL_INPUT_HEIGHT = 256;

        cv::Mat resized_img;
        cv::resize(frame.image, resized_img, cv::Size(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT));

        // 塞入 Tensor 并推理
        pretreat->convert(resized_img.data, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, resized_img.step[0], input_tensor);
        net->runSession(session);

        // 获取输出
        MNN::Tensor* output_tensor = net->getSessionOutput(session, nullptr);
        MNN::Tensor host_output(output_tensor, MNN::Tensor::CAFFE);
        output_tensor->copyToHostTensor(&host_output);

        int channel = host_output.channel();
        int hm_height = host_output.height();
        int hm_width = host_output.width();

        InferenceResult res;
        res.sequence = frame.sequence;

        if (channel == 4 && hm_width > 0 && hm_height > 0) {
            float* out_data = host_output.host<float>();
            int plane_size = hm_width * hm_height;

            // 1. 创建一个容器，专门存放 AI 输出的浮点数类型“粗坐标”
            std::vector<cv::Point2f> rough_corners;

            for (int c = 0; c < 4; ++c) {
                float max_val = -1e9;
                int max_idx = 0;
                for (int i = 0; i < plane_size; ++i) {
                    float val = out_data[c * plane_size + i];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = i;
                    }
                }
                float hm_x = max_idx % hm_width;
                float hm_y = max_idx / hm_width;

                // 计算回原图的坐标，这里保留小数部分，存为 float
                float orig_x = (hm_x / hm_width) * frame.image.cols;
                float orig_y = (hm_y / hm_height) * frame.image.rows;
                rough_corners.push_back(cv::Point2f(orig_x, orig_y));
            }

            // 2. 方案三核心：OpenCV 亚像素级角点精调 (Sub-pixel Refinement)
            if (!rough_corners.empty()) {
                cv::Mat gray_img;
                // cornerSubPix 必须在单通道灰度图上运行
                cv::cvtColor(frame.image, gray_img, cv::COLOR_BGR2GRAY);

                // 配置搜索窗口大小：Size(5,5) 表示以粗坐标为中心，划定一个 11x11 (2*5+1) 的搜索区域
                cv::Size winSize(5, 5);
                // 死区大小：-1 表示不使用死区
                cv::Size zeroZone(-1, -1);
                // 迭代停止条件：最多迭代 40 次，或者精度达到 0.001 像素时停止
                cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001);

                // 执行亚像素吸附！该函数会原地修改 rough_corners 里的坐标
                cv::cornerSubPix(gray_img, rough_corners, winSize, zeroZone, criteria);

                // 3. 将吸附优化后的极高精度坐标，保存进你的结果中
                for (const auto& pt : rough_corners) {
                    // 如果你的 res.corners 只能存整数 (cv::Point)，这里做四舍五入。
                    // 强烈建议：后续把 InferenceResult 里的 std::vector<cv::Point> 改成 std::vector<cv::Point2f>！
                    // 透视变换 (getPerspectiveTransform) 接收带小数点的 Point2f 精度会更高。
                    res.corners.push_back(cv::Point2f(pt.x, pt.y));
                }
            }
            res.success = true;
        }

        resultBuffer.publish(std::move(res));
    }
    resultBuffer.wakeAll();
}

int main(int argc, char** argv) {
    std::cout << cv::getBuildInformation() << std::endl;
    Options options;
    options.captureBackend = cv::CAP_ANY;
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    // 初始化 MNN 模型
    std::string model_path = "D:/Project2/models/lcnet100_int8.mnn";
    std::cout << "Loading the model: " << model_path << std::endl;

    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (!net) {
        std::cerr << "Loading model failed!" << std::endl;
        return -1;
    }

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;

    MNN::Session* session = net->createSession(config);
    if (!session) {
        std::cerr << "Failed to create Session!" << std::endl;
        return -1;
    }
    std::cout << "Model loaded and initialized successfully!" << std::endl;

    // 多线程缓冲区与控制变量
    LatestDoubleBuffer<Frame> frameBuffer;
    LatestDoubleBuffer<InferenceResult> resultBuffer;
    std::atomic_bool running(true);
    FpsMeter captureFps;   // 1号计步器：专门负责统计摄像头采集帧率
    FpsMeter processingFps; // 2号计步器：专门负责统计模型处理帧率

    // 启动抓帧线程与后台推理线程
    std::thread captureThread(captureLoop, std::ref(options), std::ref(frameBuffer),
                              std::ref(running), std::ref(captureFps));
    std::thread inferenceThread(inferenceLoop, std::ref(frameBuffer), std::ref(resultBuffer),
                                std::ref(running), net, session);

    // 主线程：只负责极速 UI 渲染与事件响应 (再也不会未响应了！)
    if (options.display) {
         cv::namedWindow("Whiteboard Pipeline - Stage 1", cv::WINDOW_NORMAL);
         cv::namedWindow("Enhanced Whiteboard", cv::WINDOW_NORMAL);

        cv::resizeWindow("Whiteboard Pipeline - Stage 1", 800, 600);
         cv::resizeWindow("Enhanced Whiteboard", 800, 600);
    }

    Frame currentFrame;
    InferenceResult currentResult;
    bool hasValidFrame = false;
    bool hasValidResult = false;
    std::cout << "Initializing OpenCL GPU Engine..." << std::endl;
    OpenCLEnhancer gpuEnhancer;
    if (!cv::ocl::haveOpenCL()) {
        std::cout << "[fatal error] system or OpenCV didnot support OpenCLforce to use CPU！" << std::endl;
    } else {
        cv::ocl::setUseOpenCL(true);
        std::cout << "[successfully] OpenCL already Open" << std::endl;
        cv::ocl::Context context = cv::ocl::Context::getDefault();
        std::cout << "The device which take over: " << context.device(0).name() << std::endl;
    }
    while (running.load()) {
        // 非阻塞或最新获取：刷新画面
        // 尝试拿最新的一帧图像（如果有的话）
        Frame tempFrame;
        if (frameBuffer.takeLatest(tempFrame, running)) {
            currentFrame = std::move(tempFrame);
            hasValidFrame = true;
        }
        // 尝试拿最新的推理结果（如果有的话）
        InferenceResult tempRes;
        if (resultBuffer.takeLatest(tempRes, running)) {
            currentResult = std::move(tempRes);
            hasValidResult = true;
            processingFps.tick();
        }

        if (hasValidFrame) {
            cv::Mat preview = currentFrame.image;

            // 如果AI处理后有对应的角点结果，画在图上
            if (hasValidResult && currentResult.success && (currentResult.corners.size()== 4)) {
                for (const auto& pt : currentResult.corners) {
                    cv::circle(preview, pt, 8, cv::Scalar(0, 0, 255), -1);
                }
                for (size_t i = 0; i < currentResult.corners.size(); ++i) {
                    cv::line(preview, currentResult.corners[i], currentResult.corners[(i + 1) % currentResult.corners.size()], cv::Scalar(255, 0, 0), 2);
                }



                // 显示 FPS 信息
                std::ostringstream text;
                text << std::fixed << std::setprecision(1)
                     << "capture " << captureFps.fps() << " fps | process " << processingFps.fps() << " fps " << "| corners: "<<currentResult.corners.size();
                cv::putText(preview, text.str(), {24, 42}, cv::FONT_HERSHEY_SIMPLEX, 0.75, {0, 255, 0}, 2, cv::LINE_AA);

                cv::imshow("Whiteboard Pipeline - Stage 1", preview);

            }
            // 1. 获取 AI 裁剪拉平后的原始画面（尺寸可能是扭曲的）
            cv::Mat enhancedDoc = gpuEnhancer.process(currentFrame.image, currentResult.corners);
            //动态计算以后，固定比例
            cv::resize(enhancedDoc, enhancedDoc, cv::Size(800, 600));

            // 2. 在固定好比例的“干净画布”上写字
            std::ostringstream text2;
            text2 << std::fixed << std::setprecision(1)
                  << "process " << processingFps.fps() << " fps"; // 建议这里改成 processingFps
            cv::putText(enhancedDoc, text2.str(), {24, 42}, cv::FONT_HERSHEY_COMPLEX, 1.00, {0, 255, 0}, 2, cv::LINE_AA);
            // 3. 显示出来，此时字体绝对端正！
            cv::imshow("Enhanced Whiteboard", enhancedDoc);

        }

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            running = false;
            frameBuffer.wakeAll();
            resultBuffer.wakeAll();
            break;
           }
       }

            // 回收子线程
            captureThread.join();
            inferenceThread.join();

            if (options.display) {
                cv::destroyAllWindows();
            }

            return 0;
   }