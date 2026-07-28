# 白板视频流管线：阶段一

本工程实现了 1080p/60FPS 摄像头采集的 C++ 双线程骨架：

- 生产者线程请求摄像头输出 `1920×1080 @ 60FPS`，并尽量将后端缓存限制为 1 帧。
- `LatestFrameDoubleBuffer` 使用两个可复用帧槽位。若处理线程未取走旧帧，采集线程会直接覆盖它并计入丢帧数，避免延迟不断累积。
- 消费者线程只取得最新帧；当前的处理位置预留给后续 MNN/OpenVINO 推理、透视校正和 OpenCL 算子。
- 窗口叠加显示采集 FPS、处理 FPS、端到端延迟及主动丢帧数。按 `Q` 或 `Esc` 退出。

## 构建

安装 CMake、C++17 编译器和 OpenCV 开发包后：

```powershell
cmake -S . -B build
cmake --build build --config Release
./build/Release/whiteboard_pipeline.exe
```

非 MSVC 单配置生成器通常输出为 `./build/whiteboard_pipeline`。

## 运行选项

```powershell
# 指定摄像头索引
./build/Release/whiteboard_pipeline.exe --camera 1

# Windows USB 摄像头可按需尝试不同媒体后端（默认自动选择）
./build/Release/whiteboard_pipeline.exe --backend dshow
./build/Release/whiteboard_pipeline.exe --backend msmf

# 模拟慢处理，验证旧帧会被主动丢弃而非排队
./build/Release/whiteboard_pipeline.exe --process-delay-ms 80

# 无界面压测
./build/Release/whiteboard_pipeline.exe --no-display
```

摄像头驱动不一定支持 1080p/60FPS；程序启动时会打印设备实际采用的分辨率和帧率。
