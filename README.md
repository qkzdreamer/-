# -
Simple face tracking with OpenCV"
使用说明 (User Guide)

1. 环境要求
   - 需要安装 Python (建议 3.8 或以上版本)
   - 需要安装 OpenCV 库

2. 安装依赖
   在终端或命令行中运行：
   pip install opencv-python

3. 运行程序
   在终端中运行：
   python face_center_detect.py

4. 功能说明
   - 程序会自动打开摄像头。
   - 自动检测画面中的人脸，并锁定最接近正对或正在说话的人。
   - 画面中会显示人脸框和中心红点。
   - 控制台会输出中心点坐标 (x, y)。
   - 按 'q' 键退出程序。

5. 注意事项
   - 第一次运行时，程序会尝试从网络下载必要的人脸识别模型文件 (xml)，请保持网络连接。
   - 如果运行报错，请确保摄像头未被其他程序占用。
