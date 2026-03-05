import cv2
import os
import urllib.request

import cv2
import os
import urllib.request
import math

def get_cascade(xml_file):
    """获取级联分类器，如果本地没有则下载"""
    # 1. 尝试直接加载内置路径
    path = os.path.join(cv2.data.haarcascades, xml_file) if cv2.data.haarcascades else xml_file
    try:
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            return cascade
    except:
        pass
    
    # 2. 如果内置路径失败，检查当前目录
    if os.path.exists(xml_file):
        cascade = cv2.CascadeClassifier(xml_file)
        if not cascade.empty():
            return cascade
            
    # 3. 尝试下载
    print(f"正在下载模型文件 {xml_file}...")
    url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{xml_file}"
    try:
        urllib.request.urlretrieve(url, xml_file)
        print("下载成功！")
        cascade = cv2.CascadeClassifier(xml_file)
        if not cascade.empty():
            return cascade
    except Exception as e:
        print(f"下载失败: {e}")
        
    return None

# 获取分类器 (使用 alt2 模型，对仰视/俯视/侧脸的鲁棒性比 default 更好)
face_cascade = get_cascade('haarcascade_frontalface_alt2.xml')
if not face_cascade:
    print("错误: 无法加载人脸识别分类器")
    exit(1)

# 获取嘴巴/微笑分类器用于"说话"检测 (作为说话的简单替代: 张嘴/微笑)
smile_cascade = get_cascade('haarcascade_smile.xml')
if not smile_cascade:
    print("警告: 无法加载微笑检测器，将只使用人脸大小进行锁定。")

# 打开摄像头 (0代表默认摄像头)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("错误：无法打开摄像头")
    exit(1)

# 尝试设置高分辨率 (如果摄像头支持)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("按 'q' 键退出程序")

# 设置窗口为可调整大小 (WINDOW_NORMAL 允许用户手动最大化/全屏，并自适应分辨率)
window_name = 'Face Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# 设置初始弹窗大小 (例如 800x600)，以便于开始时不是全屏
cv2.resizeWindow(window_name, 800, 600)

# 平滑滤波参数
last_cx, last_cy = None, None
alpha = 0.15  # 平滑系数

# 追踪稳定性参数 (新添加)
last_locked_rect = None  # 上次锁定的位置 (x, y, w, h)
missing_frames = 0       # 连续丢失帧数
MAX_MISSING_FRAMES = 8   # 允许最大丢失帧数 (约0.3秒)，用于防抖

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收摄像头信号")
        break
    
    frame_h, frame_w = frame.shape[:2]
    
    # 转换为灰度图以提高检测效率
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸 (降低 scaleFactor 提高检测精度, 降低 minNeighbors 增加召回率)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    
    target_face = None
    max_score = -1
    
    # 1. 筛选最佳人脸
    for (x, y, w, h) in faces:
        # 基础分：面积越大越接近/正对 (w * h)
        score = w * h
        
        # A. 追踪粘性 (Tracking Stickiness)
        # 如果当前人脸位置接近上次锁定的位置，给予巨大加分
        # 这能防止人脸转动导致检测失效时瞬间切换到路人，或防止目标跳动
        if last_locked_rect is not None:
            lx, ly, lw, lh = last_locked_rect
            # 计算中心点距离
            curr_cx, curr_cy = x + w//2, y + h//2
            last_cx_rect, last_cy_rect = lx + lw//2, ly + lh//2
            dist = ((curr_cx - last_cx_rect)**2 + (curr_cy - last_cy_rect)**2)**0.5
            
            # 如果距离在合理范围内 (比如小于人脸宽度的2/3)，判定为同一个人
            if dist < lw * 0.8:
                score *= 10.0  # 锁定加成：只要还在附近，就优先锁定它！
        
        # B. 说话/张嘴检测
        if smile_cascade:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray_lower = roi_gray[h//2:, :]
            try:
                smiles = smile_cascade.detectMultiScale(roi_gray_lower, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
                if len(smiles) > 0:
                    score *= 2.5  # 说话加成
            except:
                pass
        
        if score > max_score:
            max_score = score
            target_face = (x, y, w, h)
            
    # 决策逻辑：使用检测结果 OR 预测结果
    final_rect = None
    is_prediction = False
    
    if target_face:
        # 找到了人脸，更新追踪器
        final_rect = target_face
        last_locked_rect = target_face
        missing_frames = 0
    elif last_locked_rect is not None and missing_frames < MAX_MISSING_FRAMES:
        # 没找到人脸，但是之前锁定了，且丢失不久 -> 启动预测模式
        # 保持显示上一次的位置，防止画面闪烁丢失
        final_rect = last_locked_rect
        missing_frames += 1
        is_prediction = True
    else:
        # 确实丢了
        last_locked_rect = None
        # 如果长时间没识别到，重置平滑器，避免下次识别瞬间飞过去
        if last_cx is not None and missing_frames > 20:
             last_cx, last_cy = None, None

    # 2. 如果确定了目标 (无论是检测到的还是预测的)，进行绘制
    if final_rect:
        x, y, w, h = final_rect
        
        # 边缘推测 (如果人脸在边缘，推测完整中心)
        raw_cx = x + w // 2
        raw_cy = y + h // 2
        
        is_edge = False
        
        # 左边缘修正 (看左半边，推测中心在更左侧)
        if x < 10: 
            raw_cx = (x + w) - (h // 2)
            is_edge = True 
        # 右边缘修正
        elif (x + w) > (frame_w - 10):
            raw_cx = x + (h // 2)
            is_edge = True
            
        # 坐标平滑处理 (Exponential Smoothing)
        # 用预测模式时，稍微增加平滑系数，让它停得更稳
        current_alpha = alpha * 0.5 if is_prediction else alpha 
        
        if last_cx is None:
             last_cx, last_cy = float(raw_cx), float(raw_cy)
        else:
             last_cx = last_cx * (1 - current_alpha) + raw_cx * current_alpha
             last_cy = last_cy * (1 - current_alpha) + raw_cy * current_alpha
        
        # 输出用的坐标
        final_x, final_y = int(last_cx), int(last_cy)

        # 绘制
        # 锁定框颜色：正常绿色，预测状态(丢失中)用黄色
        color = (0, 255, 255) if is_prediction else (0, 255, 0)
        thick = 1 if is_prediction else 2
        
        # 绘制矩形 (在预测模式下画虚线或者是细线)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thick)
        
        # 中心点
        draw_center_x = max(0, min(final_x, frame_w))
        draw_center_y = max(0, min(final_y, frame_h))
        
        cv2.circle(frame, (draw_center_x, draw_center_y), 5, (0, 0, 255), -1)
        
        info_text = f"TGT ({final_x}, {final_y})"
        if is_prediction:
            info_text += " [Lost?]"
        elif is_edge:
            info_text += " [Edge]"
            
        cv2.putText(frame, info_text, (max(10, x), max(30, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 只在实际检测到时大量输出，避免刷屏，或者都输出
        status = "Locked" if not is_prediction else "Tracking(Lost)"
        print(f"状态: {status} | 坐标: x={final_x}, y={final_y}")
    else:
        # 彻底丢失
        pass
    
    # 显示结果画面
    cv2.imshow(window_name, frame)
    
    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
