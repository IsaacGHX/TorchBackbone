import onnxruntime
import cv2
import numpy as np
import time

# 指定ONNX模型文件的路径
model_path = "D:/datasets/Models/openCV/face/face_detection_yunet_2023mar.onnx"

# 创建ONNX运行时的推理会话
ort_session = onnxruntime.InferenceSession(model_path)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置视频帧的大小
frame_width = 640
frame_height = 480
cap.set(3, frame_width)
cap.set(4, frame_height)

while True:
    start_time = time.time()

    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        break

    # 调整帧大小和通道顺序以匹配模型输入
    frame = cv2.resize(frame, (640, 640))
    input_blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(640, 640), mean=(0, 0, 0), swapRB=True, crop=False)

    # 将输入数据传递给ONNX模型
    ort_inputs = {ort_session.get_inputs()[0].name: np.array(input_blob)}
    output = ort_session.run(None, ort_inputs)

    # 检查输出是否包含检测结果
    if len(output) > 0:
        for detection in output[0]:
            confidence = detection[2]
            if confidence > 0.5:  # 仅绘制置信度大于0.5的检测结果
                x1, y1, x2, y2 = map(int, detection[3:7] * np.array([frame_width, frame_height, frame_width, frame_height]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 计算并显示FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示帧
    cv2.imshow('Face Detection', frame)

    # 检查是否按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
