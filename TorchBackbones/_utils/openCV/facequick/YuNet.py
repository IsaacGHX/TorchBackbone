import cv2
import numpy as np
print(cv2.__version__)
if __name__ == '__main__':
    model_file = "D:/datasets/Models/openCV/face/face_detection_yunet_2023mar.onnx"
    conf_thre = 0.9
    nms_thred = 0.3
    topK = 5000
    config = ''
    model = cv2.FaceDetectorYN.create(
        model=model_file,
        config='',
        input_size=[320, 320],
        score_threshold=conf_thre,
        nms_threshold=nms_thred,
        top_k=topK,
        backend_id=0,
        target_id=0
    )

    freq = cv2.getTickFrequency()

    cap = cv2.VideoCapture(0)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model.setInputSize([w, h])

if __name__ == '__main__':

    import cv2

    cap = cv2.VideoCapture(1)
    cap.set(3, 720)  # set video width
    cap.set(4, 680)  # set video height

    # Variables for calculating moving average
    fps_values = []
    num_frames_to_average = 10

    while True:
        startTime = cv2.getTickCount()

        # Read frame
        ret, frame = cap.read()
        faces = model.detect(frame)
        results = faces[1]

        endTime = cv2.getTickCount()
        timeDelta = (endTime - startTime) / cv2.getTickFrequency()

        fps = 1 / timeDelta
        fps_values.append(fps)

        # Keep only the last 'num_frames_to_average' values for smoother calculation
        if len(fps_values) > num_frames_to_average:
            fps_values = fps_values[-num_frames_to_average:]

        # Calculate the average FPS
        avg_fps = sum(fps_values) / len(fps_values)

        cv2.putText(frame, "FPS: {:.2f}".format(avg_fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for det in (results if results is not None else []):
            x, y, w, h = det[0:4].astype(np.int32)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            score = det[-1]
            cv2.putText(frame, "%.2f" % (score), (x, y + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            landmarks = det[4:14].astype(np.int32).reshape((5, 2))
            for idx, landmark in enumerate(landmarks):
                cv2.circle(frame, landmark, 2, (0, 255, 255), 2)

        cv2.imshow('fourcc', frame)
        k = cv2.waitKey(1)

        # Press 'q' to exit
        if k & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# import onnxruntime
# import cv2
# import numpy as np

# # 指定ONNX模型文件的路径
# model_path = "D:/datasets/Models/openCV/face/face_detection_yunet_2023mar.onnx"
#
# # 创建ONNX运行时的推理会话
# ort_session = onnxruntime.InferenceSession(model_path)
#
# # 准备输入数据
# image = cv2.imread('test.jpg')
# image = cv2.resize(image, (640, 640))
# input_blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(640, 640), mean=(0, 0, 0), swapRB=True, crop=False)
#
# # 将输入数据传递给ONNX模型
# ort_inputs = {ort_session.get_inputs()[0].name: np.array(input_blob)}
# ort_outputs = ort_session.run(None, ort_inputs)
#
# print(ort_outputs)