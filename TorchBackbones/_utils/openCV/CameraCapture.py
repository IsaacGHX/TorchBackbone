import cv2, uuid, os, time
from threading import Thread
"""实现的是从摄像头调用一张图片，并且显示"""

count = 0
def image_collect(cap):
    global count
    while True:
        success, img = cap.read()
        if success:
            file_name = str(uuid.uuid4()) + '.jpg'
            cv2.imwrite(os.path.join('images, file_name'), img)
            count += 1
            print(f"save {count} {file_name}")
        time.sleep(0.4)


if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)

    cap = cv2.VideoCapture(1)
    m_thread = Thread(target=image_collect, args=([cap]), daemon=True)
    while True:
        success, img = cap.read()
        if not success:
            continue
        cv2.imshow("video", img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('c'):
            m_thread.start()
            continue
        elif key == ord('q'):
            break

    cap.release()
