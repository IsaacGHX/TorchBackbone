import cv2

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

        cv2.imshow('fourcc', frame)
        k = cv2.waitKey(1)

        # Press 'q' to exit
        if k & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

