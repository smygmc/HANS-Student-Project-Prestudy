import cv2
from HandTracker import HandTracker
from ObjectDetector import ObjectDetector

def main():
    camera_source = "http://192.168.1.196:8080/video"

    cap = cv2.VideoCapture(camera_source)
    
    #create handtracker object
    hand_tracker = HandTracker()
    object_detector = ObjectDetector(model_name="yolov8n.pt", target_label="bottle", conf_threshold=0.5)
    print("HANS Pipeline running... Press 'q' to quit.")

    while cap.isOpened():
        #read asingle frame from the video stream
        # ret is boolean, if the frame was read successfully
        #frame is the actual image array
        ret, frame=cap.read()

        if not ret:
            print("failed to grab frame from camera")
            break

        frame=cv2.flip(frame,1) #camera will be in front of me (not first person view) so adding mirror effect to be more intuitive

        hand_tracker.process_frame(frame)
        # Retrieve palm center coordinates in pixels
        hand_center = hand_tracker.get_hand_center(frame, format=0)
        frame = hand_tracker.draw_landmarks(frame)

        # Highlight the center point and overlay coordinates
        
        if hand_center:
            
            cx, cy = hand_center

            # green
            cv2.circle(frame, (cx, cy), 12, (0, 255, 0), cv2.FILLED)

            # coordinate text
            cv2.putText(
                frame,
                f"Center: ({cx}, {cy})",
                (cx + 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        object_detector.process_frame(frame)
        target_box, target_center = object_detector.get_target_box_and_center()

        if target_box and target_center:
            frame = object_detector.draw_target(frame, target_box, target_center)

        #display the final processed frame
        cv2.imshow("hans step 1: hand tracking",frame)

        if cv2.waitKey(1) & 0xFF==ord('q'): #wait for 1ms and check if 'q' key is pressed
            break

    cap.release() #release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV pop-up windows


if __name__ == "__main__":
    main()