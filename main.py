import cv2
import config
from HandTracker import HandTracker
from ObjectDetector import ObjectDetector
from GuidanceController import GuidanceController
import numpy as np

def main():
    
    cap = cv2.VideoCapture(config.CAMERA_SOURCE)
    
    #create handtracker object
    hand_tracker = HandTracker()
    object_detector = ObjectDetector(model_name="yolov8n.pt", target_label="bottle", conf_threshold=0.2)
    guidance_controller = GuidanceController(tolerance_radius=75, z_tolerance=60.0)
    

    print("HANS Pipeline running... Press 'q' to quit.")

    frame_count = 0#video stream is so slow ehwn yolo processes every frame
    depth_map = None
    while cap.isOpened():
        #read asingle frame from the video stream
        # ret is boolean, if the frame was read successfully
        #frame is the actual image array
        ret, frame=cap.read()

        if not ret:
            print("failed to grab frame from camera")
            break
        frame_count += 1
        #frame=cv2.flip(frame,1) #camera will be in front of me (not first person view) so adding mirror effect to be more intuitive


        hand_tracker.process_frame(frame)
        # Retrieve palm center coordinates in pixels
        hand_center = hand_tracker.get_hand_center(frame)

        if frame_count % 5 == 0:
            object_detector.process_frame(frame)
        

        target_box, target_center = object_detector.get_target_box_and_center()
        
        if frame_count % 7 == 0 or depth_map is None:
            depth_map = guidance_controller.estimate_scene_depth(frame)

        # 3. [GÜNCELLENDİ] 3D Yönlendirmeyi Hesapla (depth_map parametresi eklendi)
        guidance_data = guidance_controller.compute_guidance(
            hand_center, target_center, depth_map=depth_map
        )

        if hand_center is not None:
            frame = hand_tracker.draw_landmarks(frame)

        if target_box and target_center:
            frame = object_detector.draw_target(frame, target_box, target_center)

        # Draw 2D trajectory arrow and navigation commands
        if guidance_data:
            guidance_controller.draw_guidance(
                frame, hand_center, target_center, guidance_data
            )

        #display the final processed frame
        cv2.imshow("hans step 1: hand tracking",frame)

        if cv2.waitKey(1) & 0xFF==ord('q'): #wait for 1ms and check if 'q' key is pressed
            break

    cap.release() #release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV pop-up windows


if __name__ == "__main__":
    main()