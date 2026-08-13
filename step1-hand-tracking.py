import cv2 #real-time computer vision and video processing
import mediapipe as mp #landmark detection hand, face, pose

# Initialize the MediaPipe Hands module
mp_hands = mp.solutions.hands #access hand tracking module withing mediapipe

mp_drawing = mp.solutions.drawing_utils #access utilities to draw lins and dots on screen

hands = mp_hands.Hands(
    static_image_mode=False,#video stream
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
      ) #initialize hand tracking model with confidence thresholds



cap = cv2.VideoCapture(0)
#capture video from cam

print("starting camera, press q to quit")

#main loop to process video frames
print(cap.isOpened())
while cap.isOpened():
    #read asingle frame from the video stream
    # ret is boolean, if the frame was read successfully
    #frame is the actual image array
    ret, frame=cap.read()

    if not ret:
        print("failed to grab frame from camera")
        break

    frame=cv2.flip(frame,1) #camera will be in front of me (not first person view) so adding mirror effect to be more intuitive

    #opencv reads images in bgr format, but mediapipe expects rgb format, so we need to convert the color space
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #pass the rgb frame  to mediapipe model to detect handlandmarks
    results=hands.process(rgb_frame) #process the frame to detect hands and landmarks

    if results.multi_hand_landmarks: #if any hand landmarks are detected
        #iterate through each detected hand (1 in this case)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, #image to draw on
                hand_landmarks, #detected 21 hand keypoints
                mp_hands.HAND_CONNECTIONS #connecitons mapping between fingers
                )

    #display the final processed frame
    cv2.imshow("hans step 1: hand tracking",frame)

    if cv2.waitKey(1) & 0xFF==ord('q'): #wait for 1ms and check if 'q' key is pressed
        break

cap.release() #release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV pop-up windows