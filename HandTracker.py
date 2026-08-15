import cv2 #real-time computer vision and video processing
import mediapipe as mp #landmark detection hand, face, pose

class HandTracker:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands #access hand tracking module within mediapipe
        self.mp_drawing = mp.solutions.drawing_utils #access utilities to draw lines and dots on screen

        self.hands = self.mp_hands.Hands(
            static_image_mode=False, #video stream
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        ) #initialize hand tracking model with confidence thresholds

        self.results = None  # Store the results of hand tracking

    def process_frame(self, frame):
        """
        process the input frame to find landmarks
        frame is the BGR image from openCV
        return is the detected handlandmarks
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert BGR to RGB because mediapipe uses rgb
        self.results=self.hands.process(rgb_frame) #results include landmarks,handedness,metric 3d coordinates
        return self.results 

    def draw_landmarks(self, frame):
        """
        draw the detected landmarks on the frame
        frame is the BGR image from openCV
        return is the frame with landmarks drawn
        """
        if self.results and self.results.multi_hand_landmarks: #if any hand landmarks are detected
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, #image to draw on
                    hand_landmarks, #detected 21 hand keypoints
                    self.mp_hands.HAND_CONNECTIONS #connections mapping between fingers
                )
        return frame

    def get_hand_center(self,frame=None, format=0):
        """
        pixel coodinates of trhe hand center
        format {
        0=pixel
        1=normalized
        2=world_metric        
        }
        """

        if not self.results or not self.results.multi_hand_landmarks: #if any hand landmarks are detected
           return None

        landmark_2d = self.results.multi_hand_landmarks[0].landmark[9]

        if format==1:
            return round(landmark_2d.x,4), round(landmark_2d.y,4) #normalized coordinates

        elif format==0:
            if frame is None:
                return None
            height, width, _ = frame.shape
            cx= int(landmark_2d.x * width)
            cy= int(landmark_2d.y * height)
            return cx,cy

        elif format==2:
            if self.results.multi_hand_world_landmarks:
                landmark_3d = self.results.multi_hand_world_landmarks[0].landmark[9]
                return round(landmark_3d.x,4), round(landmark_3d.y,4), round(landmark_3d.z,4) #world metric coordinates
            return None

    

