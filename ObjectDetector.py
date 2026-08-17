import cv2
from ultralytics import YOLO

class ObjectDetector:
    #real time object detection using YOLOv8 model
    def __init__(self, model_name="yolov8n.pt", target_label="bottle", conf_threshold=0.5, max_missing_frames=5 ):
        self.model = YOLO(model_name)  # Load the YOLOv8 model
        self.target_label = target_label.lower()  # Set the target label for detection
        self.conf_threshold = conf_threshold  # Set the confidence threshold for detection
        self.results = None  # Store the results of object detection
        #to prevent flickering
        self.last_box = None
        self.last_center = None
        self.missing_counter = 0
        self.max_missing_frames = max_missing_frames

    def set_target_label(self, new_target:str):
        """
        dynamic target detection
        """
        self.target_label = new_target.lower()

    def process_frame(self, frame):
        """
        Performs object detection inference on the input frame.
        :param frame: The BGR image frame from OpenCV.
        :return: Ultralytics Results object.
        """
        self.results = self.model(frame, conf=self.conf_threshold,imgsz=320,verbose=False)[0]  # Perform object detection on the frame
        return self.results

    def get_target_box_and_center(self):
        """
        Get the bounding box and center coordinates of the target object.
        if multiple matches are found, return the one with the highest confidence.
        :return: Tuple of (bounding_box, center_coordinates) or (None, None) if not found.
        """

        if self.results is None or len(self.results.boxes) ==0:
            return None, None

        best_box=None
        highest_confidence=-1.0

        for box in self.results.boxes:
            #retrieve class id and name
            class_id= int(box.cls[0].item())
            class_name= self.results.names[class_id].lower()
            confidence= float(box.conf[0].item())

            if class_name==self.target_label:
                if confidence > highest_confidence:
                    highest_confidence=confidence
                    coords = box.xyxy[0].cpu().numpy().astype(int)  
                    best_box=coords

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            return (x1, y1, x2, y2), (cx, cy)

        return self._handle_missing()


    def draw_target(self, frame, box, center):
        #Draws a bounding box and center marker on the frame for the detected target.

        if box is not None and center is not None:
            x1, y1, x2, y2 = box
            cx, cy = center

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,140,255),2)

            cv2.circle(frame, (cx,cy), 8, (0,140,255),cv2.FILLED)
            cv2.drawMarker(
                    frame,
                    (cx, cy),
                    (255, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=14,
                    thickness=2
                )

                # 3. Label text
            label_text = f"TARGET: {self.target_label.upper()} ({cx}, {cy})"
            cv2.putText(
                frame,
                label_text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 140, 255),
                2
            )

        return frame

    def _handle_missing(self):
        """to maintain drawed box for a few frames if the target is missing so we eont see box is flickering"""
        if (
            self.last_box is not None
            and self.missing_counter < self.max_missing_frames
        ):
            self.missing_counter += 1
            return self.last_box, self.last_center
        else:
            self.last_box = None
            self.last_center = None
            return None, None