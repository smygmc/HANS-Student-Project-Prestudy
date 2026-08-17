import math
import cv2


class GuidanceController:
    """
    Computes 2D spatial guidance metrics between the hand center and target object.
    """

    def __init__(self, tolerance_radius=45):
        
        self.tolerance_radius = tolerance_radius

    def compute_guidance(self, hand_center, target_center):
        """
        Calculates distance, delta vector, and navigation command.
        
        """
        if hand_center is None or target_center is None:
            return None

        hx, hy = hand_center
        tx, ty = target_center

        # Delta vector target-hand (screen orijin is top left)
        dx = tx - hx
        dy = ty - hy

        # Euclidean distance-distance
        distance = math.sqrt(dx**2 + dy**2)

        # Check if hand is within tolerance threshold
        if distance <= self.tolerance_radius:
            direction_command = "REACHED"
        else:
            # Determine dominant direction axis to give only one direction command at a time
            if abs(dx) > abs(dy):
                # dx>0 target on the right
                direction_command = "RIGHT" if dx > 0 else "LEFT"
            else:
                # dy>0 target is below
                direction_command = "DOWN" if dy > 0 else "UP"

        return {
            "dx": dx,
            "dy": dy,
            "distance": round(distance, 2),
            "command": direction_command,
        }

    def draw_guidance(self, frame, hand_center, target_center, guidance_data):
        """
        Visualizes the trajectory arrow, distance text, and direction command on the frame.
        """
        if (
            guidance_data is None
            or hand_center is None
            or target_center is None
        ):
            return frame

        hx, hy = hand_center
        tx, ty = target_center
        distance = guidance_data["distance"]
        command = guidance_data["command"]

        if command == "REACHED":
            # Highlight target reached with green circle and text
            cv2.circle(frame, (tx, ty), self.tolerance_radius, (0, 255, 0), 2)
            cv2.putText(
                frame,
                "TARGET REACHED!",
                (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
            )
        else:
            # Draw dynamic trajectory arrow from hand to target (Cyan)
            cv2.arrowedLine(
                frame,
                (hx, hy),
                (tx, ty),
                (255, 255, 0),
                2,
                tipLength=0.04,
            )

            # Draw tolerance threshold zone around target
            cv2.circle(
                frame, (tx, ty), self.tolerance_radius, (255, 255, 0), 1
            )

            # Overlay navigation guidance command
            cv2.putText(
                frame,
                f"ACTION: {command} | DIST: {int(distance)}px",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )

        return frame