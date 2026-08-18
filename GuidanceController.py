import math
import cv2
from DepthEstimator import DepthEstimator

class GuidanceController:
    """
    Computes 3D spatial guidance metrics between the hand center and target object.
    """

    def __init__(self, tolerance_radius=45,z_tolerance=20.0,model_type="MiDaS_small",):
        
        # 2D pixel tolerance
        self.tolerance_radius = tolerance_radius
        # 3D depth tolerance
        self.z_tolerance = z_tolerance
        # Embedded depth estimation model
        self.depth_estimator = DepthEstimator(model_type=model_type)

    def estimate_scene_depth(self, frame):
        """Generates raw depth map and colored heatmap for visualization."""
        depth_map = self.depth_estimator.estimate_depth(frame)
        
        return depth_map
    
    def compute_guidance(self, hand_center, target_center, depth_map=None):
        """
        Calculates 3D euclidian distance, delta vector, and navigation command.
        
        """
        if hand_center is None or target_center is None:
            return None

        hx, hy = hand_center
        tx, ty = target_center

        # Delta vector target-hand (screen orijin is top left)
        dx = tx - hx
        dy = ty - hy

        # Depth (Z) extraction and delta
        hand_z = None
        target_z = None
        dz = 0.0

        if depth_map is not None:
            hand_z = self.depth_estimator.get_depth_at_point(depth_map, (hx, hy))
            target_z = self.depth_estimator.get_depth_at_point(depth_map, (tx, ty))
            if hand_z is not None and target_z is not None:
                dz = target_z - hand_z

        # Euclidean distance-distance
        distance_2d = math.sqrt(dx**2 + dy**2)
        distance_3d = math.sqrt(dx**2 + dy**2 + dz**2)

        # Check if hand is within tolerance threshold on all axes
        is_2d_reached = distance_2d <= self.tolerance_radius
        is_z_reached = abs(dz) <= self.z_tolerance
        # Check if hand is within tolerance threshold
        if is_2d_reached and is_z_reached:
            direction_command = "REACHED"
        else:
            # Determine dominant direction axis to give only one direction command at a time
            abs_dx = abs(dx)
            abs_dy = abs(dy)
            abs_dz = abs(dz)
            
            if abs_dz > abs_dx and abs_dz > abs_dy and not is_z_reached:
                #near objects > far objects
                direction_command = "FORWARD" if dz < 0 else "BACKWARD"
            elif abs(dx) > abs(dy):
                # dx>0 target on the right
                direction_command = "RIGHT" if dx > 0 else "LEFT"
            else:
                # dy>0 target is below
                direction_command = "DOWN" if dy > 0 else "UP"
            

        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "distance": round(distance_3d, 2),
            "distance_2d": round(distance_2d, 2),
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
        dz = guidance_data["dz"]
        

        
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

            # Overlay navigation guidance command with dZ info
            cv2.putText(
                frame,
                f"ACTION: {command} | DIST: {int(distance)} | dZ: {dz}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )

        return frame