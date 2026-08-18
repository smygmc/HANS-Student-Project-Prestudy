import cv2
import numpy as np
import torch


class DepthEstimator:

    def __init__(self, model_type="MiDaS_small"):
      """
      Loads the MiDaS depth estimation model and its transformation pipeline.
      """
      # select hardware
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      # download and load the pre-trained modelub
      self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
      # move model to the selected device
      self.midas.to(self.device)
      # set model to evaluation mode for inference
      self.midas.eval()

      # load the corresponding input transformation pipeline
      midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
      if model_type in ["DPT_Large", "DPT_Hybrid"]:
        self.transform = midas_transforms.dpt_transform
      else:
        self.transform = midas_transforms.small_transform


    def estimate_depth(self, frame):
      """generates depth map using the 2d frame as input"""
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      input_batch = self.transform(img).to(self.device)

      # perform inference without computing gradients
      with torch.no_grad():
        prediction = self.midas(input_batch)
        # resize depth prediction back to original frame dimensions
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
      # move tensor to CPU and convert to NumPy 2D array
      depth_map = prediction.cpu().numpy()
      return depth_map

    def get_depth_at_point(self, depth_map, point):
      """Extracts depth value at a specific pixel (x, y).

      Returns a median of a small window to avoid single-pixel noise.
      """
      if point is None or depth_map is None:
        return None

      # extract coordinates and get map dimensions
      x, y = int(point[0]), int(point[1])
      h, y_limit = depth_map.shape[:2]

      # define a 5x5 bounding window with boundary clamping
      x_min, x_max = max(0, x - 2), min(y_limit, x + 3)
      y_min, y_max = max(0, y - 2), min(h, y + 3)
      # extract local neighborhood patch
      patch = depth_map[y_min:y_max, x_min:x_max]
      if patch.size == 0:
        return float(depth_map[y, x])

      return float(np.median(patch))

    