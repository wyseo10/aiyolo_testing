from ultralytics import YOLO
from collections import deque

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, new_value):
        self.values.append(new_value)
        return self.calculate_average()

    def calculate_average(self):
        return sum(self.values) / len(self.values) if self.values else 0
    
    def get_stabilized_value(self):
       return self.calculate_average()

class ObjectDetector:
  def __init__(self, model_path="", min_conf=0.5, window_size = 10):
      self.model = YOLO(model_path)
      self.min_confidence = min_conf

      self.moving_avg_x = MovingAverage(window_size)
      self.moving_avg_y = MovingAverage(window_size)
      self.moving_avg_w = MovingAverage(window_size)
      self.moving_avg_h = MovingAverage(window_size)
    
  def detect(self, image):
    results = self.model(image)[0] 
    max_box = {"found": False}
     
    for box in results.boxes:
        box_center_x, box_center_y, width, height = box.xywh[0]
        confidence = box.conf[0]
        class_id = int(box.cls[0])

        if class_id == 0 and confidence > self.min_confidence:
           if not max_box["found"] or confidence > max_box["confidence"]:
              max_box.update({
                "center_x": box_center_x,
                "center_y": box_center_y,
                "width": width,
                "height": height,
                "confidence": confidence,
                "class_name": self.model.names[class_id],
                "found": True
              })
    return self.stabilize(max_box)

  def stabilize(self, max_box):
    if max_box["found"]:
      max_box["center_x"] = self.moving_avg_x.update(max_box["center_x"])
      max_box["center_y"] = self.moving_avg_y.update(max_box["center_y"])
      max_box["width"] = self.moving_avg_w.update(max_box["width"])
      max_box["height"] = self.moving_avg_h.update(max_box["height"])
    return max_box