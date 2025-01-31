from ultralytics import YOLO
import cv2
import argparse
import numpy as np
import time
import math
import socket, struct
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

# 카메라 설정
cam_width = 162
cam_height = 162
min_confidence = 0.5
window_size = 10

# 필터
moving_average_x = MovingAverage(window_size)
moving_average_y = MovingAverage(window_size)
moving_average_width = MovingAverage(window_size)
moving_average_height = MovingAverage(window_size)

# YOLO 모델 불러오기
model = YOLO("yolo11n.pt")

# AI-deck IP/port 불러오기
parser = argparse.ArgumentParser(description='Connect to AI-deck streamer')
parser.add_argument("-n",  default="192.168.4.1", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
parser.add_argument('--save', action='store_true', help="Save streamed images")
args = parser.parse_args()

deck_port = args.p
deck_ip = args.n

print("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((deck_ip, deck_port))
print("Socket connected")

imgdata = None
data_buffer = bytearray()

def rx_bytes(size):
  data = bytearray()
  while len(data) < size:
    data.extend(client_socket.recv(size-len(data)))
  return data

start = time.time()
count = 0
video_writer = None

while True:
    #Get info
    packetInfoRaw = rx_bytes(4)
    [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)

    imgHeader = rx_bytes(length - 2)
    [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)

    if magic == 0xBC:
      imgStream = bytearray()
      while len(imgStream) < size:
          packetInfoRaw = rx_bytes(4)
          [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
          chunk = rx_bytes(length - 2)
          imgStream.extend(chunk)
     
      count += 1
      meanTimePerImage = (time.time()-start) / count
      print("Frame rate : {:.2f} fps".format(1 / meanTimePerImage))

      if format == 0:
          bayer_img = np.frombuffer(imgStream, dtype=np.uint8)
          bayer_img.shape = (cam_height, cam_width) 
          color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGR)
      else:
        #JPEG format image
          nparr = np.frombuffer(imgStream, np.uint8)
          decoded = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
      
      #Img Center Position
      cam_center_x = bayer_img.shape[1] // 2
      cam_center_y = bayer_img.shape[0] // 2
      cam_center_img = (cam_center_x, cam_center_y)
      #print(f"Screen Center : {cam_center_img}")
      cv2.circle(color_img, cam_center_img, 2, (0, 0, 255), -1)

      #YOLO detecting 및 results
      results = model(color_img)

      max_box = {
        "center_x": 0,
        "center_y": 0,
        "width": 0,
        "height": 0,
        "confidence": 0,
        "class_name": 0,
        "found": False
      }

      for result in results:
        boxes = result.boxes

        for box in boxes:
          box_center_x, box_center_y, width, height = box.xywh[0]
          confidence = box.conf[0]
          class_id = int(box.cls[0])
          class_name = model.names[class_id]

          if class_id == 0 and confidence > min_confidence and confidence > max_box['confidence']:
            max_box.update({
              "center_x": box_center_x,
              "center_y": box_center_y,
              "width": width,
              "height": height,
              "confidence": confidence,
              "class_name": class_name,
              "found": True
            })

        if max_box["found"]:
          #안정화된 box_center_x 계산
          stabilized_x = moving_average_x.update(max_box["center_x"])
          stabilized_y = moving_average_y.update(max_box["center_y"])
          stabilized_width = moving_average_width.update(max_box["width"])
          stabilized_height = moving_average_height.update(max_box["height"])
          #distance_x = stabilized_x - cam_center_x
          #distance_y = box_center_y - cam_center_y
          #euclidean_distance = math.sqrt(distance_x**2 + distance_y**2)

          #print(f"Class ID : {class_id}, Confidence : {confidence}")
          #print(f"box_center:({box_center_x},{box_center_y})")
          print(f"box_center_x = {max_box['center_x']}")
          print(f"Stabilized x = {stabilized_x:.4f}")
          #print(f"Distance : (x,y) = ({distance_x},{distance_y}), eucl : {euclidean_distance}")

        stabilized_x = moving_average_x.get_stabilized_value()
        stabilized_y = moving_average_y.get_stabilized_value()
        stabilized_w = moving_average_width.get_stabilized_value()
        stabilized_h = moving_average_height.get_stabilized_value()
        
        x1 = int(stabilized_x - (stabilized_w / 2))
        y1 = int(stabilized_y - (stabilized_h / 2))
        x2 = int(stabilized_x + (stabilized_w / 2))
        y2 = int(stabilized_y + (stabilized_h / 2))
            
        cv2.circle(color_img, (int(stabilized_x), int(stabilized_y)), 2,(0,0,255),-1)
        cv2.rectangle(color_img, (x1, y1),(x2, y2), (0, 255, 0), 2)
        cv2.putText(color_img,f"{max_box['class_name']} {max_box['confidence']:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
      
      # result output.mp4로 저장(수정)
      if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('/home/wy/aiyolo/runs/output.mp4', fourcc, 10, (color_img.shape[1], color_img.shape[0]))
      video_writer.write(color_img)

      # 화면 출력부
      cv2.imshow("YOLO Detection", color_img)
      if args.save:
        cv2.imwrite(f"/home/wy/aiyolo/frame_{count:06d}.jpg", color_img)
      cv2.waitKey(1)