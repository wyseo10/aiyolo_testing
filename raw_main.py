import cv2
import argparse
import numpy as np
import time
import math
import socket, struct
from include.object_detector import ObjectDetector
from include.video_recorder import VideoRecorder


# VideoRecorder 변수들
cam_width = 162
cam_height = 162

detector = ObjectDetector()
recorder = VideoRecorder()

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
          decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
      
      #Img Center Position
      cam_center_x = bayer_img.shape[1] // 2
      cam_center_y = bayer_img.shape[0] // 2
      cam_center_img = (cam_center_x, cam_center_y)
      cv2.circle(color_img, cam_center_img, 2, (0, 0, 255), -1)

      #YOLO detecting 및 stabilized
      max_box = detector.detect(color_img)

      #Draw box
      detector.draw_box(color_img, max_box)
    
      #.webm file video recorder
      recorder.write_frame(color_img)

      # 화면 출력부
      cv2.imshow("YOLO Detection", color_img)
      cv2.waitKey(1)