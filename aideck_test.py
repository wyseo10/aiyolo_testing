from ultralytics import YOLO
import cv2
import argparse
import numpy as np
import time
import socket, struct

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
     
      count = count + 1
      meanTimePerImage = (time.time()-start) / count
      print("Frame rate : {:.2f} fps".format(1 / meanTimePerImage))

      if format == 0:
          bayer_img = np.frombuffer(imgStream, dtype=np.uint8)
          bayer_img.shape = (244, 324)
          color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGR)
      else:
        #JPEG format image
          nparr = np.frombuffer(imgStream, np.uint8)
          decoded = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
      
      #Img Center Position
      cam_center_x = bayer_img.shape[1] // 2
      cam_center_y = bayer_img.shape[0] // 2
      cam_center_img = (cam_center_x, cam_center_y)
      print(f"Screen Center : {cam_center_img}")
      cv2.circle(color_img, cam_center_img, 2, (0, 0, 255), -1)

      #YOLO detecting 및 results
      results = model(color_img)

      for result in results:
        boxes = result.boxes
        for box in boxes:
          x1, y1, x2, y2 = box.xyxy[0]
          confidence = box.conf[0]
          class_id = box.cls[0]

          print(f"Class ID : {class_id}, Confidence : {confidence}")
          print(f"Bounding Box : [{x1}, {y1}, {x2}, {y2}]")

          cv2.rectangle(color_img, (int(x1), int(y1), int(x2), int(y2)), (0,255,0),2)

      annotated_img = results[0].plot()
      
      # result output.mp4로 저장
      if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('/home/wy/aiyolo/runs/output.mp4', fourcc, 10, (annotated_img.shape[1], annotated_img.shape[0]))
      video_writer.write(annotated_img)

      # 화면 출력부
      cv2.imshow("YOLO Detection", annotated_img)
      if args.save:
        cv2.imwrite(f"/home/wy/aiyolo/frame_{count:06d}.jpg", annotated_img)

      if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

if video_writer:
    video_writer.release()
