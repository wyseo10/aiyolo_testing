from ultralytics import YOLO
import cv2
import argparse
import numpy as np
import time
import socket, struct

# YOLO 모델 불러오기
model = YOLO("aiyolo/yolo11n.pt")

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
video_path = "/home/wy/aiyolo/output.mp4"

while True:
    #Get info
    packetInfoRaw = rx_bytes(4)
    #print(packetInfoRaw)
    [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
    #print("Length is {}".format(length))
    #print("Route is 0x{:02X}->0x{:02X}".format(routing & 0xF, routing >> 4))
    #print("Function is 0x{:02X}".format(function))

    imgHeader = rx_bytes(length - 2)
    #print(imgHeader)
    #print("Length of data is {}".format(len(imgHeader)))
    [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)

    if magic == 0xBC:
      #print("Magic is good")
      #print("Resolution is {}x{} with depth of {} byte(s)".format(width, height, depth))
      #print("Image format is {}".format(format))
      #print("Image size is {} bytes".format(size))

      # Now we start rx the image, this will be split up in packages of some size
      imgStream = bytearray()

      while len(imgStream) < size:
          packetInfoRaw = rx_bytes(4)
          [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
          #print("Chunk size is {} ({:02X}->{:02X})".format(length, src, dst))
          chunk = rx_bytes(length - 2)
          imgStream.extend(chunk)
     
      count = count + 1
      meanTimePerImage = (time.time()-start) / count
      print("Frame rate : {:.2f} fps".format(1 / meanTimePerImage))

      if format == 0:
          bayer_img = np.frombuffer(imgStream, dtype=np.uint8)
          bayer_img.shape = (244, 324)
          color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGRA)

          cv2.imshow('Raw', bayer_img)
          cv2.imshow('Color', color_img)
          cv2.waitKey(1)
      else:
          nparr = np.frombuffer(imgStream, np.uint8)
          decoded = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
          cv2.imshow('JPEG', decoded)
          cv2.waitKey(1)

          if args.save and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 10, (decoded.shape[1], decoded.shape[0]))
            if not video_writer.isOpened():
               print("VideoWriter failed to open!")
               break
          if args.save and video_writer is not None:
             video_writer.write(decoded)
if video_writer is not None:
   video_writer.release()
client_socket.close()

