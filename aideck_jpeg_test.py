from ultralytics import YOLO
import cv2
import argparse
import numpy as np
import time
import math
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
    # Get info
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
        nparr = np.frombuffer(imgStream, np.uint8)
        color_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        # JPEG color_img.shape check
        if color_img is None:
            print("[ERROR] JPEG 디코딩 실패")
            continue        
        if len(color_img.shape) == 2:  # Grayscale image
            color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)
        if color_img.shape[2] != 3:
            print(f"[ERROR] 이미지가 3채널이 아님: {color_img.shape}")
            continue
        #Checking ends
        
        count += 1
        meanTimePerImage = (time.time() - start) / count
        print("Frame rate: {:.2f} fps".format(1 / meanTimePerImage))

        # Center position of the camera view
        cam_center_x = color_img.shape[1] // 2
        cam_center_y = color_img.shape[0] // 2
        cam_center_img = (cam_center_x, cam_center_y)
        cv2.circle(color_img, cam_center_img, 2, (0, 0, 255), -1)

        # YOLO detection
        results = model(color_img)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                box_center_x, box_center_y, width, height = box.xywh[0]
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                distance_x = box_center_x - cam_center_x
                distance_y = box_center_y - cam_center_y
                euclidean_distance = math.sqrt(distance_x**2 + distance_y**2)

                cv2.circle(color_img, (int(box_center_x), int(box_center_y)), 2, (0, 0, 255), -1)

        # Annotated image
        annotated_img = results[0].plot()

        # Save video
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter('/home/wy/aiyolo/runs/output.mp4', fourcc, 10,
                                            (annotated_img.shape[1], annotated_img.shape[0]))
        video_writer.write(annotated_img)

        # Display the image
        cv2.imshow("YOLO Detection", annotated_img)
        if args.save:
            cv2.imwrite(f"/home/wy/aiyolo/frame_{count:06d}.jpg", annotated_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        