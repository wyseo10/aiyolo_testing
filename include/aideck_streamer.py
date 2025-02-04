import socket
import struct
import numpy as np
import cv2
import argparse

class AIDeckStreamer:
    def __init__(self, cam_width=162, cam_height=162):
        # AI-deck IP/port 불러오기
        parser = argparse.ArgumentParser(description='Connect to AI-deck streamer')
        parser.add_argument("-n",  default="192.168.4.1", metavar="ip", help="AI-deck IP")
        parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
        parser.add_argument('--save', action='store_true', help="Save streamed images")
        args = parser.parse_args()

        self.ip = args.n
        self.port = args.p
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.client_socket = None

    # Connecting Aideck
    def connect(self):
        print(f"Connecting to socket on {self.ip}:{self.port}...")
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.ip, self.port))
        print("Socket Connected")
        
    def rx_bytes(self, size):
        data = bytearray()
        while len(data) < size:
            data.extend(self.client_socket.recv(size-len(data)))
        return data 
    
    #Getting img from Aideck
    def get_frame(self):
        packetInfoRaw = self.rx_bytes(4)
        [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)

        img_header = self.rx_bytes(length - 2)
        [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', img_header)

        if magic != 0xBC:
            return None
            
        img_stream = bytearray()
        while len(img_stream) < size:
            packetInfoRaw = self.rx_bytes(4)
            [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
            img_stream.extend(self.rx_bytes(length - 2))

        if format == 0:
            # Bayer format -> BGR 
            bayer_img = np.frombuffer(img_stream, dtype=np.uint8)
            bayer_img.shape = (self.cam_height, self.cam_width)
            color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGR)
        else:
            # JPEG format
            nparr = np.frombuffer(img_stream, np.uint8)
            color_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        return color_img