import cv2
import time
from include.object_detector import ObjectDetector
from include.video_recorder import VideoRecorder
from include.aideck_streamer import AIDeckStreamer

# Include modules
detector = ObjectDetector()
recorder = VideoRecorder()
streamer = AIDeckStreamer()

# Connect Aideck
streamer.connect()

imgdata = None
data_buffer = bytearray()

start = time.time()
count = 0

while True:
      color_img = streamer.get_frame()
      if color_img is None:
          continue
      
      count += 1
      meanTimePerImage = (time.time()-start) / count
      print("Frame rate : {:.2f} fps".format(1 / meanTimePerImage))
      
      # Show img center position
      cam_center_x = color_img.shape[1] // 2
      cam_center_y = color_img.shape[0] // 2
      cam_center_img = (cam_center_x, cam_center_y)
      cv2.circle(color_img, cam_center_img, 2, (0, 0, 255), -1)

      # YOLO detecting & stabilized
      max_box = detector.detect(color_img)

      # Draw detecting box
      detector.draw_box(color_img, max_box)
    
      # Webm file video recorder
      recorder.write_frame(color_img)

      # 화면 출력부
      cv2.imshow("YOLO Detection", color_img)
      cv2.waitKey(1)