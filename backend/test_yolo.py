import sys
import cv2
from ultralytics import YOLO

model = YOLO('/app/models/yolo11n.pt')

# Wait, I need the panda image. I'll download a dummy image of a panda or just check the mock mode output.
print("YOLO model loaded.")
