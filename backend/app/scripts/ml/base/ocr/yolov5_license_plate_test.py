# 使用YOLOv5定位车牌
from ultralytics import YOLO

yolo_model = YOLO('yolov5_license_plate.pt')
results = yolo_model('license_plate/licence-plate.jpg')
plate_boxes = results[0].boxes.xyxy.cpu().numpy()

# 提取车牌区域并输入EasyOCR
for box in plate_boxes:
    x1, y1, x2, y2 = box[:4].astype(int)
    plate_img = image[y1:y2, x1:x2]
    text = reader.readtext(plate_img, detail=0)[0]
    print("车牌号:", text)