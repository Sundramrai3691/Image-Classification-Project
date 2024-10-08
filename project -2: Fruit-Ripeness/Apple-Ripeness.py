from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

cap = cv2.VideoCapture('../Videos/apple-3.mp4')

model = YOLO('Apple.pt')

classNames = ['apple', 'damaged_apple']
myColor = (0,0,255)

while True:
    success, img = cap.read()
    img = cv2.resize(img,(1280,720))
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w,h = x2-x1, y2-y1


            #Confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf>0.5:
                if currentClass == 'apple':
                    myColor = (0, 255, 0)

                else:
                    myColor = (0, 0, 255)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2,
                                   thickness=1, colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)


