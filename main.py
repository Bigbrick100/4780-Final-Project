from network import DNN
import cv2 as cv
import os
import numpy as np
import sys
import math

classes = open('coco.names').read().strip().split('\n')
colors = {"safe": (0, 255, 0), "unsafe": (0, 0, 255)}
ftperpx = 35.0/1800.0

def calcDist(myself, other):
    (x1, y1, w1, h1) = myself["x"], myself["y"], myself["w"], myself["h"]
    (x2, y2, w2, h2) = other["x"], other["y"], other["w"], other["h"]
    heightratio = h1/h2 if h2 > h1 else h2/h1
    c1 = ((x1+x1+w1)/2, y1+y1+h1/2)
    c2 = ((x2+x2+w2)/2, y2+y2+h2/2)
    return math.sqrt((c1[0] - c2[0])**2+(c1[1] - c2[1])**2)*ftperpx/heightratio

def drawRects(outputs, img):
    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]
    people = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores, axis=None, out=None)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            if not (classes[classIDs[i]] == "person"):
                continue
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            people.append({"x": x, "y": y, "w": w, "h": h})

    
    for myself in people:
        closest = sys.maxsize
        for other in people:
            if other == myself:
                continue
            dist = calcDist(myself, other)
            if dist < closest:
                closest = dist

        safe, unsafe = "safe", "unsafe"
        (x, y, w, h) = myself["x"], myself["y"], myself["w"], myself["h"]
        color = colors[safe] if closest > 6 else colors[unsafe]
        text = safe if closest > 6 else unsafe
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            

def test(network, dir):
    for filename in os.listdir(dir):
        if filename == "desktop.ini":
            continue
        img = cv.imread(dir+"/"+filename)
        output = network(img, True)
        drawRects(output, img)
        while (1):
            cv.imshow("Preview", img)

            k = cv.waitKey(10) & 0xFF
            if k == ord(' '):
                break
            if k == ord('z'):
                exit(0)

def main():
    network = DNN()
    test(network, "./test_images")

if __name__ == '__main__':
    main()