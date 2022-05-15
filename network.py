import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
class DNN(nn.Module):
    def __init__(self):
        super().__init__()

        #convolutional layers
        #self.conv1 = nn.Conv2d(320, 318, kernel_size=(3,1))
        #self.conv2 = nn.Conv2d(318, 157, kernel_size=(3,1))
        #self.conv3 = nn.Conv2d(157, 155, kernel_size=(3,1))
        #self.conv4 = nn.Conv2d(155, 77, kernel_size=(3,1))
        #self.conv5 = nn.Conv2d(77, 38, kernel_size=(3,1))

        #batch normalizing layers
        #self.norm1 = nn.BatchNorm2d(318)
        #self.norm2 = nn.BatchNorm2d(157)
        #self.norm3 = nn.BatchNorm2d(155)
        #self.norm4 = nn.BatchNorm2d(77)
        #self.norm5 = nn.BatchNorm2d(38)

        #yolo net
        self.yolo = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        self.yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)


        
    def forward(self, x, image = False):
        size = (704, 640)

        #x = F.max_pool2d(F.relu(self.norm1(self.conv1(x))), (2, 1))
        #x = F.max_pool2d(F.relu(self.norm2(self.conv2(x))), (2, 1))
        #x = F.max_pool2d(F.relu(self.norm3(self.conv3(x))), (2, 1))
        #x = F.max_pool2d(F.relu(self.norm4(self.conv4(x))), (2, 1))
        #x = F.relu(self.norm5(self.conv5(x)))

        ln = self.yolo.getLayerNames()
        ln = [ln[i - 1] for i in self.yolo.getUnconnectedOutLayers()]

        if(not image):
            y = cv.dnn.blobFromImage(x.detach().numpy(), 1/255.0, size, swapRB=True, crop=True)
            self.yolo.setInput(y)
        else:
            y = cv.dnn.blobFromImage(x, 1/255.0, size, swapRB=True, crop=True)
            self.yolo.setInput(y)

        return self.yolo.forward(ln)
