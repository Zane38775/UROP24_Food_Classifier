import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# 加载保存的模型权重
from food_classifier import FoodClassifier  # 导入您的模型类
model = FoodClassifier(num_classes=101)
model.load_state_dict(torch.load('food_classifier.pth'))
model.eval()

# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载类别标签
data_dir = os.path.join('data', 'food-101')
with open(os.path.join(data_dir, 'meta', 'classes.txt'), 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 获取每一帧
    ret, frame = cap.read()

    # 将OpenCV图像转换为PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 预处理图像
    img = preprocess(img).unsqueeze(0)

    # 进行预测
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        label = class_labels[predicted.item()]

    # 在图像上显示预测结果
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Food Classification', frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()