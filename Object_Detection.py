import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

path = "D:/best.pt"
model = torch.hub.load("WongKinYiu/yolov7", "custom", f"{path}", trust_repo=True)
model.conf = 0.3  # Set the conf-thres to 0.3
model.iou = 0.3  # Set the iou-thres to 0.3

# Load the weights from the 'best.pt' file
model.eval()

# Load an image for prediction
image = Image.open("D:/058ae8ff-f000000030_png.rf.34a7967630618a23ca4ab991e4ceaa5c.jpg")

# Preprocess the image
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((480, 480)),  # Resize the image to (480, 480)
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_image = transform(image).unsqueeze(0)

# Perform prediction
with torch.no_grad():
    output = model(input_image)
    print(output)

# Process the output
# ...

# Display the image and predicted bounding boxes
# ...
