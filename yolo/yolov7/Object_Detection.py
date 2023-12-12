import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel


imgsz=480
conf_thres=0.3
iou_thres=0.3
device=''
classes=None
agnostic_nms=False
# Directories
# increment run

# Initialize
# set_logging()
device = select_device("cuda" if torch.cuda.is_available() else "cpu")
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
WEIGHTS = "D:/best.pt"
model = attempt_load(WEIGHTS, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

if True:
    model = TracedModel(model, device, imgsz)

if half:
    model.half()  # to FP16


# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


def detect(source):
    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    for path, img, im0s, _ in dataset:
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)

        # Process detections
        for det in pred:  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                boundingbox = det.int().detach().cpu().numpy()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                return boundingbox, im0




a,b=detect("D:/dataset/test/images/d1ea6407-f000000017_png.rf.f292506a34acc49cfda2ebe1b05e097d.jpg")

print(a)
import cv2
cv2.imshow('1',b)
cv2.waitKey(0)
cv2.destroyAllWindows()