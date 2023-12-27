# AIoT Beehive Monitoring System

This repository hosts our AIoT Beehive Monitoring project, aiming to enhance beekeeping with AI and IoT. Our web application provides insights into hive health, bee counting, and wasp detection using image classification and object detection. We explore diverse approaches, including the [BeeAlarmed project](https://github.com/BeeAlarmed/BeeAlarmed) integration and YOLOv7 deployment, showcasing the potential of AI in beekeeping.

Here is our project report: [Link](https://drive.google.com/file/d/1mBOaYGjJRPZtV_cR9se_-DsUp4MsGhS4/view?usp=sharing)

All info regarding our project can be found within the report. This README only contains pointers to where our individual codes are.

## Bee Image classification

The first core component of our system is image classification, empowering us to discern various bee activities through advanced deep learning neural network models.

### Multi-class classification

Our multi-class classification model and training code can be found on the `training_method_2` branch: [training_method_2:classification.ipynb](https://github.com/WataNekko/aiot-beehive/blob/training_method_2/classification.ipynb)

This model was originally our backup plan and only served for our learning purposes.

### Multi-label classification

Our primary focus lies in multi-label classification, where each bee is categorized based on distinct activities and statuses. This pivotal aspect enables binary classification for each label, providing comprehensive insights into the diverse behaviors of bees.

#### MobileNet-based model

This is our attempt to build our own CNN model based on the MobileNet image classification model. The result, as stated in our report, is not so good in accuracy so this model was not used for our final application.

The model building and training code for this can be found in the [`classification` folder](classification/).

#### BeeAlarmed-based model (BeeNet)

This is a classification model based on the BeeAlarmed project's approach to classify bees. This integrated nicely with the rest of our system, so this will be our main classification model.

The model building and training code for this can be found in the [`BeeAlarmed-classification` folder](BeeAlarmed-classification/).

## Bee detection

The bee detection component of our system is crucial for isolating individual bees within hive images, laying the foundation for subsequent classification. Following are the methods employed to identify and extract bees from images, paving the way for detailed analysis of their activities and health.

### OpenCV detection

This approach uses OpenCV to detect bee in an input image. This is inspired by the BeeAlarmed project's approach to object detection.

The code can be found in the [`BeeAlarmed-detection` folder](BeeAlarmed-detection/).

### YOLOv7 detection

This approach uses the more advanced and accurate YOLOv7 deep learning model for object detection. The code to this approach can be found on the `training_method_2` branch.

YOLOv7 model training code: [training_method_2:Object_Detection_training.ipynb](https://github.com/WataNekko/aiot-beehive/blob/training_method_2/Object_Detection_training.ipynb)

YOLOv7 object detection code (model usage): [training_method_2:yolo/yolov7/Object_Detection.py](https://github.com/WataNekko/aiot-beehive/blob/training_method_2/yolo/yolov7/Object_Detection.py)

## Prediction Website

What ties everything together is our website, which incorporates the afformentioned BeeAlarmed-powered object detection and classification, and are optionally enhanced with the YOLOv7 component for more effective object detection. The system is designed to perform predictions on static images of beehive entrances and present the results in multiple visual graphs. You can access the source code at [`./website`](website/) or through the following [link](https://github.com/18874studentvgu/M-24_AIBeeSite/).
