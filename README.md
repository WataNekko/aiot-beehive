# AIoT Beehive Monitoring System

This repository hosts our AIoT Beehive Monitoring project, aiming to enhance beekeeping with AI and IoT. Our web application provides insights into hive health, bee counting, and wasp detection using image classification and object detection. We explore diverse approaches, including the [BeeAlarmed project](https://github.com/BeeAlarmed/BeeAlarmed) integration and YOLOv7 deployment, showcasing the potential of AI in beekeeping.

Here is our project report: [Link](https://drive.google.com/file/d/1mBOaYGjJRPZtV_cR9se_-DsUp4MsGhS4/view?usp=sharing)

All info regarding our project can be found within the report. This README only contains pointers to where our individual codes are.

## Bee Image classification

The first core component of our system is image classification, empowering us to discern various bee activities through advanced deep learning neural network models.

### Multi-class classification

Our multi-class classification model and training code can be found on the trainining_method_2 branch: [training_method_2:classification.ipynb](https://github.com/WataNekko/aiot-beehive/blob/training_method_2/classification.ipynb)

This model was originally our backup plan and only served for our learning purposes.

## Prediction Website

What ties everything together is our website, which incorporates the afformentioned BeeAlarmed-powered object detection and classification, and are optionally enhanced with the YOLOv7 component for more effective object detection. The system is designed to perform predictions on static images of beehive entrances and present the results in multiple visual graphs. You can access the source code at `./website` or through the following [link](https://github.com/18874studentvgu/M-24_AIBeeSite/).
