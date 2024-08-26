# install the required version of Ultralytics
!pip install ultralytics==8.0.28


# Use Roboflow for Custom Datasets
# dataset이 있다고 가정 (roboflow)

import os
from IPython.display import Image

# Home Directory
%cd {os.path.expanduser("~")}

# predictions with YOLOv8
!yolo task=segment mode=predict model=yolov8s-seg.pt conf=0.25 source='https://media.roboflow.com/user path' save=true

Image(filename='user path', height=600)


!mkdir {HOME}/datasets
%cd {HOME}/datasets

!pip install roboflow --quiet

# Import Roboflow classes from the Roboflow library
from roboflow import Roboflow

# Create a Roboflow object using the user's API key
rf = Roboflow(api_key="YOUR_API_KEY")

# Access the specified workspace and project
project = rf.workspace("alpaco5-f3woi").project("part-autolabeld")

# Download data corresponding to version 5 of the project in YOLOv8 format
dataset = project.version(5).download("yolov8")


# Training YOLOv8 Instance Segmentation Model
%cd {HOME}
!yolo task=segment mode=train model=yolov8s-seg.pt  data={dataset.location}/data.yaml epochs=10 imgsz=640!ls {HOME}/runs/segment/train/



# Visualization
%cd {HOME}
Image(filename=f'{HOME}/runs/segment/train/confusion_matrix.png', width=600)

%cd {HOME}
Image(filename=f'{HOME}/runs/segment/train/results.png', width=600)

%cd {HOME}
Image(filename=f'{HOME}/runs/segment/train/val_batch0_pred.jpg', width=600)




# Predict with Your Custom YOLOv8 Instance Segmentation Model
!yolo task=segment mode=val model={HOME}/runs/segment/train/weights/best.pt data={dataset.location}/data.yaml

!yolo task=segment mode=predict model={HOME}/runs/segment/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=true



# Deploy the Trained Model to Roboflow
# retrieve your workspace and project names
rf = Roboflow(api_key="API_KEY")
print(rf.workspace())


# upload the trained model back to your desired project in Roboflow
project = rf.workspace("roboflow-ngkro").project("car-parts-ecsse")
project.version(3).deploy("yolov8", "runs/segment/train/")
