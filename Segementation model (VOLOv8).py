from ultralytics import YOLO

# load a pre-trained model
model = YOLO("yolov8n-seg.pt")

# train the model
model.train(data="data-seg.yaml", epochs=100, imgsz=640)

# predict with the model
results = model.predict(source="test.jpg")



# install YOLOv8
pip install ultralytics

# train model
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
model.train('data=mydata.yaml', epochs=10)

# prediction
results = model.predict(source='/content/test/')



!weget -0 data.zip https://universe.roboflow.com/custom data path

import zipfile
with zipfile.zipfile('/content/custom data.zip') as target_file:
  target_file.extractall('/content/custom data')

!cat /content/custom data/data.yaml


!pip install PyYAML
import yaml
data = { 'train': '/content/custom data/train/images'
         'val': '/content/custom data/vaild/images'
         'test': '/content/custom data/test/images'
         'names': ['trash']
         'nc': 1 }

with open('/content/custom data/custom data.yaml', 'w') as f:
  yaml.dump(data, f)

with open('/content/custom data/custom data.yaml', 'r') as f:
  taco_yaml = yaml.safe_load(f)
  display(custom data_yaml)


!pip install ultralytics
import ultralytics
ultralytics.checks()


# Load YOLOv8n-seg
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

print(type(model.names), len(model.names))
print(model.names)



model.train*(data='/content/custom data/custom data.yaml', epochs=50, patience=10, batch=32, imgsz=416)
print(type(model.names), len(model.names))
print(model.names)


results = model.predict(source='/content/custom data/test/images', save = True)
