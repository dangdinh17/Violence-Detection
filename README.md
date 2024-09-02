# Violence-Detection

## A simple intro to use the pre-trained model

#### Step 1

If you want to retrain the model, please download the dataset from [here](https://www.kaggle.com/datasets/anginhok/dataset)

The structure of the dataset will be like this:
```
dataset/
        training/
              Fight/
              NonFight/
        val/
              Fight/
              NonFight/
        test/
              Fight/
              NonFight/
```
#### Step 2

To retrain the project with the Flow Gated Model, use the code file [here](train_flow_gated_models.ipynb)

To retrain the project with the YOLOv8, use the code file [here](train_yolov8.ipynb)

#### Step 3

The pre-trained models are implemented by Keras and PyTorch, and you can load it easily by using the below codes. 

```python
from keras.models import load_model
from keras.optimizers import SGD
from models.flow_gated_models import *
from ultralytics import YOLO

#Flow Gated Models
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

fgm = FlowGatedModels().build_model()
fgm.load_weights('./weights/best_models_fgm.h5')

fgm.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#YOLOv8
yolo = YOLO('./weights/best_yolov8.h5')
```
#### Step 4
To test the performance of the models in this project, please run the [demo](demo.py) file code with the installed necessary package in the [requirement](requirements.txt) file.

### License
A part of this code is implemented in the project [RWF2000 - A Large Scale Video Database for Violence Detection](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection/blob/master/README.md?plain=1), so if you want to use this code in your paper, please cite this:
```
@INPROCEEDINGS{9412502,
     author={Cheng, Ming and Cai, Kunjing and Li, Ming},
     booktitle={2020 25th International Conference on Pattern Recognition (ICPR)}, 
     title={RWF-2000: An Open Large Scale Video Database for Violence Detection}, 
     year={2021},
     volume={},
     number={},
     pages={4183-4190},
     doi={10.1109/ICPR48806.2021.9412502}}
```