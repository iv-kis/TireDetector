# Vehicle tire detector based on YOLO v.4  
*I. Kisialiou, 2021*  

![Output image example](/Output/yolo4_car4.jpg)  
## How to run
To run the detector, download the project and enter the following line into terminal:    
`python tire_detector.py -f <filename> -i <inputtype> -m <model>`  
> `<filename>` - name of the file in the `Input` folder  
>`<inputtype>` - either image or video  
>`<model>` - either `yolo4` or `yolo4tiny`  
> e.g. `python tire_detector.py -f car3.jpg -i image -m yolo4`  
> *Make sure that your environment corresponds to the requirements.*  

It is also possible to run the detector in Colaboratory with the notebook **Notebooks/TireDetector_(colab).ipynb**  

## Project structure  
- **Input**  
The folder for input files  
- **Notebooks**  
Colab-adapted .ipynb files for training, testing and run of the detector  
- **Output**  
Destination folder for the output of **tire_detector.py** script  
Find in this folder the examples of Detector's output and enjoy  
- **model**  
The folder with model weights and config files. Numbers (1000/2000/3000..) mean the number of learned batches (different stages of training).  
- **Root directory contains:**   
  - zip file with the dataset  
  - training process charts  
  - environment files  
  - main script **tire_detector.py**  

## Dataset
### Original dataset
The [Stanford Cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)  
The source of demo videos: https://www.pexels.com/search/videos/car/
### Annotation
Annotated 1500 images in [Roboflow online tool](roboflow.com)  

*Annotation rules:*  
- trade-off between "don't punish for the truth" and "don't force to see the invisible"  
- annotated all visible tires, except for the smallest  
- annotated truncated tires  
- annotated occluded tires  
- annotated front-view and far-side tires  
### Augmentation  
Only brightness variations between 0 and 40%, 2 x train data   
### Train - Valid - Test split after augmentation of train data  
2000 - 279 - 145 images correspondingly 

## Model
The developed detector can use two models: YOLOv4 and YOLOv4 Tiny ([Darknet, roboflow fork](https://github.com/roboflow-ai/darknet)) 

**YOLO v4:**
  - CSPDarknet53 backbone (137 pretrained conv layers)  
  - 3 YOLO layers  
  - Mish activation  
  - 416 x 416 input  
 
**YOLO v4 tiny:**   
  - CSPDarknet53 backbone (29 pretrained conv layers)  
  - 2 YOLO layers  
  - Leaky ReLU activation  
  - 416 x 416 input  
 > [Original paper](https://arxiv.org/pdf/2004.10934.pdf)
 > 

## Training
The training was performed on Google Colaboratory on GPU.  
Batch size: 64, subdivisions: 32  
Number of classes: 1  

  - YOLO-v4-tiny trained for 4000 batches, during 3 hours  
  - The training of YOLO-v4 took 10 h for 3800 batches.  
    However, only the weights on stage 3000 were saved, because of Colab kernel drop by GPU time limit.  

**Trainig charts:**  
YOLO v4 Tiny | YOLO v4
------------ | -------------
![YOLO v4 Tiny](/chart_yolov4tiny.png) | ![YOLO v4](/chart_yolov4.png)
  
## Testing
The results on the test set:  
Num. batches | YOLO v4 tiny | YOLO v4
------------ | ------------- | -------------
1000 | 86.26 | 92.11
2000 | 86.23 | 94.09
3000 | 88.85 | 93.15
4000 | 88.16 | None
"best" | 89.21 | None

## Inference speed  
Windows-10-10.0.19041-SP0  
AMD64 Family 23 Model 24 Stepping 1, AuthenticAMD  
(CPU)  
> **Yolo v.4:** 1000 - 1200 ms  
> **Yolo v.4 tiny:** 100 - 150 ms  
