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
### Annotation
Annotated 1500 images in [Roboflow online tool](roboflow.com)  

*Annotation rules:*  
- trade-off between "don't punish for the truth" and "don't force to see the invisible"  
- annotated all visible tires, except for the smallest  
- annotated truncated tires  
- annotated occluded tires  
- annotated fron-view and far-side tires  
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
  [Original paper](https://arxiv.org/pdf/2004.10934.pdf)
