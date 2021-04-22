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
