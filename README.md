# Vehicle tire detector based on YOLO v.4  
*I. Kisialiou, 2021*  
## How to run
To run the detector, download the project and enter the following line into terminal:    
`python tire_detector.py -f <filename> -i <inputtype> -m <model>`  
> `<filename>` - name of the file in the `Input` folder  
>`<inputtype>` - either image or video  
>`<model>` - either `yolo4` or `yolo4tiny`  
> e.g. `python tire_detector.py -f car3.jpg -i image -m yolo4`  
> *Make sure that your environment corresponds to the requirements.*  

It is also possible to run the detector in Colaboratory with the notebook  
`Notebooks/TireDetector_(colab).ipynb`  
