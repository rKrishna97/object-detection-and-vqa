# Object Detection and VQA


Installation
------------


### Requirements
Python 3 needs to be installed. Install Python 3 with [Anaconda](https://www.anaconda.com/products/individual#Downloads "Download Anaconda")

Once Python 3 is installed, Clone this repository and then create a conda environment


```
conda create -n blind-vision python=3.8
conda activate blind-vision
conda install -c anaconda pyaudio
pip install -r requirements.txt
```
You also need to install ffmpeg.

If you are on Linux then run below command in the terminal
```
sudo apt update
sudo apt install ffmpeg
```
If you are on Windows, Then download [ffmpeg](https://www.ffmpeg.org/download.html "Download ffmpeg") package for windows and then extract it.
Once extracted, Add the path of bin folder to Environment Variable in Windows.

Now the installation if done.

### How to run the project
Go to the project directory and make sure conda environment is active. If not then you can activate by running below command
```
conda activate blind-vision
```
First you need to train the model and place the model in the project root directory. You can train model from `train_vqa.ipynb` in `train_vqa` folder.\
or\
You can download the model for vqa from [here](https://drive.google.com/file/d/1qOeBMiT-ikegcwr5PJ7k879CL2AITB3O/view?usp=sharing "Download model"). Once Downloaded place the file in project root directory

For object detection download Yolov4 models from [here](https://drive.google.com/file/d/1-bN-hO-ZIV_ZPaptwMfP3d8-oAIaop2H/view?usp=sharing). Extract and place all the files in yolo_model folder.

Now start server by running
```
python app.py
```
Now the server will start. You can ctrl+click on the link in the terminal you can paste `http://localhost:5000/` in your browser to launch the site.

