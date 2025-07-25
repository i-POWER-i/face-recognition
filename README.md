# Face Detection with Age & Smile Recognition

This project uses OpenCV and a pretrained Caffe model to:

- Detect faces in real-time using webcam
- Estimate age range of detected faces
- Detect smiles
- Save only unique faces (no duplicates) to disk

## 📂 Files
- sen_unique_faces.py: Main Python script
- age_deploy.prototxt and age_net.caffemodel: Required age estimation model files (not included in this repo)
- faces/: Folder where unique face images are saved

## 🛠 Requirements

Install dependencies:

pip install opencv-python numpy


Download these required model files and place them in the same folder:
 
 • age_deploy.prototxt ( https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_deploy.prototxt )
 
 • age_net.caffemodel ( https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel )


*🚀 Run*

in cmd install pencv

```bash
pip install opencv-python numpy
```
Press ENTER to quit the application

All unique detected faces will be saved in the faces/ folder with timestamped filenames.
