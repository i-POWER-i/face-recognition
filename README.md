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

```bash
pip install opencv-python numpy
