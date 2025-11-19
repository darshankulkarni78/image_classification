# Car vs Truck Classifier — MobileNetV2 + ONNX + FastAPI

## Overview
A compact computer-vision pipeline for classifying cars vs trucks using:
- OpenCV preprocessing  
- MobileNetV2 (trained from scratch)  
- ONNX FP32 → FP16 conversion  
- ONNX Runtime inference  
- FastAPI deployment  

## Folder Structure
```bash
project/
├── data_final/ # train/test split (ignored)
├── data_processed/ # preprocessed images (ignored)
├── models/
│ ├── mobilenet_car_truck.pth
│ ├── mobilenet_car_truck.onnx
│ └── mobilenet_car_truck_fp16.onnx
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── export_onnx.py
│ └── infer_onnx.py
├── app/
│ └── app.py
└── requirements.txt
```

## Dataset
Subset of STL-10 reorganized into:
- `car/`
- `truck/`

Split into:
- `data_final/train/`
- `data_final/test/`

## Preprocessing
(Defined in `src/preprocess.py`)
- RGB → LAB  
- CLAHE (L-channel)  
- Bilateral filter  
- Resize to 224×224  

## Training
(Defined in `src/train.py`)
- MobileNetV2 (no pretrained weights)  
- Final layer → `Linear(1280, 2)`  
- Adam optimizer  
- 20 epochs  
- Saves PyTorch model to `models/mobilenet_car_truck.pth`

## ONNX Export + FP16
(Defined in `src/export_onnx.py`)
- Export PyTorch → ONNX  
- Convert ONNX FP32 → FP16  
- Writes both ONNX models into `models/`

## Inference (ONNX Runtime)
(Defined in `src/infer_onnx.py`)
- Loads FP16 ONNX model  
- Preprocesses image  
- Returns:
  - `0 = car`
  - `1 = truck`

## FastAPI Deployment
(Defined in `app/app.py`)
Start server:
```bash
uvicorn app.app:app --reload
```
Run prediction:
```bash
curl -X POST -F "file=@image.jpg" http://127.0.0.1:8000/predict
```
