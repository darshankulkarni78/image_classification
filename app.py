from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import onnxruntime as ort

app = FastAPI()
sess = ort.InferenceSession("mobilenet_car_truck_fp16.onnx")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))

    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2,0,1))
    x = np.expand_dims(x, 0)
    x = x.astype(np.float16)

    out = sess.run(None, {"input": x})[0]
    pred = int(np.argmax(out))
    return {"prediction": pred}