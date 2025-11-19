import onnxruntime as ort
import numpy as np
import cv2

sess = ort.InferenceSession("models/mobilenet_car_truck_fp16.onnx")

def infer(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))

    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2,0,1))
    x = np.expand_dims(x, 0)
    x = x.astype(np.float16)

    out = sess.run(None, {"input": x})[0]
    pred = int(np.argmax(out))
    return pred

if __name__ == "__main__":
    print(infer("data_final/test/car/144.jpg"))