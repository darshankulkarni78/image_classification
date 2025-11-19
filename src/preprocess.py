import cv2
import numpy as np
import os

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.resize(img, (224, 224))
    return img

def build_processed_dataset(subset_images, subset_labels, out_root="data_processed"):
    os.makedirs(f"{out_root}/car", exist_ok=True)
    os.makedirs(f"{out_root}/truck", exist_ok=True)

    idx = 0
    for img, label in zip(subset_images, subset_labels):
        out = preprocess(img)
        cls = "car" if label == 3 else "truck"
        cv2.imwrite(f"{out_root}/{cls}/{idx}.jpg", out)
        idx += 1