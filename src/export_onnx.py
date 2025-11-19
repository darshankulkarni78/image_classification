import torch
import onnx
from onnxconverter_common import float16
from torchvision import models
import os

def export_onnx():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, 2)
    model.load_state_dict(torch.load("models/mobilenet_car_truck.pth"))
    model = model.cuda()
    model.eval()

    dummy = torch.randn(1, 3, 224, 224).cuda()
    os.makedirs("models", exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        "models/mobilenet_car_truck.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )

    fp32 = onnx.load("models/mobilenet_car_truck.onnx")
    fp16 = float16.convert_float_to_float16(fp32)
    onnx.save(fp16, "models/mobilenet_car_truck_fp16.onnx")

if __name__ == "__main__":
    export_onnx()