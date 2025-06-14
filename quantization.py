import torch
import torchvision.models as models
import time

model_fp32 = models.resnet18(pretrained=True)
model_fp32.eval()
input_tensor = torch.randn(1, 3, 224, 224)
start = time.time()
for _ in range(100):
    with torch.no_grad():
        model_fp32(input_tensor)
end = time.time()
print(f"FP32 model inference time: {(end - start):.4f} seconds")
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, 
    {torch.nn.Linear},
    dtype=torch.qint8
)
start = time.time()
for _ in range(100):
    with torch.no_grad():
        model_int8(input_tensor)
end = time.time()
print(f"INT8 quantized model inference time: {(end - start):.4f} seconds")
torch.save(model_fp32.state_dict(), "resnet18_fp32.pth")
torch.save(model_int8.state_dict(), "resnet18_int8.pth")

import os
print("FP32 size:", os.path.getsize("resnet18_fp32.pth") / 1024, "KB")
print("INT8 size:", os.path.getsize("resnet18_int8.pth") / 1024, "KB")

