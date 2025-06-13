import torch
import torchvision.models as models


model = models.resnet18(weights="IMAGENET1K_V1")
model.eval()


dummy_input = torch.randn(1, 3, 224, 224)


torch.onnx.export(model, dummy_input, "model/resnet18.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

print("✅ Model converted and saved as resnet18.onnx")
