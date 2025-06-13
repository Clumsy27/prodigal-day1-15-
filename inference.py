import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load the ONNX model
session = ort.InferenceSession("model/resnet18.onnx", providers=['CPUExecutionProvider'])

# Load a real image (make sure this file exists)
image_path = "images/Dog-Images.jpg"
image = Image.open(image_path).convert("RGB")

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize(256),          
    transforms.CenterCrop(224),      
    transforms.ToTensor(),           
    transforms.Normalize(            
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]
    )
])
input_tensor = transform(image).unsqueeze(0).numpy()  # Add batch dimension (1, C, H, W)

#  Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
outputs = session.run([output_name], {input_name: input_tensor})

# Get prediction
predicted_class = np.argmax(outputs[0])
print(f"🎯 Predicted class index: {predicted_class}")
