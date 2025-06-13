#  Hello Inference!

This project demonstrates how to convert a PyTorch model (ResNet18) to ONNX format and run inference inside a Docker container using GPU acceleration (CUDA).

## Project Structure

```
hello-inference/
├── Dockerfile
├── conversion.py          
├── inference.py           # Loads ONNX model and runs inference on input image
├── model/
│   └── resnet18.onnx      # Exported ONNX model
├── images/
│   ├── Dog-Images.jpg     # Sample input image
│   └── cat.jpg
├── requirements.txt       # Python dependencies
└── README.md              # You're here!
```

---

##  Setup (Locally)

1. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # Windows
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run ONNX Conversion:**

   ```bash
   python conversion.py
   ```

4. **Run Inference:**

   ```bash
   python inference.py
   ```

---

##  Run with Docker

### 1. Build Docker Image

```bash
docker build -t hello-inference .
```

### 2. Run Docker Container

```bash
docker run --rm hello-inference
```

> Make sure the `images/Dog-Images.jpg` exists and is accessible inside the container.

---

##  Sample Output

```
Predicted class: golden_retriever
```

---

##  Technologies Used

* Python
* PyTorch
* ONNX
* Docker
* torchvision / PIL

---

##  Purpose

To learn:

* How to convert models to ONNX
* How to use Docker with AI workloads
* How to run GPU-powered inference
