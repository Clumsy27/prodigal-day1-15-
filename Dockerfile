FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY conversion.py .
COPY inference.py .
COPY model/ model/
COPY images/ images/

CMD ["python", "inference.py"]
