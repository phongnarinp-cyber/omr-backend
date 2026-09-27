FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# คำสั่งติดตั้ง lib สำคัญสำหรับ OpenCV และ Zbar
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libzbar0
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
