# Bước 1: Dùng Base Image chính thức của NVIDIA đã tích hợp sẵn CUDA 12.1 và Python 3.11
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Bước 2: Cài đặt Python 3.11 và các thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Bước 3: Tạo thư mục làm việc bên trong Docker
WORKDIR /app

# Bước 4: Copy file requirements và cài đặt PyTorch bản GPU CUDA 12.1 chuẩn
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Bước 5: Copy toàn bộ code web, backend và file trọng số 'best.pt' vào Docker
COPY . .

# Bước 6: Mở cổng 8000 để máy tính truy cập giao diện web
EXPOSE 8000

# Bước 7: Health check — Docker tự kiểm tra server còn sống không
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Bước 8: Lệnh kích hoạt server Uvicorn chạy FastAPI Web
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]