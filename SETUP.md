# Hướng dẫn cài đặt trên máy khác

## 1. Clone repository từ GitHub

```bash
git clone <your-github-url>
cd detect
```

## 2. Tạo virtual environment (khuyên dùng)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

## 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý:** 
- `torch` + `torchvision` sẽ tải ~2GB
- `insightface` sẽ tự động tải model ArcFace (~400MB) lần đầu chạy
- Khi đầu tiên, hãy chắc chắn có **kết nối Internet ổn định**

## 4. Tải weights đã train

Tải 2 file từ Google Drive:
- `best.pt` - YOLO26 weights (phát hiện khuôn mặt)
- `face_db.pt` - ArcFace database (5 mặt đã train)

**Link:** [Folder Weights](your-google-drive-link-here)

**Cách tải:**
1. Mở link Google Drive
2. Download 2 file `best.pt` + `face_db.pt`
3. Đặt vào thư mục `ultralytics/`:
   ```
   detect/
   └── ultralytics/
       ├── best.pt          ← paste ở đây
       ├── face_db.pt       ← paste ở đây
       ├── main.py
       └── recognizer.py
   ```

Nếu không có files này:
- Hệ thống vẫn chạy được, chỉ nhận diện được "Người lạ"
- Enroll người qua `/capture` endpoint trên web

## 5. Chạy server

Từ thư mục gốc (`detect/`):

```bash
cd ultralytics
uvicorn main:app --host 0.0.0.0 --port 8000
```

Server sẽ chạy tại: `http://localhost:8000`

## 6. Sử dụng

- **Web UI**: Mở `http://localhost:8000` trên trình duyệt
- **Export Excel**: Nhấn nút export trên web, file Excel sẽ tải về
- **API endpoints**:
  - `POST /process-frame` - xử lý frame webcam
  - `POST /log-event` - ghi nhật ký vào/ra
  - `GET /export` - xuất Excel 3 sheet (Ngày/Tuần/Tháng)
  - `GET /health` - kiểm tra server

## Troubleshooting

**Lỗi: ModuleNotFoundError: No module named 'insightface'**
- Chạy: `pip install --upgrade insightface`

**Lỗi: Cannot find face_db.pt**
- Đây là database nhân viên. Nếu không có, hệ thống sẽ chỉ nhận diện "Người lạ"
- Enroll người qua `/capture` endpoint

**Server chậm lần đầu**
- Lần đầu chạy, insightface sẽ download model (~400MB) từ Internet
- Bình thường sau đó
