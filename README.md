# Hệ thống chấm công bằng nhận diện khuôn mặt

Ứng dụng điểm danh ra/vào bằng camera. Phát hiện khuôn mặt bằng YOLO26, nhận diện
danh tính bằng ArcFace, ghi nhật ký ra/vào theo từng camera và xuất báo cáo Excel
theo Ngày / Tuần / Tháng. Có thêm lớp xuất tín hiệu ra đèn tháp qua PLC Mitsubishi
(MC Protocol) để báo hiệu tại chỗ.

---

## 1. Tổng quan

### Luồng xử lý

1. Trình duyệt lấy hình từ webcam, gửi lên server hai luồng song song:
   - `POST /track` (~10 fps): chỉ chạy YOLO để lấy vị trí khung mặt, giúp khung bám
     sát chuyển động.
   - `POST /process-frame` (thưa hơn): chạy đầy đủ YOLO + ArcFace để xác định tên.
2. YOLO26 (`best.pt`) phát hiện và khoanh mọi khuôn mặt trong khung hình.
3. ArcFace (InsightFace `buffalo_l`) cắt lại vùng mặt từ ảnh gốc, căn theo 5 điểm
   mốc rồi tạo vector 512 chiều.
4. So cosine vector đó với các mẫu trong `face_db.pt` để ra người giống nhất. Nhận
   tên khi vừa đạt ngưỡng cosine, vừa cách người đứng nhì đủ xa; nếu không thì coi
   là "chưa chắc".
5. Bộ làm mượt theo track: mỗi khuôn mặt được bám qua nhiều khung liên tiếp, một
   danh tính phải chiếm đa số phiếu trong cửa sổ gần nhất mới được gán, tránh nhấp
   nháy Người quen / Người lạ.
6. Ghi nhật ký: đứng im đủ 2 giây ghi một lần; di chuyển ghi mỗi lần đổi trạng
   thái. Check-in/check-out gán theo camera nhận ra người.
7. Xuất Excel và (tuỳ chọn) bật đèn tháp qua PLC.

### Hai camera

Hệ thống hỗ trợ hai camera chạy độc lập, mỗi camera có state tracking riêng:

| Camera | Vai trò | Ghi chú |
|--------|---------|---------|
| `cam1` | Cổng vào (check-in) | Ngưỡng chuẩn |
| `cam2` | Cổng ra (check-out) | Siết ngưỡng chặt hơn và bù sáng/độ nét (CLAHE + unsharp) cho camera thiếu autofocus/HDR |

Tên hiển thị và cách gán vai trò cấu hình trong `ultralytics/main.py` (`CAMERAS`,
`CAM_ROLE`, `CAM_TUNE`, `CAM_ENHANCE`).

---

## 2. Yêu cầu

- Python 3.8 trở lên (khuyến nghị 3.10 – 3.12).
- Windows / macOS / Linux.
- Khoảng 3 GB dung lượng trống cho thư viện và model.
- Kết nối Internet ổn định trong lần chạy đầu (tải PyTorch và model ArcFace).
- GPU NVIDIA có CUDA là tuỳ chọn. Không có GPU thì hệ thống tự chuyển sang chế độ
  nhẹ (giảm kích thước ảnh đưa vào YOLO) và vẫn chạy được trên CPU.
- Webcam USB. Máy có 1 webcam vẫn dùng được, chỉ là hai ô camera trên web trỏ vào
  cùng thiết bị.

---

## 3. Cài đặt

### 3.1. Tải mã nguồn

```bash
git clone https://github.com/boanuen/FaceRecog
cd detect
```

### 3.2. Tạo môi trường ảo

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3.3. Cài thư viện

```bash
pip install -r requirements.txt
```

Lưu ý:

- `torch` + `torchvision` tải khoảng 2 GB.
- `insightface` tự tải model ArcFace (khoảng 400 MB) trong lần chạy đầu tiên.
- File `requirements.txt` ghim `torch==2.0.0` bản CPU. Nếu muốn dùng GPU, cài lại
  `torch`/`torchvision` theo hướng dẫn tại https://pytorch.org phù hợp phiên bản
  CUDA của máy.

### 3.4. Tải weights đã train

Tải hai file từ Google Drive: https://bit.ly/4zgLmth

- `best.pt` — weights YOLO26 phát hiện khuôn mặt.
- `face_db.pt` — cơ sở dữ liệu embedding của những người đã đăng ký.

Đặt cả hai vào thư mục `ultralytics/`:

```
detect/
└── ultralytics/
    ├── best.pt         <- đặt vào đây
    ├── face_db.pt      <- đặt vào đây
    ├── main.py
    └── recognizer.py
```

Nếu chưa có `face_db.pt`, hệ thống vẫn chạy nhưng chỉ báo "Người lạ". Có thể đăng
ký người trực tiếp trên web (mục 5.3) mà không cần train lại.

---

## 4. Chạy server

```bash
cd ultralytics
uvicorn main:app --host 0.0.0.0 --port 8000
```

Lần đầu khởi động mất vài chục giây để nạp model và làm nóng. Khi log hiện
`Warm-up done` là sẵn sàng.

Mở trình duyệt: http://localhost:8000

Truy cập từ máy khác trong cùng mạng LAN: `http://<IP-máy-chủ>:8000` (webcam của
trình duyệt cần chạy trên `localhost` hoặc HTTPS, nên máy khác cần cấu hình thêm
hoặc dùng trực tiếp trên máy chủ).

---

## 5. Sử dụng web

### 5.1. Bật camera

- Chọn thiết bị cho Camera 1 và Camera 2 ở ô dropdown dưới mỗi khung hình. Nếu tên
  thiết bị chứa "ugreen" hoặc "rapoo/c260", hệ thống tự chọn giúp.
- Bấm "Bật cả 2 camera" hoặc chờ trang tự bật khi tải xong.

### 5.2. Ngưỡng nhận diện

- **Người quen ≥**: ngưỡng cosine. Đặt cao thì phải giống chắc mới nhận tên (ít
  nhầm hơn, dễ báo "lạ" hơn). Camera 2 tự cộng thêm 0.07.
- **Bắt mặt ≥**: ngưỡng confidence của YOLO. Đặt thấp thì bắt được cả mặt nhỏ/mờ.

### 5.3. Đăng ký người

- Nhập tên vào ô "Thêm / Quản lý người" (có thể là tên mới), chọn vai trò, đưa mặt
  vào camera rồi bấm "Chụp mẫu". Nên chụp vài mẫu ở các góc và điều kiện sáng khác
  nhau.
- Danh sách người bên dưới cho phép đổi vai trò, xoá người, hoặc xem từng mẫu để
  tìm mẫu đăng ký nhầm (mẫu "tự-khớp" thấp mà "giống người khác" cao).

### 5.4. Nhật ký và Excel

- Khu vực "Nhật ký" hiện các sự kiện OK / FAIL / LẠ theo thời gian thực.
- "Xuất Excel" có hai kiểu:
  - **Đầy đủ**: gồm cả sự kiện di chuyển, fail và người lạ.
  - **Chỉ nhận dạng 1 lần/người**: mỗi người chỉ lấy lần OK đầu tiên.
- File Excel có 3 sheet: Ngày (chi tiết từng lượt), Tuần và Tháng (số ngày đi làm
  của từng người).

### 5.5. Xem kết quả train

Nút "Xem kết quả train" (hoặc `GET /results`) hiển thị confusion matrix, đường cong
loss/mAP và ảnh dự đoán của các lần train nằm trong `ultralytics/runs/`.

---

## 6. Đèn tháp PLC (tuỳ chọn)

Khi một sự kiện được ghi, `main.py` gửi tín hiệu bật đèn:

| Trạng thái | Màu đèn | Ý nghĩa |
|-----------|---------|---------|
| OK | Xanh | Nhận diện đúng (check-in hoặc check-out) |
| FAIL | Vàng | Người quen bị nhận nhầm thành người lạ |
| LẠ | Đỏ | Người lạ |

### 6.1. Demo bằng simulator (không cần phần cứng)

`plc_simulator.py` giả lập một PLC Mitsubishi, in ra log mỗi lệnh ghi bit.

```bash
# Terminal 1
cd ultralytics
python plc_simulator.py --port 5007

# Terminal 2
cd ultralytics
uvicorn main:app --host 0.0.0.0 --port 8000
```

Mặc định `main.py` đã trỏ `PLC_IP = 127.0.0.1`, `PLC_PORT = 5007` vào simulator.
Trên web sẽ thấy đèn tháp mô phỏng đổi màu; terminal 1 in ra dòng
`GHI Y0 = 1 (BẬT ĐÈN)` tương ứng.

Có thể chạy thử toàn bộ luồng camera → nhận diện → PLC mà không cần mở web:

```bash
python demo_camera_to_plc.py --cam 0 --role vao --plc-port 5007
```

### 6.2. Đấu PLC thật

Sửa phần cấu hình trong `ultralytics/main.py`:

```python
PLC_IP      = "192.168.1.10"   # IP module Ethernet của PLC
PLC_PORT    = 5000             # port MC Protocol cấu hình trên module
PLC_TYPE    = "Q"              # "Q" hoặc "iQ-R"
PLC_COILS   = {"green": "Y0", "yellow": "Y1", "red": "Y2"}   # địa chỉ ngõ ra thật
PLC_PULSE_S = 2.5             # thời gian đèn sáng mỗi lần báo (giây)
```

Trước khi tích hợp, kiểm tra riêng phần đấu dây:

```bash
python test_plc_light.py                  # bật lần lượt xanh → vàng → đỏ
python test_plc_light.py green            # chỉ bật xanh
```

Có thể bấm thử từ trình duyệt: `GET /plc-test?color=green|yellow|red`.

Chi tiết cấu hình GX Works và MC Protocol: xem
[giao-tiep-plc-mitsubishi.md](giao-tiep-plc-mitsubishi.md).

Nếu không kết nối được PLC, phần nhận diện và chấm công vẫn chạy bình thường; đèn
mô phỏng trên web vẫn phản ánh đúng quyết định của phần mềm.

---

## 7. Cấu trúc thư mục

```
detect/
├── README.md
├── SETUP.md                       # hướng dẫn cài đặt ngắn gọn
├── giao-tiep-plc-mitsubishi.md    # tài liệu MC Protocol / GX Works
├── requirements.txt
└── ultralytics/
    ├── main.py                    # FastAPI server, tracking, ghi log, xuất Excel
    ├── recognizer.py              # bọc YOLO + ArcFace, quản lý face_db.pt
    ├── plc_light.py               # gửi lệnh bật đèn xuống PLC qua MC Protocol
    ├── plc_simulator.py           # PLC giả lập để demo/test
    ├── test_plc_light.py          # test riêng phần đèn/đấu dây
    ├── demo_camera_to_plc.py      # demo toàn luồng camera → PLC, không cần web
    ├── index.html                 # giao diện web
    ├── enroll.py                  # tạo face_db.pt từ ảnh trong train/valid
    ├── train.py                   # train YOLO26 detect (cần GPU)
    ├── train_filtered.py          # train thử trên dataset đã lọc
    ├── add_captures.py            # gộp ảnh webcam đã chụp vào tập train
    ├── eval_recognition.py        # đo độ chính xác nhận diện trên test set
    ├── best.pt                    # weights YOLO26 (tải riêng)
    ├── face_db.pt                 # embedding người đã đăng ký (tải riêng)
    ├── data.yaml                  # cấu hình dataset
    ├── train/  valid/  test/      # ảnh và nhãn dataset
    ├── captures/                  # ảnh chụp từ web (dùng để train bổ sung)
    └── runs/                      # kết quả các lần train
```

---

## 8. Endpoint API

| Method | Đường dẫn | Mô tả |
|--------|-----------|-------|
| GET | `/` | Giao diện web |
| GET | `/health` | Trạng thái server, thiết bị, số người trong DB |
| POST | `/process-frame` | Nhận diện đầy đủ một frame (YOLO + ArcFace), trả toạ độ khung và tên |
| POST | `/track` | Chỉ chạy YOLO để bám khung, tên lấy từ cache |
| POST | `/capture` | Thêm một mẫu ảnh cho một người vào `face_db.pt` |
| GET | `/people` | Danh sách người và số mẫu |
| POST | `/update-role` | Đổi vai trò của một người |
| POST | `/remove-person` | Xoá một người khỏi DB |
| GET | `/people/samples` | Chẩn đoán từng mẫu của một người |
| POST | `/remove-sample` | Xoá một mẫu cụ thể |
| GET | `/export?mode=full\|once` | Tải file Excel nhật ký |
| GET | `/results` | Trang xem biểu đồ kết quả train |
| GET | `/plc-status` | Màu đèn tháp hiện tại và trạng thái kết nối PLC |
| GET | `/plc-test?color=green\|yellow\|red` | Bấm thử một màu đèn |

---

## 9. Tham số cấu hình chính (`ultralytics/main.py`)

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `CONF_DETECT` | `0.25` | Ngưỡng confidence của YOLO khi bắt mặt |
| `THRESHOLD` | `0.28` | Ngưỡng cosine để coi là người quen |
| `SMOOTH_WINDOW` | `5` | Số frame gần nhất dùng để làm mượt điểm số |
| `CAM_TUNE` | cam2 +0.07 / +0.05 | Cộng thêm ngưỡng và khoảng cách top1–top2 cho từng camera |
| `BLUR_THR` | cam1 60 / cam2 35 | Ngưỡng phương sai Laplacian để loại khung mờ |
| `STABLE_SECONDS` | `2.0` | Thời gian đứng im tối thiểu để ghi log một lần |
| `STRANGER_MIN` | `4` | Số frame "lạ" liên tiếp tối thiểu để ghi người lạ |
| `PLC_IP` / `PLC_PORT` | `127.0.0.1` / `5007` | Địa chỉ PLC (đang trỏ vào simulator) |

Sau khi sửa các tham số này cần khởi động lại server.

---

## 10. Train lại model

Các bước dưới đây cần dataset ảnh và nhãn theo định dạng YOLO trong `train/`,
`valid/`, `test/`.

```bash
cd ultralytics

# Train detector YOLO26 (cần GPU NVIDIA)
python train.py

# Tạo lại face_db.pt từ ảnh trong train/ và valid/
python enroll.py

# Đo độ chính xác nhận diện trên test set
python eval_recognition.py

# Gộp ảnh đã chụp từ web (captures/) vào tập train rồi train lại
python add_captures.py
```

`enroll.py` sau khi tạo DB sẽ in ra ngưỡng cosine gợi ý để đặt lại cho `THRESHOLD`
trong `main.py`.

---

## 11. Xử lý sự cố

**`ModuleNotFoundError: No module named 'insightface'`**
Chạy lại `pip install -r requirements.txt`, hoặc `pip install --upgrade insightface`.

**Không tìm thấy `best.pt` / `face_db.pt`**
Kiểm tra hai file đã nằm trong `ultralytics/`. Thiếu `face_db.pt` thì hệ thống chỉ
báo "Người lạ" cho tới khi đăng ký người.

**Server rất chậm ở lần chạy đầu**
Lần đầu `insightface` tải model từ Internet. Các lần sau sẽ nhanh.

**Nhận diện chậm, khung hình giật**
Máy không có GPU. Hệ thống đã tự vào chế độ nhẹ; có thể giảm thêm số camera đang
bật hoặc hạ độ phân giải webcam.

**Camera 2 hay nhận nhầm**
Camera chất lượng thấp (không autofocus, không HDR) cho ảnh mềm. Tăng `thr_bump` và
`margin` cho `cam2` trong `CAM_TUNE`, và nên đăng ký thêm mẫu chụp bằng chính
camera đó.

**Đèn PLC không sáng dù log báo đã ghi bit**
Kiểm tra `PLC_COILS` đúng địa chỉ ngõ ra, đấu dây và nguồn cấp cho đèn. Dùng
`test_plc_light.py` và GX Works ở chế độ Online Monitor để khoanh vùng lỗi.

---

## 12. Tài liệu liên quan

- [SETUP.md](SETUP.md) — hướng dẫn cài đặt ngắn.
- [giao-tiep-plc-mitsubishi.md](giao-tiep-plc-mitsubishi.md) — giao tiếp PLC
  Mitsubishi qua MC Protocol.
