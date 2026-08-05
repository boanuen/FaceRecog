# Face Recognition Attendance System
YOLO26 + ArcFace face recognition → Excel export (Ngày/Tuần/Tháng)

## Features
- 🎥 Real-time face detection (YOLO26) + recognition (ArcFace)
- 📊 Automatic attendance log with in/out timestamps
- 📁 Export to Excel with 3 sheets:
  - **Ngày** (Daily): All sessions with duration
  - **Tuần** (Weekly): Days attended per person per week
  - **Tháng** (Monthly): Days attended per person per month
- 🌐 FastAPI web interface
- 👥 Support 2 roles: Kỹ sư (Engineer) + Sinh viên (Student)

## Quick Start

### 1. Clone & Install
```bash
git clone <repo-url>
cd detect
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Download Weights
- Download `best.pt` + `face_db.pt` from [Google Drive Link]
- Place in `ultralytics/` folder

### 3. Run
```bash
cd ultralytics
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open: http://localhost:8000

## Project Structure
```
detect/
├── requirements.txt          # Dependencies
├── SETUP.md                 # Detailed setup guide
├── README.md                # This file
└── ultralytics/
    ├── main.py              # FastAPI server + Excel export
    ├── recognizer.py        # YOLO + ArcFace wrapper
    ├── best.pt              # YOLO26 weights (download)
    ├── face_db.pt           # Face embeddings (download)
    └── index.html           # Web UI
```

## API Endpoints
- `POST /process-frame` - Detect faces in frame
- `POST /log-event` - Log attendance event
- `GET /export` - Export Excel file
- `GET /health` - Health check

## Configuration
Edit `ultralytics/main.py`:
```python
CONF_DETECT = 0.25      # YOLO detection threshold
THRESHOLD = 0.28        # Face recognition threshold
WORK_START_HOUR = 8     # Work start time (for late detection)
WORK_START_MIN = 0
```

## Troubleshooting
See [SETUP.md](SETUP.md) for detailed troubleshooting guide.

## License
Private project
