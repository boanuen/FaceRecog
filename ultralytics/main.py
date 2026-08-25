"""
Chạy local:  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from collections import deque, Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import asyncio
import cv2
import numpy as np
import base64
import os
import glob
import time
import io
import calendar
from datetime import datetime, timedelta
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from recognizer import FaceRecognizer, BASE_DIR, DB_PATH
from plc_light import PLCLight

executor = ThreadPoolExecutor(max_workers=3)

app = FastAPI(title="YOLO26 + ArcFace Face Recognition")

#   YOLO26 (best.pt)  → phát hiện + khoanh mặt   (DETECTOR)
#   ArcFace           → embedding 512-d mỗi mặt  (NHẬN DIỆN)
#   Cosine vs face_db.pt → ra tên, hoặc "người lạ" nếu không đủ gần


CONF_DETECT = 0.25   # ngưỡng YOLO BẮT mặt (để thấp để không bỏ sót)
THRESHOLD   = 0.28   # ngưỡng COSINE (matcher top-5 mean): >= coi là người quen.
                     # Đo trên test: 100% đúng ở 0.25-0.28, 0 nhầm danh tính. Xem eval_recognition.py.

# ── LÀM MƯỢT THEO THỜI GIAN
# Quyết định = làm mượt điểm số trên vài frame gần nhất của CÙNG một mặt (ghép theo vị trí).
# Nhỏ hơn → nhận ra NHANH hơn cho người đi qua; lớn hơn → ổn định hơn cho người đứng im.
SMOOTH_WINDOW = 5

# Enroll trực tiếp từ webcam: /capture thêm embedding của mặt vào DB (không cần train).
KNOWN_PEOPLE = ["nghia", "quan", "son", "tri", "tui"]

# ── NHẬT KÝ RA VÀO (lưu toàn bộ phiên, export ra Excel — dùng cho sheet Tuần/Tháng) ──
# Mỗi entry: {dt, type, person, role, duration_min, ks_present, sv_present, total_ks, total_sv}
# Check-in/check-out TÁCH THEO CAMERA (như cổng vào/cổng ra vật lý) chứ không suy từ việc
# "ai đang được thấy" gộp cả 2 cam — xem CAM_ROLE + _log_attendance().
_attendance_log: list[dict] = []
_in_room: dict[str, datetime] = {}   # name → thời điểm vào phòng

# ── NHẬT KÝ QUÉT MẶT (mỗi lần 1 track ĐỔI trạng thái — dùng cho sheet Ngày) ──
# Mỗi entry: {dt, person, role, status}  status ∈ {"OK","FAIL","LẠ"}
#   OK   = một track nhận ra người quen (đứng im hay đi qua → chỉ ghi 1 lần/track)
#   FAIL = track đang là người quen X bỗng thành "người lạ" (nhận dạng hỏng)
#   LẠ   = người lạ hoàn toàn xuất hiện
# motion ∈ {"Đứng im","Di chuyển"} — trạng thái chuyển động của mặt lúc xảy ra sự kiện.
_scan_log: list[dict] = []   # mỗi entry kèm "cam" (camera nào sinh ra sự kiện)

# ── TRẠNG THÁI TRACK THEO TỪNG CAMERA ──
# 2 camera = 2 luồng độc lập → state tracking RIÊNG cho mỗi cam (mặt cam1 không lẫn cam2).
CAMERAS = {"cam1": "UGreen 2K", "cam2": "Rapoo C260"}   # id → tên hiển thị

# ── SIẾT NHẬN DIỆN RIÊNG TỪNG CAMERA (chống NHẬN SAI) ──
# Rapoo C260 (cam2) cho ảnh MỀM/NHIỄU hơn UGreen 2K → cosine "lưng chừng" nhiều → dễ gán
# NHẦM tên (người A thành B) hoặc nhận bừa người lạ. Với cam đó, siết chặt hơn cam1:
#   thr_bump : CỘNG thêm vào ngưỡng cosine người-quen (chỉ nhận khi thật giống).
#   margin   : top-1 phải hơn top-2 ÍT NHẤT chừng này; sát nhau → coi frame CHƯA CHẮC
#              (để bộ làm mượt tự quyết), đây là đòn chính chống nhầm 2 người quen với nhau.
#
# HỆ SỐ cam2 suy từ chênh THÔNG SỐ so với UGreen 2K (không đoán mò):
#   EQI = (1/FOV)·focus·hdr    focus: AF=1.0 / fixed=0.85   hdr: có=1.0 / không=0.90
#     UGreen(nền) = (1/80)·1.00·1.00 = 0.01250   [4MP, 2K, AF, HDR]
#     Rapoo       = (1/95)·0.85·0.90 = 0.00805   [2MP, 1080p, fixed-focus, không HDR, góc 95°]
#   Q = 1.55 → Rapoo mất ~36% chất lượng hiệu dụng.
#   bump = deficit(0.36) × headroom(~0.20) ≈ 0.07 ;  margin ≈ 0.7·bump ≈ 0.05.
# cam1 để 0/0 → GIỮ NGUYÊN hành vi cũ. Đây là điểm KHỞI ĐẦU có căn cứ — vẫn nên soi Excel
# "người lạ/fail" rồi nhích nếu còn nhầm (tăng) hay bị miss người thật (giảm).
CAM_TUNE = {
    "cam1": {"thr_bump": 0.00, "margin": 0.00},
    "cam2": {"thr_bump": 0.07, "margin": 0.05},
}

# ── CAMERA NÀO = CỔNG VÀO, CAMERA NÀO = CỔNG RA ──
# UGreen (cam1) đặt ở lối VÀO, Rapoo (cam2) đặt ở lối RA. Check-in/out do đó gán THẲNG theo
# camera nhận ra người — không suy luận qua "người này có đang được nhìn thấy hay không" gộp
# cả 2 cam (cách cũ dễ lẫn: ai đó chỉ lướt qua cam RA mà chưa từng vào vẫn bị tính "vào").
CAM_ROLE = {"cam1": "vào", "cam2": "ra"}

# ── LỌC MẶT MỜ TRƯỚC KHI CHẠY ArcFace (chống NHẬN SAI do motion-blur) ──
# ArcFace vẫn "chạy được" trên ảnh mờ nhưng cho embedding kém tin cậy → dễ nhận NHẦM người quen
# A thành B (khác với "người lạ" — đây là lỗi nặng hơn). Cùng nguyên lý các kiosk nhận diện
# (như Vinwonders): chỉ MATCH trên khung ĐỦ NÉT, khung mờ thì bỏ qua phiếu (không chạy ArcFace,
# không tính vào cửa sổ làm mượt) — track vẫn bám vị trí, đợi khung nét hơn (vài trăm ms sau)
# mới nhận. Đo bằng phương sai Laplacian trên vùng mặt resize chuẩn 100x100 (độc lập khoảng
# cách gần/xa camera). Rapoo (cam2) vốn mềm hơn (fixed-focus) → ngưỡng thấp hơn cam1, nếu không
# sẽ bỏ qua gần hết khung của cam2.
BLUR_THR = {"cam1": 60.0, "cam2": 35.0}
SKIP_VOTE = "__SKIP__"   # sentinel best_name: khung bị lọc mờ, KHÔNG tính là phiếu "lạ"/"chưa chắc"

# ── BÙ ẢNH MỀM/THIẾU SÁNG (chỉ cam thiếu AF/HDR — KHÔNG đụng vùng nhìn) ──
# Rapoo không HDR (dễ cháy/tối chỗ sáng-tối lệch, nhất là lắp trên cao chiếu xuống → trán
# sáng, phần dưới mặt tối) + fixed-focus (ảnh mềm hơn AF). Bù trực tiếp lên ẢNH GỐC ĐẦY ĐỦ
# (không crop — giữ nguyên toàn bộ khung nhìn) TRƯỚC khi đưa vào YOLO **và** ArcFace, nên
# tác động thẳng vào pixel mà ArcFace thực sự tạo embedding — khác lần crop trước (crop chỉ
# đổi input YOLO, không hề chạm tới ArcFace vì ArcFace tự cắt lại từ ảnh gốc).
#   CLAHE (kênh L của LAB) → cân sáng CỤC BỘ, bù thiếu HDR.
#   Unsharp mask nhẹ       → bù mềm do fixed-focus.
# cam1 đã có AF+HDR phần cứng lo rồi → False, không xử lý thêm (tránh xử lý dư làm hại ảnh tốt).
CAM_ENHANCE = {"cam1": False, "cam2": True}
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _enhance(img, cam):
    if not CAM_ENHANCE.get(cam, False):
        return img
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = _clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.35, blur, -0.35, 0)   # unsharp mask nhẹ, tránh tạo nhiễu/viền giả

class CamState:
    def __init__(self):
        self.tracks = []              # track của riêng camera này
        self.stable_start = {}        # key → lúc bắt đầu đứng
        self.stable_seen  = {}        # key → lần thấy đứng gần nhất
        self.stable_logged = set()    # key đã ghi đứng im (1 lần/người/camera)

_cam_states: dict = defaultdict(CamState)

# ── ĐÈN THÁP (PLC Mitsubishi Q/iQ-R, MC Protocol qua Ethernet) ──────────────
# Đổi PLC_IP/PLC_PORT theo module Ethernet thật của PLC (QJ71E71 / cổng built-in
# iQ-R). Đổi PLC_TYPE thành "iQ-R" nếu dùng dòng iQ-R (mặc định "Q").
# PLC_COILS: địa chỉ ngõ ra Y thật đấu tới từng màu đèn tháp — chỉnh lại cho khớp
# tủ điện thực tế trước khi chạy thật.
PLC_IP      = "192.168.1.10"      # TODO: IP thật của PLC
PLC_PORT    = 5000                 # TODO: port MC Protocol cấu hình trên module Ethernet PLC
PLC_TYPE    = "Q"                  # "Q" hoặc "iQ-R"
PLC_COILS   = {"green": "Y0", "yellow": "Y1", "red": "Y2"}   # TODO: đúng địa chỉ Y thật
PLC_PULSE_S = 1.5                  # thời gian đèn sáng mỗi lần báo (giây)

plc = PLCLight(PLC_IP, PLC_PORT, PLC_COILS, plctype=PLC_TYPE, pulse_sec=PLC_PULSE_S)


def _log_scan(person: str, status: str, cam: str, motion: str = "Đứng im"):
    _scan_log.append({"dt": datetime.now(), "person": person,
                      "role": rec.db_roles.get(person, "kỹ sư"),
                      "status": status, "motion": motion,
                      "cam": CAMERAS.get(cam, cam)})
    if status == "OK":
        _log_attendance(person, cam)   # OK ở cam VÀO/RA → check-in/out (xem CAM_ROLE)
        # báo đèn: cam VÀO (cam1) → xanh ; cam RA/check-out (cam2) → vàng
        plc.signal("green" if CAM_ROLE.get(cam) == "vào" else "yellow")
    else:
        # FAIL (quen bỗng thành lạ) hoặc LẠ (người lạ) → đỏ
        plc.signal("red")

print(f"[main] Khởi tạo YOLO26 + ArcFace...")
rec = FaceRecognizer(det_conf=CONF_DETECT)   # YOLO detect mặt, ArcFace nhận diện tên

# ── CHẾ ĐỘ NHẸ: tự bật khi máy KHÔNG có GPU (YOLO chạy CPU) ──
# CPU thì YOLO/ArcFace chậm. YOLO chỉ TÌM VỊ TRÍ mặt (ArcFace tự cắt+căn lại từ ảnh gốc 1920),
# nên hạ imgsz YOLO 512→416 giúp NHANH mà KHÔNG giảm độ chính xác nhận dạng.
LIGHT_MODE = (rec.device == "cpu")
if LIGHT_MODE:
    rec.det_imgsz = 416
TRACK_IMGSZ = 256 if LIGHT_MODE else 320   # YOLO cho /track (bám khung, không ảnh hưởng nhận diện)

print(f"[main] Device YOLO : {rec.device}   {'(CHẾ ĐỘ NHẸ - CPU)' if LIGHT_MODE else ''}")
print(f"[main] DB          : {DB_PATH}  ({len(rec.db_names)} người: {rec.db_names})")
print(f"[main] Ngưỡng      : detect={CONF_DETECT}  cosine={THRESHOLD}")

# Warm-up để lần nhận diện đầu không bị chậm
rec.recognize(np.zeros((320, 320, 3), dtype=np.uint8), threshold=THRESHOLD)
print("[main] Warm-up done")


# ── BỘ THEO DÕI + LÀM MƯỢT ĐIỂM SỐ (chống nhảy "người lạ") ────────────────
# Mỗi mặt được ghép với 1 track theo vị trí. Track giữ lịch sử (tên, điểm) vài frame.
# Quyết định dựa trên ĐIỂM TRUNG BÌNH của danh tính nổi trội trong cửa sổ + HYSTERESIS.
# LƯU Ý: track state nằm TRONG CamState (mỗi camera 1 bộ riêng — xem _cam_states).

HYSTERESIS  = 0.04    # đã nhận tên → hạ ngưỡng 0.04 để giữ (khó rớt sang "lạ")
MIN_PRESENCE = 0.5    # danh tính phải xuất hiện >= 50% cửa sổ mới được nhận
MATCH_FRAC  = 0.18     # ngưỡng ghép track = 18% đường chéo khung (nới để bám khi di chuyển)
MOVE_SPEED_FRAC = 0.6  # TỐC ĐỘ tâm mặt > 60% bề rộng mặt / GIÂY → DI CHUYỂN (độc lập FPS)
STABLE_SECONDS = 2.0   # đứng im đủ 2 giây → ghi log 1 lần
STABLE_GAP  = 4.0      # gián đoạn thấy > 4s → coi là lượt đứng MỚI (phải > nhịp nhận diện
                       # trên máy chậm, nếu không đồng hồ đứng im bị reset hoài → không ghi)
STRANGER_MIN = 4       # phải "lạ" liên tục >= 4 frame mới ghi Người lạ (bớt nhầm)
TRACK_TTL   = 1.5      # track không thấy > 1.5 giây → xoá (dọn theo THỜI GIAN, độc lập FPS)
REFRESH_SEC = 3.0      # đã nhận ra người quen → DÙNG LẠI danh tính, chỉ chạy ArcFace lại sau 3s
                       # (cache danh tính → bỏ phần lớn ArcFace trên CPU khi người đã nhận ra)
REACQUIRE_GAP = 0.6    # ghép LẠI 1 track sau khi mất dấu > 0.6s (nhưng < TRACK_TTL nên chưa bị xoá)
                       # → COI NHƯ CÓ THỂ LÀ NGƯỜI KHÁC đứng đúng chỗ đó, xoá sạch danh tính cũ,
                       # buộc nhận diện lại từ đầu. Chống lỗi: người A rời đi, người B đứng gần
                       # chỗ A vừa đứng → track "thừa kế" nhầm tên A tới tận khi đủ phiếu mới đè lên.
                       # Người đi liên tục (ghép mỗi ~100ms qua /track, gap luôn < 0.6s) KHÔNG bị
                       # reset — chỉ trigger khi có GIÁN ĐOẠN thật (rời khung/bị che/đổi người).

def _reacquire_if_stale(t, now):
    """Gọi TRƯỚC khi cập nhật seen cho 1 track vừa được ghép lại — xem REACQUIRE_GAP."""
    if (now - t["seen"]).total_seconds() > REACQUIRE_GAP:
        t["hist"].clear(); t["label"] = "LẠ"; t["logged_label"] = ""
        t["ever_known"] = False; t["arc_time"] = None; t["arc_score"] = 0.5

def _is_moving(pos, wsize):
    """pos: deque (cx, cy, dt). TỐC ĐỘ theo GIÂY (độc lập FPS) + BỀN với nhiễu:
    so vị trí TRUNG BÌNH nửa đầu vs nửa sau cửa sổ (1 điểm nhiễu bị trung bình hoá,
    không làm đứng im thành di chuyển)."""
    if len(pos) < 3 or wsize <= 0:
        return False
    h = len(pos) // 2
    first, last = list(pos)[:h], list(pos)[-h:]
    fx = sum(p[0] for p in first) / h; fy = sum(p[1] for p in first) / h
    lx = sum(p[0] for p in last) / h;  ly = sum(p[1] for p in last) / h
    dt = (last[-1][2] - first[0][2]).total_seconds()
    if dt < 0.15:
        return False
    disp = ((lx - fx) ** 2 + (ly - fy) ** 2) ** 0.5
    return (disp / dt) > MOVE_SPEED_FRAC * wsize

def _match_thr():
    # ngưỡng ghép trong hệ toạ độ CHUẨN HOÁ [0,1] (đường chéo = √2)
    return MATCH_FRAC * (2 ** 0.5)

def _new_track(cx, cy, now):
    return {"cx": cx, "cy": cy, "hist": deque(maxlen=SMOOTH_WINDOW),
            "seen": now, "label": "LẠ", "logged_label": "",
            "pos": deque(maxlen=SMOOTH_WINDOW), "moving": False, "ever_known": False,
            "arc_time": None, "arc_score": 0.5}   # lần ArcFace gần nhất + điểm (cache)

def _nearest_track(tracks, cxn, cyn, thr):
    """Track gần nhất (CHỈ ĐỌC, không chiếm) — để quyết định có cần chạy ArcFace không."""
    best, bd = None, thr
    for t in tracks:
        d = ((t["cx"] - cxn) ** 2 + (t["cy"] - cyn) ** 2) ** 0.5
        if d < bd:
            bd, best = d, t
    return best

def _match_track_idx(tracks, cx, cy, thr, used):
    """Ghép (cx,cy) với track GẦN NHẤT chưa dùng trong `tracks`. Trả index hoặc None."""
    bt, bd = None, thr
    for ti, t in enumerate(tracks):
        if ti in used:
            continue
        d = ((t["cx"] - cx) ** 2 + (t["cy"] - cy) ** 2) ** 0.5
        if d < bd:
            bd, bt = d, ti
    return bt

def _prune_tracks(st, now):
    """Dọn track cũ của MỘT camera theo THỜI GIAN (độc lập FPS)."""
    st.tracks = [t for t in st.tracks if (now - t["seen"]).total_seconds() <= TRACK_TTL]

def smooth_labels(faces, W, H, threshold, st, cam):
    """faces → list (label, disp_score) đã làm mượt, dùng track state `st` của camera `cam`.
    Ghi nhật ký kèm tên camera."""
    thr = _match_thr()
    now = datetime.now()
    _prune_tracks(st, now)   # dọn track cũ TRƯỚC khi ghép → không hồi sinh track tồn đọng
    tracks = st.tracks
    used, out = set(), []
    for f in faces:
        x1, y1, x2, y2 = f["box"]
        # TOẠ ĐỘ CHUẨN HOÁ [0,1] → độc lập độ phân giải ảnh (chung hệ với /track)
        cxn, cyn = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        wsn = (x2 - x1) / W
        bn, bs = f.get("best_name"), float(f.get("score", 0.0))

        bt = _match_track_idx(tracks, cxn, cyn, thr, used)
        if bt is None:
            t = _new_track(cxn, cyn, now)
            tracks.append(t); used.add(len(tracks) - 1)
        else:
            t = tracks[bt]
            _reacquire_if_stale(t, now)   # gián đoạn dài → có thể người KHÁC, xoá danh tính cũ
            t["cx"], t["cy"], t["seen"] = cxn, cyn, now
            used.add(bt)
        if bn != SKIP_VOTE:   # khung mờ → không tính phiếu, giữ nguyên hist cũ (xem BLUR_THR)
            t["hist"].append((bn, bs))

        # ── ĐO CHUYỂN ĐỘNG (theo TỐC ĐỘ, độc lập FPS): đứng im hay di chuyển ──
        t["pos"].append((cxn, cyn, now))
        t["moving"] = _is_moving(t["pos"], wsn)

        # gộp điểm theo từng tên trong cửa sổ → chọn danh tính nổi trội
        by = defaultdict(list)
        for n, s in t["hist"]:
            if n is not None:
                by[n].append(s)
        if by:
            dom = max(by, key=lambda n: (len(by[n]), sum(by[n]) / len(by[n])))
            mean_s   = sum(by[dom]) / len(by[dom])
            presence = len(by[dom]) / len(t["hist"])
            # hysteresis: nếu track đang mang chính tên này → ngưỡng giữ thấp hơn
            keep_thr = threshold - HYSTERESIS if t["label"] == dom else threshold
            if presence >= MIN_PRESENCE and mean_s >= keep_thr:
                t["label"] = dom
            else:
                t["label"] = "LẠ"
            disp = mean_s
        else:
            t["label"] = "LẠ"
            disp = bs
        out.append((t["label"], round(disp, 3)))

        # ── GHI NHẬT KÝ QUÉT ──
        #  ĐỨNG IM  : một danh tính đứng đủ STABLE_SECONDS → ghi 1 lần; lặp lại thì BỎ QUA.
        #  DI CHUYỂN: ghi OK/FAIL mỗi lần chuyển trạng thái (son→lạ→son…).
        old = t.get("logged_label", "")
        lbl = t["label"]
        if lbl != "LẠ":
            t["ever_known"] = True               # track này đã từng là người quen
        key = lbl if lbl != "LẠ" else "Người lạ"

        if t["moving"]:
            # ── DI CHUYỂN: ghi MỖI frame nhận diện (đi qua thì ghi nhiều), fail khi hỏng ──
            if lbl != "LẠ" and len(t["hist"]) >= 1:
                _log_scan(lbl, "OK", cam, "Di chuyển"); t["logged_label"] = lbl
            elif lbl == "LẠ" and old not in ("", "LẠ"):
                _log_scan(old, "FAIL", cam, "Di chuyển"); t["logged_label"] = lbl
            elif lbl == "LẠ" and old != "LẠ" and not t["ever_known"] and len(t["hist"]) >= STRANGER_MIN:
                _log_scan("Người lạ", "LẠ", cam, "Di chuyển"); t["logged_label"] = lbl
            # đang di chuyển → reset đồng hồ đứng im (GIỮ cờ đã-ghi: đứng im chỉ ghi 1 lần)
            st.stable_start.pop(key, None); st.stable_seen.pop(key, None)
        else:
            # ── ĐỨNG IM: mỗi người CHỈ ghi 1 lần DUY NHẤT trên camera này ──
            ok_stable = (lbl != "LẠ" and len(t["hist"]) >= 1)
            la_stable = (lbl == "LẠ" and not t["ever_known"] and len(t["hist"]) >= STRANGER_MIN)
            if (ok_stable or la_stable) and key not in st.stable_logged:
                prev = st.stable_seen.get(key)
                if prev is None or (now - prev).total_seconds() > STABLE_GAP:
                    st.stable_start[key] = now          # bắt đầu tính giờ đứng
                st.stable_seen[key] = now
                if (now - st.stable_start[key]).total_seconds() >= STABLE_SECONDS:
                    _log_scan(key, "LẠ" if key == "Người lạ" else "OK", cam, "Đứng im")
                    st.stable_logged.add(key)           # đã ghi → không ghi đứng im lại (cam này)
                    t["logged_label"] = lbl

    return out

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    """Kiểm tra server còn sống — dùng cho Docker HEALTHCHECK."""
    return {
        "status": "ok",
        "arch": "YOLO26 detect + ArcFace embedding",
        "device": rec.device,
        "light_mode": LIGHT_MODE,
        "people": rec.db_names,
        "db_size": 0 if rec.db_embs is None else int(rec.db_embs.shape[0]),
        "threshold": THRESHOLD,
        "plc_connected": plc.connected,
    }


@app.get("/plc-test")
def plc_test(color: str = Query("green")):
    """Bấm thử đèn tháp từ trình duyệt (debug đấu dây/địa chỉ Y trước khi chạy thật):
    GET /plc-test?color=green | yellow | red"""
    if color not in PLC_COILS:
        return JSONResponse({"ok": False, "msg": f"Màu '{color}' không hợp lệ (green/yellow/red)"},
                            status_code=400)
    plc.signal(color)
    return {"ok": True, "color": color, "device": PLC_COILS[color]}


@app.post("/process-frame")
async def process_frame(
    file: UploadFile = File(...),
    conf_detect: float = Form(CONF_DETECT),   # ngưỡng YOLO bắt mặt (web gửi lên)
    conf_known: float = Form(THRESHOLD),      # ngưỡng COSINE người quen/lạ (web gửi lên)
    cam: str = Form("cam1"),                  # camera nào gửi frame (state tracking riêng)
):
    """
    Nhận frame webcam/video → YOLO khoanh mặt → ArcFace embedding → cosine với DB.
    Trả ảnh đã annotate (base64 JPEG). Ngưỡng chỉnh trực tiếp từ web.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"status": "error", "message": "Invalid image"}, status_code=400)
    img = _enhance(img, cam)   # bù mềm/thiếu sáng cho cam thiếu AF/HDR (xem CAM_ENHANCE)

    # kẹp giá trị hợp lệ để tránh web gửi số vô lý
    conf_detect = min(max(conf_detect, 0.05), 0.9)
    threshold   = min(max(conf_known, 0.05), 0.95)   # conf_known giờ là ngưỡng cosine
    # siết riêng theo camera: cam2 (Rapoo) ảnh mềm → ngưỡng cao hơn + đòi khoảng cách top1/top2
    tune     = CAM_TUNE.get(cam, CAM_TUNE["cam1"])
    threshold = min(threshold + tune["thr_bump"], 0.95)
    margin    = tune["margin"]

    st = _cam_states[cam]                 # state tracking riêng của camera này
    rec.det_conf = conf_detect
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    H, W = img.shape[:2]

    # ── NHẬN DIỆN CÓ CACHE (tiết kiệm CPU) ─────────────────────────────────
    # 1) YOLO tìm mọi mặt (nhanh). 2) Mặt nào đã nhận ra chắc & còn hạn → DÙNG LẠI tên,
    #    KHÔNG chạy ArcFace. 3) ArcFace chỉ chạy cho mặt MỚI / chưa chắc / quá hạn refresh.
    boxes = await loop.run_in_executor(executor, lambda: rec.detect(img))
    now_dt = datetime.now()
    thr_m = _match_thr()
    blur_thr = BLUR_THR.get(cam, BLUR_THR["cam1"])
    faces = [None] * len(boxes)
    trk_of, need = [], []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b[:4]
        cxn, cyn = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        tr = _nearest_track(st.tracks, cxn, cyn, thr_m)
        trk_of.append(tr)
        if (tr is not None and tr["label"] != "LẠ" and tr["arc_time"] is not None
                and (now_dt - tr["arc_time"]).total_seconds() < REFRESH_SEC):
            faces[i] = {"box": (x1, y1, x2, y2), "best_name": tr["label"],
                        "score": tr["arc_score"], "runner": None}   # DÙNG LẠI (bỏ ArcFace)
        elif rec.is_blurry(img, (x1, y1, x2, y2), blur_thr):
            # khung mờ (motion-blur) → bỏ ArcFace hẳn, không tính phiếu (xem BLUR_THR/SKIP_VOTE)
            faces[i] = {"box": (x1, y1, x2, y2), "best_name": SKIP_VOTE, "score": 0.0, "runner": None}
        else:
            need.append(i)
    if need:
        need_boxes = [boxes[i][:4] for i in need]
        emb, keep = await loop.run_in_executor(executor, lambda: rec.embed(img, need_boxes))
        got = {}
        for j, k in enumerate(keep):
            ranked = rec._rank(emb[j])
            if ranked:
                top_name, top_score = ranked[0]
                run_score = ranked[1][1] if len(ranked) > 1 else 0.0
                runner = ({"name": ranked[1][0], "score": round(ranked[1][1], 3)}
                          if len(ranked) > 1 else None)
                # margin gate: top-1 sát top-2 → NHẬP NHẰNG 2 người quen, bỏ tên (best_name=None)
                # để frame này thành phiếu "chưa chắc"; bộ làm mượt cần đa số frame chắc mới gán.
                # Mặt ĐANG DI CHUYỂN vốn mờ hơn (motion blur) → top1/top2 tự nhiên sát nhau hơn
                # người đứng im; đòi margin y hệt sẽ loại oan hầu hết phiếu của người đi qua → giảm
                # nửa margin khi track đang moving (đã có sẵn t["moving"] cập nhật từ /track).
                m_tr = trk_of[need[k]]
                m = margin * 0.5 if (m_tr is not None and m_tr.get("moving")) else margin
                name = None if (top_score - run_score) < m else top_name
                got[k] = (name, round(top_score, 3), runner)
        for pos, i in enumerate(need):
            x1, y1, x2, y2 = boxes[i][:4]
            info = got.get(pos)
            if info is None:
                faces[i] = {"box": (x1, y1, x2, y2), "best_name": None, "score": 0.0, "runner": None}
            else:
                bn, bs, runner = info
                faces[i] = {"box": (x1, y1, x2, y2), "best_name": bn, "score": bs, "runner": runner}

                if trk_of[i] is not None:
                    trk_of[i]["arc_time"] = now_dt
                    trk_of[i]["arc_score"] = bs
    faces = [f for f in faces if f is not None]
    infer_ms = (time.perf_counter() - t0) * 1000

    n0 = len(_scan_log)
    voted = smooth_labels(faces, W, H, threshold, st, cam)   # làm mượt + ghi log theo camera
    # sự kiện quét MỚI sinh ra ở frame này → gửi về client hiện live (kèm tên camera)
    scan_events = [{"time": e["dt"].strftime("%H:%M:%S"), "person": e["person"],
                    "status": e["status"], "motion": e["motion"], "cam": e["cam"]}
                   for e in _scan_log[n0:]]

    stranger = False
    known = []
    detections = []

    # Chỉ TRẢ TOẠ ĐỘ KHUNG (client tự vẽ overlay lên video) — không encode ảnh
    # → nhẹ băng thông, detect nhanh hơn (bỏ imencode + base64).
    for f, (vname, vscore) in zip(faces, voted):
        x1, y1, x2, y2 = f["box"]
        runner = f.get("runner")
        box = [int(x1), int(y1), int(x2), int(y2)]
        if vname == "LẠ":
            stranger = True
            detections.append({"name": "Người lạ", "conf": vscore, "stranger": True,
                               "runner": runner, "box": box})
        else:
            known.append(vname)
            detections.append({"name": vname, "conf": vscore, "stranger": False,
                               "runner": runner, "box": box})

    known_unique = list(dict.fromkeys(known))   # bỏ trùng, giữ thứ tự
    # sắp xếp: người lạ lên đầu, rồi theo cosine giảm dần
    detections.sort(key=lambda d: (not d["stranger"], -d["conf"]))
    return {
        "img_w": W, "img_h": H,   # kích thước frame (client scale toạ độ khung)
        "stranger": stranger,
        "known": known_unique,
        "detections": detections,
        "count": len(faces),
        "infer_ms": round(infer_ms, 1),
        "scan_events": scan_events,   # sự kiện OK/FAIL/LẠ mới (hiện live trên UI)
    }


@app.post("/track")
async def track_faces(
    file: UploadFile = File(...),
    conf_detect: float = Form(CONF_DETECT),
    cam: str = Form("cam1"),                  # camera nào (state tracking riêng)
):
    """CHỈ chạy YOLO (nhanh ~50ms) → trả TOẠ ĐỘ KHUNG bám sát thời gian thực.
    Tên/điểm lấy từ track gần nhất (do /process-frame cập nhật) → không phải chờ ArcFace.
    Client gọi endpoint này liên tục cho khung mượt + biến mất nhanh khi rời khung hình."""
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"status": "error", "message": "Invalid image"}, status_code=400)

    rec.det_conf = min(max(conf_detect, 0.05), 0.9)
    loop = asyncio.get_event_loop()
    H, W = img.shape[:2]
    # /track chạy nhanh ~10fps chỉ để bám khung (tên lấy từ cache, không chạy ArcFace) → KHÔNG
    # áp _enhance ở đây, đỡ tốn CPU thêm ở vòng lặp tần suất cao; nhận dạng thật (nơi _enhance
    # có tác dụng) nằm ở /process-frame. YOLO imgsz nhỏ → nhanh hơn nhiều trên CPU.
    boxes = await loop.run_in_executor(executor, lambda: rec.detect(img, imgsz=TRACK_IMGSZ))

    st = _cam_states[cam]                 # state tracking riêng của camera này
    match_thr = _match_thr()
    now = datetime.now()
    _prune_tracks(st, now)   # dọn track cũ TRƯỚC khi ghép
    tracks = st.tracks
    used, dets = set(), []
    for (x1, y1, x2, y2, conf) in boxes:
        # TOẠ ĐỘ CHUẨN HOÁ [0,1] → CHUNG hệ với /process-frame 
        cxn, cyn = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        wsn = (x2 - x1) / W
        bi = _match_track_idx(tracks, cxn, cyn, match_thr, used)
        if bi is None:
            # /track TẠO track mới → chờ /process-frame gán tên.
            t = _new_track(cxn, cyn, now)
            tracks.append(t); bi = len(tracks) - 1
        else:
            t = tracks[bi]
            _reacquire_if_stale(t, now)   # gián đoạn dài → có thể người KHÁC, xoá danh tính cũ
            t["cx"], t["cy"], t["seen"] = cxn, cyn, now
        used.add(bi)
        # GÓP dữ liệu chuyển động ở nhịp nhanh của /track → đo đứng/đi chính xác
        t["pos"].append((cxn, cyn, now))
        t["moving"] = _is_moving(t["pos"], wsn)

        lab = t["label"]
        if not t["hist"]:                 
            name, score, stranger, pending = "…", 0.0, False, True
        elif lab == "LẠ":
            name, score, stranger, pending = "Người lạ", t["hist"][-1][1], True, False
        else:
            name, score, stranger, pending = lab, t["hist"][-1][1], False, False
        dets.append({"name": name, "conf": round(float(score), 3), "stranger": stranger,
                     "pending": pending, "moving": t["moving"],
                     "box": [int(x1), int(y1), int(x2), int(y2)]})

    dets.sort(key=lambda d: (not d["stranger"], -d["conf"]))
    return {"img_w": W, "img_h": H, "detections": dets, "count": len(boxes)}


@app.post("/capture")
async def capture(file: UploadFile = File(...), person: str = Form(...), role: str = Form("kỹ sư")):
    """Thêm 1 mẫu ảnh vào DB (không cần train lại). Ghi ra face_db.pt."""
    person = (person or "").strip()
    if not person:
        return JSONResponse({"ok": False, "msg": "Chưa nhập tên người"}, status_code=200)

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"ok": False, "msg": "Ảnh lỗi"}, status_code=400)

    vec = rec.embed_largest(img)   # YOLO khoanh mặt to nhất → ArcFace embedding
    if vec is None:
        return JSONResponse({"ok": False, "msg": "Không thấy/căn được khuôn mặt — đưa mặt gần & thẳng hơn"},
                            status_code=200)

    is_new = person not in rec.db_names
    rec.add_embedding(person, vec, role=role)   # thêm vào DB in-memory (tạo người mới nếu chưa có)
    rec.save_db()                    # ghi ra face_db.pt

    count = int((rec.db_owner == rec.db_names.index(person)).sum())
    total = rec.db_embs.shape[0]
    return {"ok": True, "person": person, "count": count, "total": total,
            "is_new": is_new, "people": rec.people_summary()}


@app.get("/people")
def people():
    return {"people": rec.people_summary(),
            "total": 0 if rec.db_embs is None else int(rec.db_embs.shape[0])}


@app.post("/update-role")
async def update_role(person: str = Form(...), role: str = Form(...)):
    person = (person or "").strip()
    if person not in rec.db_names:
        return JSONResponse({"ok": False, "msg": f"Không tìm thấy '{person}'"}, status_code=200)
    rec.db_roles[person] = role
    rec.save_db()
    return {"ok": True, "person": person, "role": role, "people": rec.people_summary()}


@app.post("/remove-person")
async def remove_person(person: str = Form(...)):
    """Xoá 1 người khỏi DB (không cần train). Ghi lại face_db.pt."""
    person = (person or "").strip()
    ok = rec.remove_person(person)
    if ok:
        rec.save_db()
    return {"ok": ok, "person": person, "people": rec.people_summary()}


@app.get("/people/samples")
def people_samples(person: str):
    """Chẩn đoán từng mẫu của 1 người — tìm mẫu enroll NHẦM mà không cần xem lại ảnh."""
    return {"person": person, "samples": rec.sample_diagnostics((person or "").strip())}


@app.post("/remove-sample")
async def remove_sample(person: str = Form(...), index: int = Form(...)):
    """Xoá 1 MẪU cụ thể của 'person' theo index lấy từ /people/samples. Ghi lại face_db.pt."""
    person = (person or "").strip()
    ok = rec.remove_embedding(person, index)
    if ok:
        rec.save_db()
    return {"ok": ok, "person": person, "people": rec.people_summary()}

def _room_snapshot():
    """Snapshot trạng thái phòng theo vai trò dựa trên _in_room hiện tại."""
    ks_present = [n for n in _in_room if rec.db_roles.get(n, "kỹ sư") == "kỹ sư"]
    sv_present = [n for n in _in_room if rec.db_roles.get(n, "sinh viên") == "sinh viên"]
    total_ks   = sum(1 for n in rec.db_names if rec.db_roles.get(n, "kỹ sư") == "kỹ sư")
    total_sv   = sum(1 for n in rec.db_names if rec.db_roles.get(n, "sinh viên") == "sinh viên")
    return ks_present, sv_present, total_ks, total_sv


def _log_attendance(person: str, cam: str):
    """Check-in/out THEO CAMERA (xem CAM_ROLE) — gọi từ _log_scan() ngay khi 1 track được
    xác nhận OK, không cần client tự gộp/poll trạng thái 2 cam. Idempotent: OK lặp lại khi
    di chuyển (mỗi frame) không tạo thêm sự kiện vì đã kiểm tra _in_room trước khi ghi."""
    role = CAM_ROLE.get(cam)
    now = datetime.now()
    if role == "vào" and person not in _in_room:
        _in_room[person] = now
        ks_p, sv_p, t_ks, t_sv = _room_snapshot()
        _attendance_log.append({
            "dt": now, "type": "vào", "person": person,
            "role": rec.db_roles.get(person, "kỹ sư"),
            "duration_min": None,
            "ks_present": list(ks_p), "sv_present": list(sv_p),
            "total_ks": t_ks, "total_sv": t_sv,
        })
    elif role == "ra" and person in _in_room:
        entry_time = _in_room.pop(person)
        dur = round((now - entry_time).total_seconds() / 60, 1)
        ks_p, sv_p, t_ks, t_sv = _room_snapshot()
        _attendance_log.append({
            "dt": now, "type": "ra", "person": person,
            "role": rec.db_roles.get(person, "kỹ sư"),
            "duration_min": dur,
            "ks_present": list(ks_p), "sv_present": list(sv_p),
            "total_ks": t_ks, "total_sv": t_sv,
        })


def _sessions_from_events(events: list[dict]) -> list[dict]:
    """Ghep cap vao/ra thanh session. Tra ve list {person, role, date, entry_dt, duration_min, ks_present, sv_present}."""
    pending  = {}
    sessions = []
    for ev in sorted(events, key=lambda e: e["dt"]):
        if ev["type"] == "vào":
            pending[ev["person"]] = ev
        elif ev["type"] == "ra" and ev["person"] in pending:
            vao = pending.pop(ev["person"])
            sessions.append({"person": ev["person"], "role": ev.get("role", "kỹ sư"),
                              "date": vao["dt"].date(), "entry_dt": vao["dt"],
                              "duration_min": ev["duration_min"],
                              "ks_present": vao.get("ks_present", []),
                              "sv_present": vao.get("sv_present", [])})
    for person, vao in pending.items():
        sessions.append({"person": person, "role": vao.get("role", "kỹ sư"),
                         "date": vao["dt"].date(), "entry_dt": vao["dt"],
                         "duration_min": None,
                         "ks_present": vao.get("ks_present", []),
                         "sv_present": vao.get("sv_present", [])})
    return sorted(sessions, key=lambda s: s["entry_dt"])


def _make_excel(events: list[dict], scans: list[dict], people: list[dict],
                mode: str = "full") -> bytes:
    """
    3 sheet: Ngay / Tuan / Thang. 
    - mode="full": nhật ký đầy đủ (đứng im + di chuyển + fail/lạ).
    - mode="once": CHỈ nhận dạng mỗi người 1 lần (bỏ di chuyển, bỏ lặp, bỏ fail/lạ).
    Tuan/Thang
    """
    from openpyxl.utils import get_column_letter

    if mode == "once":
        # chỉ giữ lần OK ĐẦU TIÊN của mỗi người
        seen, filtered = set(), []
        for e in sorted(scans, key=lambda x: x["dt"]):
            if e["status"] == "OK" and e["person"] not in seen:
                seen.add(e["person"]); filtered.append(e)
        scans = filtered

    now           = datetime.now()
    today         = now.date()
    week_start    = today - timedelta(days=today.weekday())
    month_start   = today.replace(day=1)
    days_in_month = calendar.monthrange(today.year, today.month)[1]

    VIET_DAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

    # ── STYLES ──────────────────────────────────────────────────────────────
    TITLE_KS   = PatternFill("solid", fgColor="1E3A5F")
    TITLE_SV   = PatternFill("solid", fgColor="4A235A")
    HDR_KS     = PatternFill("solid", fgColor="2874A6")
    HDR_SV     = PatternFill("solid", fgColor="7D3C98")
    DATE_FILL  = PatternFill("solid", fgColor="2C3E50")
    SUM_KS     = PatternFill("solid", fgColor="1A5276")
    SUM_SV     = PatternFill("solid", fgColor="6C3483")
    ALT_KS     = PatternFill("solid", fgColor="EBF5FB")
    ALT_SV     = PatternFill("solid", fgColor="F5EEF8")
    GREEN_DARK = PatternFill("solid", fgColor="1E8449")
    RED_DARK   = PatternFill("solid", fgColor="C0392B")
    GREEN_ROW  = PatternFill("solid", fgColor="D5F5E3")
    RED_ROW    = PatternFill("solid", fgColor="FADBD8")

    TITLE_FONT = Font(bold=True, color="FFFFFF", size=12)
    HDR_FONT   = Font(bold=True, color="FFFFFF", size=10)
    DATE_FONT  = Font(bold=True, color="FFFFFF", size=11)
    SUM_FONT   = Font(bold=True, color="FFFFFF", size=10)
    NORM       = Font(size=10)

    CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
    LEFT   = Alignment(horizontal="left",   vertical="center")
    THIN   = Side(style="thin", color="BDBDBD")
    BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

    # ── PEOPLE ──────────────────────────────────────────────────────────────
    ks_names  = sorted([p["name"] for p in people if p.get("role") == "kỹ sư"])
    sv_names  = sorted([p["name"] for p in people if p.get("role") == "sinh viên"])
    total_ks  = len(ks_names)
    total_sv  = len(sv_names)

    # ── SESSIONS & AGGREGATION ──────────────────────────────────────────────
    all_sessions = _sessions_from_events(events)

    def _agg_daily(sessions):
        """date -> person -> {role, first_entry, last_exit, total_min, count}"""
        agg = defaultdict(dict)
        for s in sessions:
            d, p = s["date"], s["person"]
            if p not in agg[d]:
                agg[d][p] = {"role": s["role"], "first_entry": s["entry_dt"],
                              "last_exit": None, "total_min": 0.0, "count": 0}
            r = agg[d][p]
            if s["entry_dt"] < r["first_entry"]:
                r["first_entry"] = s["entry_dt"]
            if s["duration_min"] is not None:
                exit_dt = s["entry_dt"] + timedelta(minutes=s["duration_min"])
                if r["last_exit"] is None or exit_dt > r["last_exit"]:
                    r["last_exit"] = exit_dt
                r["total_min"] += s["duration_min"]
            r["count"] += 1
        return agg

    daily_agg = _agg_daily(all_sessions)
    all_dates = sorted(daily_agg.keys())

    def _fmt_dur(minutes):
        if not minutes:
            return "—"
        h, m = int(minutes // 60), int(minutes % 60)
        return f"{h}h{m:02d}m" if h else f"{m}m"

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    ws1 = wb.create_sheet("Ngày")
    ORANGE_ROW = PatternFill("solid", fgColor="FDEBD0")

    OK_COLS   = ["STT", "Ngày", "Giờ", "Họ và Tên", "Camera", "Trạng thái"]
    OK_WIDTHS = [5, 10, 9, 18, 14, 11]
    NC  = len(OK_COLS)
    KS1 = 1
    GAP = KS1 + NC            # cột giữa 2 bảng (cũng là cột "Kết quả" của bảng fail)
    SV1 = GAP + 1

    # tiêu đề 2 bảng
    ws1.merge_cells(start_row=1, start_column=KS1, end_row=1, end_column=KS1+NC-1)
    c = ws1.cell(row=1, column=KS1, value="BẢNG NHÂN VIÊN (nhận diện OK)")
    c.font = TITLE_FONT; c.fill = TITLE_KS; c.alignment = CENTER
    ws1.merge_cells(start_row=1, start_column=SV1, end_row=1, end_column=SV1+NC-1)
    c = ws1.cell(row=1, column=SV1, value="BẢNG SINH VIÊN  (nhận diện OK)")
    c.font = TITLE_FONT; c.fill = TITLE_SV; c.alignment = CENTER
    ws1.row_dimensions[1].height = 26

    for ci, (h, w) in enumerate(zip(OK_COLS, OK_WIDTHS)):
        for start, fill in [(KS1, HDR_KS), (SV1, HDR_SV)]:
            c = ws1.cell(row=2, column=start+ci, value=h)
            c.font = HDR_FONT; c.fill = fill; c.alignment = CENTER; c.border = BORDER
        ws1.column_dimensions[get_column_letter(KS1+ci)].width = w
        ws1.column_dimensions[get_column_letter(SV1+ci)].width = w
    ws1.column_dimensions[get_column_letter(GAP)].width = 11   # gap + cột Kết quả bảng fail
    ws1.row_dimensions[2].height = 22
    ws1.freeze_panes = "A3"

    # tách sự kiện OK theo vai trò
    ok_ks = sorted([e for e in scans if e["status"] == "OK" and e["role"] == "kỹ sư"],
                   key=lambda e: e["dt"])
    ok_sv = sorted([e for e in scans if e["status"] == "OK" and e["role"] == "sinh viên"],
                   key=lambda e: e["dt"])

    def _fill_ok(events, base, alt_fill):
        for i, e in enumerate(events):
            ri = 3 + i
            vals = [i+1, e["dt"].strftime("%d/%m"), e["dt"].strftime("%H:%M:%S"),
                    e["person"], e.get("cam", ""), e.get("motion", "Đứng im")]
            for ci, val in enumerate(vals):
                c = ws1.cell(row=ri, column=base+ci, value=val)
                c.border = BORDER; c.font = NORM
                c.alignment = LEFT if ci == 3 else CENTER
                if i % 2 == 1: c.fill = alt_fill
            ws1.row_dimensions[ri].height = 20

    _fill_ok(ok_ks, KS1, ALT_KS)
    _fill_ok(ok_sv, SV1, ALT_SV)

    n_rows  = max(len(ok_ks), len(ok_sv), 1)
    sum_row = 3 + n_rows
    ws1.merge_cells(start_row=sum_row, start_column=KS1, end_row=sum_row, end_column=KS1+NC-1)
    c = ws1.cell(row=sum_row, column=KS1,
                 value=f"Nhân viên: {len(ok_ks)} lần OK  ({len({e['person'] for e in ok_ks})} người)")
    c.font = SUM_FONT; c.fill = SUM_KS; c.alignment = CENTER
    ws1.merge_cells(start_row=sum_row, start_column=SV1, end_row=sum_row, end_column=SV1+NC-1)
    c = ws1.cell(row=sum_row, column=SV1,
                 value=f"Sinh viên: {len(ok_sv)} lần OK  ({len({e['person'] for e in ok_sv})} người)")
    c.font = SUM_FONT; c.fill = SUM_SV; c.alignment = CENTER
    ws1.row_dimensions[sum_row].height = 22

    # ── BẢNG NGƯỜI LẠ / FAIL (bên dưới, span toàn bộ) ──
    ftitle = sum_row + 2      # STT | Ngày | Giờ | Họ và Tên | Kết quả (cột Kết quả = cột GAP)
    ws1.merge_cells(start_row=ftitle, start_column=1, end_row=ftitle, end_column=SV1+NC-1)
    c = ws1.cell(row=ftitle, column=1, value="BẢNG NGƯỜI LẠ / NHẬN DIỆN FAIL")
    c.font = TITLE_FONT; c.fill = RED_DARK; c.alignment = CENTER
    ws1.row_dimensions[ftitle].height = 26

    fhdr = ftitle + 1
    for ci, h in enumerate(["STT", "Ngày", "Giờ", "Họ và Tên", "Camera", "Trạng thái", "Kết quả"]):
        c = ws1.cell(row=fhdr, column=1+ci, value=h)
        c.font = HDR_FONT; c.fill = RED_DARK; c.alignment = CENTER; c.border = BORDER
    ws1.row_dimensions[fhdr].height = 22

    fails = sorted([e for e in scans if e["status"] in ("FAIL", "LẠ")], key=lambda e: e["dt"])
    for i, e in enumerate(fails):
        ri = fhdr + 1 + i
        fill = RED_ROW if e["status"] == "FAIL" else ORANGE_ROW
        fcol = "C0392B" if e["status"] == "FAIL" else "CA6F1E"
        vals = [i+1, e["dt"].strftime("%d/%m"), e["dt"].strftime("%H:%M:%S"),
                e["person"], e.get("cam", ""), e.get("motion", "Đứng im"), e["status"]]
        for ci, val in enumerate(vals):
            c = ws1.cell(row=ri, column=1+ci, value=val)
            c.border = BORDER; c.fill = fill
            c.alignment = LEFT if ci == 3 else CENTER
            c.font = Font(bold=True, size=10, color=fcol) if ci == 6 else NORM
        ws1.row_dimensions[ri].height = 20
    if not fails:
        c = ws1.cell(row=fhdr+1, column=1, value="Không có fail / người lạ.")
        c.font = NORM; c.alignment = LEFT

    W_COLS   = ["STT", "Họ và Tên", "Ngày đi làm"]
    W_WIDTHS = [5, 22, 15]
    W_NCOLS  = len(W_COLS)
    KS2 = 1
    SV2 = KS2 + W_NCOLS + 1

    ws2 = wb.create_sheet("Tuần")

    ws2.merge_cells(start_row=1, start_column=KS2, end_row=1, end_column=KS2+W_NCOLS-1)
    c = ws2.cell(row=1, column=KS2, value="BẢNG NHÂN VIÊN")
    c.font = TITLE_FONT; c.fill = TITLE_KS; c.alignment = CENTER

    ws2.merge_cells(start_row=1, start_column=SV2, end_row=1, end_column=SV2+W_NCOLS-1)
    c = ws2.cell(row=1, column=SV2, value="BẢNG SINH VIÊN")
    c.font = TITLE_FONT; c.fill = TITLE_SV; c.alignment = CENTER
    ws2.row_dimensions[1].height = 26

    for ci, (h, w) in enumerate(zip(W_COLS, W_WIDTHS)):
        for start, fill in [(KS2, HDR_KS), (SV2, HDR_SV)]:
            c = ws2.cell(row=2, column=start+ci, value=h)
            c.font = HDR_FONT; c.fill = fill; c.alignment = CENTER; c.border = BORDER
        ws2.column_dimensions[get_column_letter(KS2+ci)].width = w
        ws2.column_dimensions[get_column_letter(SV2+ci)].width = w
    ws2.column_dimensions[get_column_letter(KS2+W_NCOLS)].width = 2
    ws2.row_dimensions[2].height = 22
    ws2.freeze_panes = "B3"

    # Compute weeks
    week_to_dates_ks = defaultdict(set)
    week_to_dates_sv = defaultdict(set)
    for s in all_sessions:
        ws = s["date"] - timedelta(days=s["date"].weekday())
        if s["role"] == "kỹ sư":
            week_to_dates_ks[(ws, s["person"])].add(s["date"])
        else:
            week_to_dates_sv[(ws, s["person"])].add(s["date"])

    all_weeks = sorted(set([s["date"] - timedelta(days=s["date"].weekday()) for s in all_sessions]))

    cur_row = 3
    for ws in all_weeks:
        we = ws + timedelta(days=6)
        days_in_week = (min(we, today) - ws).days + 1 if ws <= today else 7
        week_label = f"  Tuần {ws.strftime('%d/%m')} – {we.strftime('%d/%m/%Y')}"

        ws2.merge_cells(start_row=cur_row, start_column=1, end_row=cur_row, end_column=SV2+W_NCOLS-1)
        c = ws2.cell(row=cur_row, column=1, value=week_label)
        c.font = DATE_FONT; c.fill = DATE_FILL; c.alignment = LEFT
        ws2.row_dimensions[cur_row].height = 24
        cur_row += 1

        max_people = max(len(ks_names), len(sv_names), 1)
        for i in range(max_people):
            ri = cur_row + i
            alt = i % 2 == 1
            ws2.row_dimensions[ri].height = 20

            if i < len(ks_names):
                name = ks_names[i]
                cnt = len(week_to_dates_ks[(ws, name)])
                for ci, val in enumerate([i+1, name, f"{cnt}/{days_in_week} ngày"]):
                    c = ws2.cell(row=ri, column=KS2+ci, value=val)
                    c.border = BORDER; c.font = NORM
                    c.alignment = LEFT if ci == 1 else CENTER
                    if alt: c.fill = ALT_KS

            if i < len(sv_names):
                name = sv_names[i]
                cnt = len(week_to_dates_sv[(ws, name)])
                for ci, val in enumerate([i+1, name, f"{cnt}/{days_in_week} ngày"]):
                    c = ws2.cell(row=ri, column=SV2+ci, value=val)
                    c.border = BORDER; c.font = NORM
                    c.alignment = LEFT if ci == 1 else CENTER
                    if alt: c.fill = ALT_SV

        cur_row += max_people

        ks_present_w = len([n for n in ks_names if week_to_dates_ks[(ws, n)]])
        sv_present_w = len([n for n in sv_names if week_to_dates_sv[(ws, n)]])

        ws2.merge_cells(start_row=cur_row, start_column=KS2, end_row=cur_row, end_column=KS2+W_NCOLS-1)
        c = ws2.cell(row=cur_row, column=KS2, value=f"Nhân viên đi làm: {ks_present_w}/{total_ks} người")
        c.font = SUM_FONT; c.fill = SUM_KS; c.alignment = CENTER

        ws2.merge_cells(start_row=cur_row, start_column=SV2, end_row=cur_row, end_column=SV2+W_NCOLS-1)
        c = ws2.cell(row=cur_row, column=SV2, value=f"Sinh viên đi làm: {sv_present_w}/{total_sv} người")
        c.font = SUM_FONT; c.fill = SUM_SV; c.alignment = CENTER
        ws2.row_dimensions[cur_row].height = 22
        cur_row += 2

    M_COLS   = ["STT", "Họ và Tên", "Ngày đi làm"]
    M_WIDTHS = [5, 22, 18]
    M_NCOLS  = len(M_COLS)
    KS3 = 1
    SV3 = KS3 + M_NCOLS + 1

    ws3 = wb.create_sheet("Tháng")

    ws3.merge_cells(start_row=1, start_column=KS3, end_row=1, end_column=KS3+M_NCOLS-1)
    c = ws3.cell(row=1, column=KS3, value="BẢNG NHÂN VIÊN")
    c.font = TITLE_FONT; c.fill = TITLE_KS; c.alignment = CENTER

    ws3.merge_cells(start_row=1, start_column=SV3, end_row=1, end_column=SV3+M_NCOLS-1)
    c = ws3.cell(row=1, column=SV3, value="BẢNG SINH VIÊN")
    c.font = TITLE_FONT; c.fill = TITLE_SV; c.alignment = CENTER
    ws3.row_dimensions[1].height = 26

    for ci, (h, w) in enumerate(zip(M_COLS, M_WIDTHS)):
        for start, fill in [(KS3, HDR_KS), (SV3, HDR_SV)]:
            c = ws3.cell(row=2, column=start+ci, value=h)
            c.font = HDR_FONT; c.fill = fill; c.alignment = CENTER; c.border = BORDER
        ws3.column_dimensions[get_column_letter(KS3+ci)].width = w
        ws3.column_dimensions[get_column_letter(SV3+ci)].width = w
    ws3.column_dimensions[get_column_letter(KS3+M_NCOLS)].width = 2
    ws3.row_dimensions[2].height = 22
    ws3.freeze_panes = "B3"

    # Compute months
    month_to_dates_ks = defaultdict(set)
    month_to_dates_sv = defaultdict(set)
    for s in all_sessions:
        ms = (s["date"].year, s["date"].month)
        if s["role"] == "kỹ sư":
            month_to_dates_ks[(ms, s["person"])].add(s["date"])
        else:
            month_to_dates_sv[(ms, s["person"])].add(s["date"])

    all_months = sorted(set([(s["date"].year, s["date"].month) for s in all_sessions]))

    cur_row = 3
    for ym in all_months:
        y, m = ym
        month_start_dt = datetime(y, m, 1).date()
        month_end_dt = datetime(y, m, calendar.monthrange(y, m)[1]).date()
        days_in_m = calendar.monthrange(y, m)[1]
        month_label = f"  Tháng {m}/{y} ({days_in_m} ngày)"

        ws3.merge_cells(start_row=cur_row, start_column=1, end_row=cur_row, end_column=SV3+M_NCOLS-1)
        c = ws3.cell(row=cur_row, column=1, value=month_label)
        c.font = DATE_FONT; c.fill = DATE_FILL; c.alignment = LEFT
        ws3.row_dimensions[cur_row].height = 24
        cur_row += 1

        max_people = max(len(ks_names), len(sv_names), 1)
        for i in range(max_people):
            ri = cur_row + i
            alt = i % 2 == 1
            ws3.row_dimensions[ri].height = 20

            if i < len(ks_names):
                name = ks_names[i]
                cnt = len(month_to_dates_ks[(ym, name)])
                for ci, val in enumerate([i+1, name, f"{cnt}/{days_in_m} ngày"]):
                    c = ws3.cell(row=ri, column=KS3+ci, value=val)
                    c.border = BORDER; c.font = NORM
                    c.alignment = LEFT if ci == 1 else CENTER
                    if alt: c.fill = ALT_KS

            if i < len(sv_names):
                name = sv_names[i]
                cnt = len(month_to_dates_sv[(ym, name)])
                for ci, val in enumerate([i+1, name, f"{cnt}/{days_in_m} ngày"]):
                    c = ws3.cell(row=ri, column=SV3+ci, value=val)
                    c.border = BORDER; c.font = NORM
                    c.alignment = LEFT if ci == 1 else CENTER
                    if alt: c.fill = ALT_SV

        cur_row += max_people

        ks_present_m = len([n for n in ks_names if month_to_dates_ks[(ym, n)]])
        sv_present_m = len([n for n in sv_names if month_to_dates_sv[(ym, n)]])

        ws3.merge_cells(start_row=cur_row, start_column=KS3, end_row=cur_row, end_column=KS3+M_NCOLS-1)
        c = ws3.cell(row=cur_row, column=KS3, value=f"Nhân viên đi làm: {ks_present_m}/{total_ks} người")
        c.font = SUM_FONT; c.fill = SUM_KS; c.alignment = CENTER

        ws3.merge_cells(start_row=cur_row, start_column=SV3, end_row=cur_row, end_column=SV3+M_NCOLS-1)
        c = ws3.cell(row=cur_row, column=SV3, value=f"Sinh viên đi làm: {sv_present_m}/{total_sv} người")
        c.font = SUM_FONT; c.fill = SUM_SV; c.alignment = CENTER
        ws3.row_dimensions[cur_row].height = 22
        cur_row += 2

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


@app.get("/export")
def export_attendance(mode: str = "full"):
    """Xuat nhat ky ra Excel. mode: 'full' (đầy đủ có di chuyển) | 'once' (mỗi người 1 lần)."""
    mode = "once" if mode == "once" else "full"
    data = _make_excel(_attendance_log, _scan_log, rec.people_summary(), mode=mode)
    tag = "1lan" if mode == "once" else "daydu"
    filename = f"nhat_ky_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# Các ảnh biểu đồ mà YOLO xuất ra sau khi train.
PLOT_FILES = [
    "confusion_matrix_normalized.png", "confusion_matrix.png",
    "results.png", "BoxPR_curve.png", "BoxF1_curve.png",
    "val_batch0_pred.jpg", "val_batch0_labels.jpg",
]


def find_runs():
    """Tìm mọi thư mục run (chứa results.png) dưới BASE_DIR/runs. Trả {tên: đường_dẫn}."""
    runs = {}
    for p in glob.glob(os.path.join(BASE_DIR, "runs", "**", "results.png"), recursive=True):
        d = os.path.dirname(p)
        runs[os.path.basename(d)] = d
    # mới nhất trước
    return dict(sorted(runs.items(), key=lambda kv: os.path.getmtime(kv[1]), reverse=True))


@app.get("/results/file")
def results_file(run: str, f: str):
    """1 ảnh biểu đồ của 1 run."""
    runs = find_runs()
    d = runs.get(run)
    if not d or f not in PLOT_FILES:
        return JSONResponse({"error": "not found"}, status_code=404)
    fp = os.path.join(d, f)
    if not os.path.isfile(fp):
        return JSONResponse({"error": "no file"}, status_code=404)
    return FileResponse(fp)


@app.get("/results", response_class=HTMLResponse)
def results_page(run: str = ""):
    """Trang hiển thị confusion matrix, mAP curve, ảnh predict của run đã train."""
    runs = find_runs()
    if not runs:
        return "<body style='background:#0f172a;color:#e2e8f0;font-family:Arial;padding:40px'>" \
               "<h2>Chưa có run nào</h2>" \
               "<a href='/' style='color:#38bdf8'>← Về trang nhận diện</a></body>"
    if run not in runs:
        run = next(iter(runs))  # mặc định: run mới nhất

    options = "".join(
        f"<option value='{n}'{' selected' if n == run else ''}>{n}</option>" for n in runs
    )
    titles = {
        "confusion_matrix_normalized.png": "Confusion Matrix (chuẩn hoá) — nhìn ô quan↔tri",
        "results.png": "Đường cong loss & mAP theo epoch",
        "BoxPR_curve.png": "Precision–Recall",
        "BoxF1_curve.png": "F1 theo confidence",
        "val_batch0_pred.jpg": "Ảnh dự đoán (val)",
        "val_batch0_labels.jpg": "Ảnh nhãn gốc (val)",
    }
    cards = ""
    for f, title in titles.items():
        if os.path.isfile(os.path.join(runs[run], f)):
            cards += (
                f"<div class='card'><h3>{title}</h3>"
                f"<a href='/results/file?run={run}&f={f}' target='_blank'>"
                f"<img src='/results/file?run={run}&f={f}'></a></div>"
            )
    return f"""<!DOCTYPE html><html lang='vi'><head><meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1.0'>
<title>Kết quả Train</title><style>
*{{box-sizing:border-box}} body{{background:#0f172a;color:#e2e8f0;font-family:Arial;margin:0;padding:20px}}
h1{{color:#38bdf8;text-align:center;margin:0 0 4px}} .sub{{text-align:center;color:#94a3b8;margin-bottom:16px}}
.top{{max-width:1100px;margin:0 auto 18px;display:flex;gap:12px;align-items:center;justify-content:center;flex-wrap:wrap}}
select{{padding:9px;background:#334155;color:#fff;border:1px solid #475569;border-radius:6px;font-size:14px}}
a.back{{color:#38bdf8;text-decoration:none}}
.grid{{max-width:1100px;margin:0 auto;display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px}}
.card h3{{margin:0 0 10px;font-size:14px;color:#cbd5e1}}
.card img{{width:100%;border-radius:8px;background:#fff}}
</style></head><body>
<h1>Kết quả Training</h1><div class='sub'>Run: <b>{run}</b></div>
<div class='top'>
  <form method='get' action='/results'>
    <label style='color:#94a3b8;font-size:14px'>Chọn run: </label>
    <select name='run' onchange='this.form.submit()'>{options}</select>
  </form>
  <a class='back' href='/'>← Về trang nhận diện</a>
</div>
<div class='grid'>{cards or "<p>Run này chưa có ảnh biểu đồ.</p>"}</div>
</body></html>"""


