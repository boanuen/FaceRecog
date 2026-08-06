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
_attendance_log: list[dict] = []
_in_room: dict[str, datetime] = {}   # name → thời điểm vào phòng

# ── NHẬT KÝ QUÉT MẶT (mỗi lần 1 track ĐỔI trạng thái — dùng cho sheet Ngày) ──
# Mỗi entry: {dt, person, role, status}  status ∈ {"OK","FAIL","LẠ"}
#   OK   = một track nhận ra người quen (đứng im hay đi qua → chỉ ghi 1 lần/track)
#   FAIL = track đang là người quen X bỗng thành "người lạ" (nhận dạng hỏng)
#   LẠ   = người lạ hoàn toàn xuất hiện
# motion ∈ {"Đứng im","Di chuyển"} — trạng thái chuyển động của mặt lúc xảy ra sự kiện.
_scan_log: list[dict] = []
# ĐỨNG IM: theo dõi theo DANH TÍNH (không theo track) → đứng đủ 2s ghi 1 lần, dù track
# bị tái tạo. key = tên người (hoặc "Người lạ").
_stable_start:  dict = {}   # key → lúc bắt đầu lượt đứng hiện tại
_stable_seen:   dict = {}   # key → lần thấy đứng gần nhất
_stable_logged: set  = set()  # key đã ghi cho lượt đứng hiện tại

def _log_scan(person: str, status: str, motion: str = "Đứng im"):
    _scan_log.append({"dt": datetime.now(), "person": person,
                      "role": rec.db_roles.get(person, "kỹ sư"),
                      "status": status, "motion": motion})

print(f"[main] Khởi tạo YOLO26 + ArcFace...")
rec = FaceRecognizer(det_conf=CONF_DETECT)   # YOLO detect mặt, ArcFace nhận diện tên
print(f"[main] Device YOLO : {rec.device}")
print(f"[main] DB          : {DB_PATH}  ({len(rec.db_names)} người: {rec.db_names})")
print(f"[main] Ngưỡng      : detect={CONF_DETECT}  cosine={THRESHOLD}")

# Warm-up để lần nhận diện đầu không bị chậm
rec.recognize(np.zeros((320, 320, 3), dtype=np.uint8), threshold=THRESHOLD)
print("[main] Warm-up done")


# ── BỘ THEO DÕI + LÀM MƯỢT ĐIỂM SỐ (chống nhảy "người lạ") ────────────────
# Mỗi mặt được ghép với 1 track theo vị trí. Track giữ lịch sử (tên, điểm) vài frame.
# Quyết định dựa trên ĐIỂM TRUNG BÌNH của danh tính nổi trội trong cửa sổ + HYSTERESIS
# (đã nhận là ai thì bám lấy, một frame điểm thấp KHÔNG lật ngay sang "lạ").
_tracks = []   # mỗi track: {'cx','cy','hist':deque((name,score)), 'miss':int, 'label':str}

HYSTERESIS  = 0.04    # đã nhận tên → hạ ngưỡng 0.04 để giữ (khó rớt sang "lạ")
MIN_PRESENCE = 0.5    # danh tính phải xuất hiện >= 50% cửa sổ mới được nhận
MATCH_FRAC  = 0.18     # ngưỡng ghép track = 18% đường chéo khung (nới để bám khi di chuyển)
MOVE_STEP_FRAC = 0.15  # bước dịch TRUNG BÌNH/frame > 15% bề rộng mặt → DI CHUYỂN
STABLE_SECONDS = 2.0   # đứng im đủ 2 giây → ghi log 1 lần
STABLE_GAP  = 1.5      # gián đoạn thấy > 1.5s → coi là lượt đứng MỚI (ghi lại)
STRANGER_MIN = 4       # phải "lạ" liên tục >= 4 frame mới ghi Người lạ (bớt nhầm)

def _is_moving(pos, wsize):
    """pos: deque (cx,cy). Dùng BƯỚC DỊCH TRUNG BÌNH mỗi frame (ổn định hơn đỉnh):
    đứng im rung nhẹ → bước nhỏ; đi bộ → bước lớn."""
    if len(pos) < 3 or wsize <= 0:
        return False
    steps = [((pos[i][0] - pos[i-1][0]) ** 2 + (pos[i][1] - pos[i-1][1]) ** 2) ** 0.5
             for i in range(1, len(pos))]
    return (sum(steps) / len(steps)) > MOVE_STEP_FRAC * wsize

def _match_thr(W, H):
    return MATCH_FRAC * (W * W + H * H) ** 0.5

def smooth_labels(faces, W, H, threshold):
    """faces: list {box, best_name, score, ...}. Trả list (label, disp_score) đã làm mượt.
    label = tên hoặc 'LẠ'. disp_score = điểm trung bình (ổn định, đỡ nhảy %)."""
    global _tracks
    thr = _match_thr(W, H)
    used, out = set(), []
    for f in faces:
        x1, y1, x2, y2 = f["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bn, bs = f.get("best_name"), float(f.get("score", 0.0))

        bt, bd = None, thr
        for ti, t in enumerate(_tracks):
            if ti in used:
                continue
            dist = ((t["cx"] - cx) ** 2 + (t["cy"] - cy) ** 2) ** 0.5
            if dist < bd:
                bd, bt = dist, ti
        if bt is None:
            t = {"cx": cx, "cy": cy, "hist": deque(maxlen=SMOOTH_WINDOW),
                 "miss": 0, "label": "LẠ", "logged_label": "",
                 "pos": deque(maxlen=SMOOTH_WINDOW), "moving": False,
                 "ever_known": False}
            _tracks.append(t); used.add(len(_tracks) - 1)
        else:
            t = _tracks[bt]
            t["cx"], t["cy"], t["miss"] = cx, cy, 0
            used.add(bt)
        t["hist"].append((bn, bs))

        # ── ĐO CHUYỂN ĐỘNG: đứng im hay di chuyển ──
        t["pos"].append((cx, cy))
        t["moving"] = _is_moving(t["pos"], x2 - x1)
        motion = "Di chuyển" if t["moving"] else "Đứng im"

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
        now = datetime.now()
        old = t.get("logged_label", "")
        lbl = t["label"]
        if lbl != "LẠ":
            t["ever_known"] = True               # track này đã từng là người quen
        key = lbl if lbl != "LẠ" else "Người lạ"

        if t["moving"]:
            # ── DI CHUYỂN: ghi theo chuyển trạng thái (so với nhãn ĐÃ GHI gần nhất) ──
            if lbl != old:
                if lbl != "LẠ" and len(t["hist"]) >= 3:
                    _log_scan(lbl, "OK", "Di chuyển");        t["logged_label"] = lbl
                elif lbl == "LẠ" and old not in ("", "LẠ"):
                    _log_scan(old, "FAIL", "Di chuyển");       t["logged_label"] = lbl
                elif lbl == "LẠ" and not t["ever_known"] and len(t["hist"]) >= STRANGER_MIN:
                    _log_scan("Người lạ", "LẠ", "Di chuyển");  t["logged_label"] = lbl
            # đang di chuyển → reset đồng hồ đứng im của danh tính này
            _stable_start.pop(key, None); _stable_seen.pop(key, None)
            _stable_logged.discard(key)
        else:
            # ── ĐỨNG IM: đủ STABLE_SECONDS → ghi 1 lần ──
            ok_stable = (lbl != "LẠ" and len(t["hist"]) >= 3)
            la_stable = (lbl == "LẠ" and not t["ever_known"] and len(t["hist"]) >= STRANGER_MIN)
            if ok_stable or la_stable:
                prev = _stable_seen.get(key)
                if prev is None or (now - prev).total_seconds() > STABLE_GAP:
                    _stable_start[key] = now          # bắt đầu lượt đứng mới
                    _stable_logged.discard(key)
                _stable_seen[key] = now
                if (now - _stable_start[key]).total_seconds() >= STABLE_SECONDS \
                        and key not in _stable_logged:
                    _log_scan(key, "LẠ" if key == "Người lạ" else "OK", "Đứng im")
                    _stable_logged.add(key)
                    t["logged_label"] = lbl   # đã ghi → di chuyển sau này không ghi lại OK dư
        # KHÔNG cập nhật logged_label mỗi frame (chỉ cập nhật khi thực sự ghi ở trên)

    for ti, t in enumerate(_tracks):   # track không thấy frame này → tăng miss
        if ti not in used:
            t["miss"] += 1
    _tracks = [t for t in _tracks if t["miss"] <= 8]   # giữ track lâu hơn (mất <=8 frame)
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
        "people": rec.db_names,
        "db_size": 0 if rec.db_embs is None else int(rec.db_embs.shape[0]),
        "threshold": THRESHOLD,
    }


@app.post("/process-frame")
async def process_frame(
    file: UploadFile = File(...),
    conf_detect: float = Form(CONF_DETECT),   # ngưỡng YOLO bắt mặt (web gửi lên)
    conf_known: float = Form(THRESHOLD),      # ngưỡng COSINE người quen/lạ (web gửi lên)
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

    # kẹp giá trị hợp lệ để tránh web gửi số vô lý
    conf_detect = min(max(conf_detect, 0.05), 0.9)
    threshold   = min(max(conf_known, 0.05), 0.95)   # conf_known giờ là ngưỡng cosine

    rec.det_conf = conf_detect
    t0 = time.perf_counter()
    # chạy nhận diện (nặng, CPU) trong threadpool → không chặn endpoint /track (YOLO nhanh)
    loop = asyncio.get_event_loop()
    faces = await loop.run_in_executor(executor, lambda: rec.recognize(img, threshold=threshold))
    infer_ms = (time.perf_counter() - t0) * 1000

    H, W = img.shape[:2]
    n0 = len(_scan_log)
    voted = smooth_labels(faces, W, H, threshold)   # [(label, disp_score)] đã làm mượt
    # sự kiện quét MỚI sinh ra ở frame này → gửi về client hiện live
    scan_events = [{"time": e["dt"].strftime("%H:%M:%S"), "person": e["person"],
                    "status": e["status"], "motion": e["motion"]} for e in _scan_log[n0:]]

    stranger = False
    known = []
    detections = []

    # Chỉ TRẢ TOẠ ĐỘ KHUNG (client tự vẽ overlay lên video) — không encode ảnh
    # → nhẹ băng thông, không nhấp nháy, detect nhanh hơn (bỏ imencode + base64).
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
    boxes = await loop.run_in_executor(executor, rec.detect, img)   # YOLO only

    H, W = img.shape[:2]
    match_thr = _match_thr(W, H)
    used, dets = set(), []
    for (x1, y1, x2, y2, conf) in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # ghép khung với track gần nhất để lấy nhãn đã nhận diện gần đây
        best, bi, bd = None, -1, match_thr
        for ti, t in enumerate(_tracks):
            if ti in used:
                continue
            dist = ((t["cx"] - cx) ** 2 + (t["cy"] - cy) ** 2) ** 0.5
            if dist < bd:
                bd, best, bi = dist, t, ti
        moving = False
        if best is not None:
            # GIỮ track sống + cập nhật vị trí ở 5fps → liền mạch khi di chuyển
            best["cx"], best["cy"], best["miss"] = cx, cy, 0
            used.add(bi)
            lab = best.get("label", "LẠ")
            score = best["hist"][-1][1] if best.get("hist") else 0.0
            moving = best.get("moving", False)
            if lab == "LẠ":
                name, stranger, pending = "Người lạ", True, False
            else:
                name, stranger, pending = lab, False, False
        else:
            name, score, stranger, pending = "…", 0.0, False, True   # chưa nhận diện kịp → xám
        dets.append({"name": name, "conf": round(float(score), 3), "stranger": stranger,
                     "pending": pending, "moving": moving,
                     "box": [int(x1), int(y1), int(x2), int(y2)]})

    dets.sort(key=lambda d: (not d["stranger"], -d["conf"]))
    return {"img_w": W, "img_h": H, "detections": dets, "count": len(boxes)}


@app.post("/capture")
async def capture(file: UploadFile = File(...), person: str = Form(...), role: str = Form("kỹ sư")):
    """ENROLL trực tiếp từ webcam: lấy mặt to nhất → ArcFace embedding → thêm vào DB.

    KHÔNG train lại. 'person' có thể là tên MỚI (tự tạo người mới) hoặc người đã có.
    Ảnh webcam thật nạp thẳng vào DB → đóng khoảng cách train/thực-tế. Ghi đè face_db.pt.
    """
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
    rec.save_db()                    # ghi ra face_db.pt (bền sau khi tắt)

    count = int((rec.db_owner == rec.db_names.index(person)).sum())
    total = rec.db_embs.shape[0]
    return {"ok": True, "person": person, "count": count, "total": total,
            "is_new": is_new, "people": rec.people_summary()}


@app.get("/people")
def people():
    """Danh sách người trong DB + số mẫu mỗi người (cho web quản lý add/remove)."""
    return {"people": rec.people_summary(),
            "total": 0 if rec.db_embs is None else int(rec.db_embs.shape[0])}


@app.post("/update-role")
async def update_role(person: str = Form(...), role: str = Form(...)):
    """Đổi role của người đã có trong DB mà không mất embedding."""
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


# ======================= NHẬT KÝ + EXPORT =======================

def _room_snapshot():
    """Snapshot trạng thái phòng theo vai trò dựa trên _in_room hiện tại."""
    ks_present = [n for n in _in_room if rec.db_roles.get(n, "kỹ sư") == "kỹ sư"]
    sv_present = [n for n in _in_room if rec.db_roles.get(n, "sinh viên") == "sinh viên"]
    total_ks   = sum(1 for n in rec.db_names if rec.db_roles.get(n, "kỹ sư") == "kỹ sư")
    total_sv   = sum(1 for n in rec.db_names if rec.db_roles.get(n, "sinh viên") == "sinh viên")
    return ks_present, sv_present, total_ks, total_sv


@app.post("/log-event")
async def log_event(
    current_names: str = Form(""),   # tên người quen đang nhìn thấy, cách nhau dấu phẩy
):

    now     = datetime.now()
    current = {n.strip() for n in current_names.split(",") if n.strip()}
    prev    = set(_in_room.keys())

    entered = current - prev
    exited  = prev - current

    for name in entered:
        _in_room[name] = now
        ks_p, sv_p, t_ks, t_sv = _room_snapshot()
        _attendance_log.append({
            "dt": now, "type": "vào", "person": name,
            "role": rec.db_roles.get(name, "kỹ sư"),
            "duration_min": None,
            "ks_present": list(ks_p), "sv_present": list(sv_p),
            "total_ks": t_ks, "total_sv": t_sv,
        })

    for name in exited:
        entry_time = _in_room.pop(name)
        dur = round((now - entry_time).total_seconds() / 60, 1)
        ks_p, sv_p, t_ks, t_sv = _room_snapshot()
        _attendance_log.append({
            "dt": now, "type": "ra", "person": name,
            "role": rec.db_roles.get(name, "kỹ sư"),
            "duration_min": dur,
            "ks_present": list(ks_p), "sv_present": list(sv_p),
            "total_ks": t_ks, "total_sv": t_sv,
        })

    return {"ok": True, "in_room": list(_in_room.keys()), "total_events": len(_attendance_log)}


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


def _make_excel(events: list[dict], scans: list[dict], people: list[dict]) -> bytes:
    """
    3 sheet: Ngay / Tuan / Thang.
    - Ngay : NHẬT KÝ QUÉT (mỗi sự kiện OK/FAIL/LẠ 1 dòng), nhóm theo ngày.
    - Tuan/Thang: bảng chấm công theo phiên (giữ nguyên như cũ).
    """
    from openpyxl.utils import get_column_letter

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

    # ════════════════════════════════════════════════════════════════════════
    # SHEET NGÀY  —  NHẬT KÝ QUÉT
    #   BẢNG KỸ SƯ (trái) | BẢNG SINH VIÊN (phải)  → các lần nhận diện OK
    #   BẢNG NGƯỜI LẠ / FAIL (bên dưới)            → các lần fail + người lạ
    # ════════════════════════════════════════════════════════════════════════
    ws1 = wb.create_sheet("Ngày")
    ORANGE_ROW = PatternFill("solid", fgColor="FDEBD0")

    OK_COLS   = ["STT", "Ngày", "Giờ", "Họ và Tên", "Trạng thái"]
    OK_WIDTHS = [5, 10, 9, 20, 11]
    NC  = len(OK_COLS)
    KS1 = 1
    GAP = KS1 + NC            # cột giữa 2 bảng (cũng là cột "Kết quả" của bảng fail)
    SV1 = GAP + 1

    # tiêu đề 2 bảng
    ws1.merge_cells(start_row=1, start_column=KS1, end_row=1, end_column=KS1+NC-1)
    c = ws1.cell(row=1, column=KS1, value="BẢNG KỸ SƯ  (nhận diện OK)")
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
                    e["person"], e.get("motion", "Đứng im")]
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
                 value=f"KS: {len(ok_ks)} lần OK  ({len({e['person'] for e in ok_ks})} người)")
    c.font = SUM_FONT; c.fill = SUM_KS; c.alignment = CENTER
    ws1.merge_cells(start_row=sum_row, start_column=SV1, end_row=sum_row, end_column=SV1+NC-1)
    c = ws1.cell(row=sum_row, column=SV1,
                 value=f"SV: {len(ok_sv)} lần OK  ({len({e['person'] for e in ok_sv})} người)")
    c.font = SUM_FONT; c.fill = SUM_SV; c.alignment = CENTER
    ws1.row_dimensions[sum_row].height = 22

    # ── BẢNG NGƯỜI LẠ / FAIL (bên dưới, span toàn bộ) ──
    ftitle = sum_row + 2      # STT | Ngày | Giờ | Họ và Tên | Kết quả (cột Kết quả = cột GAP)
    ws1.merge_cells(start_row=ftitle, start_column=1, end_row=ftitle, end_column=SV1+NC-1)
    c = ws1.cell(row=ftitle, column=1, value="BẢNG NGƯỜI LẠ / NHẬN DIỆN FAIL")
    c.font = TITLE_FONT; c.fill = RED_DARK; c.alignment = CENTER
    ws1.row_dimensions[ftitle].height = 26

    fhdr = ftitle + 1
    for ci, h in enumerate(["STT", "Ngày", "Giờ", "Họ và Tên", "Trạng thái", "Kết quả"]):
        c = ws1.cell(row=fhdr, column=1+ci, value=h)
        c.font = HDR_FONT; c.fill = RED_DARK; c.alignment = CENTER; c.border = BORDER
    ws1.row_dimensions[fhdr].height = 22

    fails = sorted([e for e in scans if e["status"] in ("FAIL", "LẠ")], key=lambda e: e["dt"])
    for i, e in enumerate(fails):
        ri = fhdr + 1 + i
        fill = RED_ROW if e["status"] == "FAIL" else ORANGE_ROW
        fcol = "C0392B" if e["status"] == "FAIL" else "CA6F1E"
        vals = [i+1, e["dt"].strftime("%d/%m"), e["dt"].strftime("%H:%M:%S"),
                e["person"], e.get("motion", "Đứng im"), e["status"]]
        for ci, val in enumerate(vals):
            c = ws1.cell(row=ri, column=1+ci, value=val)
            c.border = BORDER; c.fill = fill
            c.alignment = LEFT if ci == 3 else CENTER
            c.font = Font(bold=True, size=10, color=fcol) if ci == 5 else NORM
        ws1.row_dimensions[ri].height = 20
    if not fails:
        c = ws1.cell(row=fhdr+1, column=1, value="Không có fail / người lạ.")
        c.font = NORM; c.alignment = LEFT

    # ════════════════════════════════════════════════════════════════════════
    # SHEET TUẦN  —  mỗi tuần 1 nhóm (separator + hàng người + dòng tổng)
    # Cột: STT | Họ và Tên | Ngày đi làm (X/7)
    # ════════════════════════════════════════════════════════════════════════
    W_COLS   = ["STT", "Họ và Tên", "Ngày đi làm"]
    W_WIDTHS = [5, 22, 15]
    W_NCOLS  = len(W_COLS)
    KS2 = 1
    SV2 = KS2 + W_NCOLS + 1

    ws2 = wb.create_sheet("Tuần")

    ws2.merge_cells(start_row=1, start_column=KS2, end_row=1, end_column=KS2+W_NCOLS-1)
    c = ws2.cell(row=1, column=KS2, value="BẢNG KỸ SƯ")
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
        c = ws2.cell(row=cur_row, column=KS2, value=f"KS đi làm: {ks_present_w}/{total_ks} người")
        c.font = SUM_FONT; c.fill = SUM_KS; c.alignment = CENTER

        ws2.merge_cells(start_row=cur_row, start_column=SV2, end_row=cur_row, end_column=SV2+W_NCOLS-1)
        c = ws2.cell(row=cur_row, column=SV2, value=f"SV đi làm: {sv_present_w}/{total_sv} người")
        c.font = SUM_FONT; c.fill = SUM_SV; c.alignment = CENTER
        ws2.row_dimensions[cur_row].height = 22
        cur_row += 2

    # ════════════════════════════════════════════════════════════════════════
    # SHEET THÁNG  —  mỗi tháng 1 nhóm (separator + hàng người + dòng tổng)
    # Cột: STT | Họ và Tên | Ngày đi làm (X/31 ...)
    # ════════════════════════════════════════════════════════════════════════
    M_COLS   = ["STT", "Họ và Tên", "Ngày đi làm"]
    M_WIDTHS = [5, 22, 18]
    M_NCOLS  = len(M_COLS)
    KS3 = 1
    SV3 = KS3 + M_NCOLS + 1

    ws3 = wb.create_sheet("Tháng")

    ws3.merge_cells(start_row=1, start_column=KS3, end_row=1, end_column=KS3+M_NCOLS-1)
    c = ws3.cell(row=1, column=KS3, value="BẢNG KỸ SƯ")
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
        c = ws3.cell(row=cur_row, column=KS3, value=f"KS đi làm: {ks_present_m}/{total_ks} người")
        c.font = SUM_FONT; c.fill = SUM_KS; c.alignment = CENTER

        ws3.merge_cells(start_row=cur_row, start_column=SV3, end_row=cur_row, end_column=SV3+M_NCOLS-1)
        c = ws3.cell(row=cur_row, column=SV3, value=f"SV đi làm: {sv_present_m}/{total_sv} người")
        c.font = SUM_FONT; c.fill = SUM_SV; c.alignment = CENTER
        ws3.row_dimensions[cur_row].height = 22
        cur_row += 2

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


@app.get("/export")
def export_attendance():
    """Xuat nhat ky ra Excel (3 sheet: Ngay / Tuan / Thang)."""
    data = _make_excel(_attendance_log, _scan_log, rec.people_summary())
    filename = f"nhat_ky_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# Các ảnh biểu đồ mà YOLO xuất ra sau khi train (nằm trong thư mục run).
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
    """Phục vụ 1 ảnh biểu đồ của 1 run (có kiểm tra an toàn đường dẫn)."""
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
               "<h2>Chưa có run nào (chưa train xong).</h2>" \
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


