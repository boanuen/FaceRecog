"""

  YOLO26 là DETECTOR chính (nhanh, đã fine-tune trên đúng 5 mặt này nên bắt mặt rất chắc).
  ArcFace chỉ căn chỉnh + tạo embedding trên vùng mặt YOLO đã khoanh.

Nhận diện = KHOẢNG CÁCH embedding, không phải softmax 5 lớp:
  → "người lạ" đáng tin (không bị ép chọn 1 trong 5)
  → thêm người = thêm vào DB, KHÔNG train lại.
"""
import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from insightface.app import FaceAnalysis

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "best.pt")
DB_PATH   = os.path.join(BASE_DIR, "face_db.pt")

MARGIN = 0.30   # nới quanh box YOLO trước khi đưa vào ArcFace (cần ít context để căn landmark)


class FaceRecognizer:
    def __init__(self, yolo_path=YOLO_PATH, db_path=DB_PATH, device=None,
                 det_imgsz=512, det_conf=0.25, det_iou=0.45, use_gpu_arcface=False,
                 arc_det_size=(224, 224)):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.det_imgsz = det_imgsz
        self.det_conf  = det_conf
        self.det_iou   = det_iou
        self.half      = self.device == "cuda"

        # YOLO26 — detector khuôn mặt
        self.yolo = YOLO(yolo_path)

        # ArcFace — căn chỉnh + embedding. CPU mặc định (onnxruntime CPU).
        # arc_det_size nhỏ hơn (224 thay vì 320) → SCRFD nhanh hơn ~30% mà vẫn căn tốt
        # use_gpu_arcface=True cần onnxruntime-gpu + đủ CUDA runtime libs 
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if use_gpu_arcface else ["CPUExecutionProvider"])
        self.arc = FaceAnalysis(name="buffalo_l", providers=providers,
                                allowed_modules=["detection", "recognition"])
        self.arc.prepare(ctx_id=0 if use_gpu_arcface else -1, det_size=arc_det_size)

        # DB phẳng để so khớp bằng 1 phép nhân ma trận
        self.db_names = []      # ['nghia','quan',...]
        self.db_roles = {}      # {name: 'kỹ sư'|'sinh viên'}
        self.db_embs  = None    # tensor [M,512] (L2-normalize)
        self.db_owner = None    # tensor [M] -> chỉ số vào db_names
        self.db_path  = db_path
        if db_path and os.path.isfile(db_path):
            self.load_db(db_path)

    def detect(self, img_bgr):
        """Trả list (x1,y1,x2,y2,conf) của MỌI mặt."""
        r = self.yolo(
            img_bgr, conf=self.det_conf, iou=self.det_iou,
            imgsz=self.det_imgsz, half=self.half,
            device=0 if self.device == "cuda" else "cpu",
            end2end=False, verbose=False,
        )[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out

    def _embed_crop(self, img_bgr, box):
        """Cắt vùng box (có margin) → ArcFace tự căn 5 landmark + embedding. Trả vec[512]"""
        H, W = img_bgr.shape[:2]
        x1, y1, x2, y2 = box[:4]
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * MARGIN), int(bh * MARGIN)
        x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
        x2 = min(W, x2 + mx); y2 = min(H, y2 + my)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        faces = self.arc.get(crop)
        if not faces:
            return None
        # crop chỉ nên có 1 mặt; nếu nhiều, lấy mặt to nhất
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return torch.tensor(f.normed_embedding, dtype=torch.float32)

    def embed(self, img_bgr, boxes):
        """boxes → (emb [K,512] normalized, keep: chỉ số box embed thành công)."""
        vecs, keep = [], []
        for i, b in enumerate(boxes):
            v = self._embed_crop(img_bgr, b)
            if v is not None:
                vecs.append(v); keep.append(i)
        if not vecs:
            return torch.empty(0, 512), []
        return torch.stack(vecs), keep

    def load_db(self, db_path=None):
        db_path = db_path or self.db_path
        data = torch.load(db_path, map_location="cpu")
        self.db_names = list(data["names"])
        self.db_roles = dict(data.get("roles", {}))
        embs, owner = [], []
        for i, name in enumerate(self.db_names):
            e = data["embs"][name]
            embs.append(e)
            owner += [i] * e.shape[0]
        self.db_embs  = torch.cat(embs, 0) if embs else None
        self.db_owner = torch.tensor(owner) if owner else None

    def save_db(self, db_path=None):
        db_path = db_path or self.db_path
        embs = {name: self.db_embs[self.db_owner == i]
                for i, name in enumerate(self.db_names)}
        torch.save({"names": self.db_names, "embs": embs, "roles": self.db_roles}, db_path)

    def add_embedding(self, name, vec, role: str = "kỹ sư"):
        """Thêm 1 embedding (đã normalize) cho 'name' vào DB in-memory.
        Nếu 'name' chưa có → tạo NGƯỜI MỚI (không cần train)."""
        if name not in self.db_names:
            self.db_names.append(name)
            self.db_roles[name] = role
        idx = self.db_names.index(name)
        vec = vec.view(1, 512)
        if self.db_embs is None:
            self.db_embs, self.db_owner = vec, torch.tensor([idx])
        else:
            self.db_embs  = torch.cat([self.db_embs, vec], 0)
            self.db_owner = torch.cat([self.db_owner, torch.tensor([idx])])

    def remove_person(self, name):
        """Xoá TOÀN BỘ embedding của 'name' khỏi DB (không cần train). True nếu có xoá."""
        if name not in self.db_names or self.db_owner is None:
            return False
        idx = self.db_names.index(name)
        keep = self.db_owner != idx
        self.db_embs  = self.db_embs[keep]
        remaining     = self.db_owner[keep]
        remaining     = remaining - (remaining > idx).long()   # dồn chỉ số sau idx xuống 1
        self.db_owner = remaining
        self.db_names.pop(idx)
        if self.db_embs.shape[0] == 0:
            self.db_embs, self.db_owner = None, None
        return True

    def people_summary(self):
        """List {name, role, count} theo db_names — để web hiện danh sách người + số mẫu."""
        return [{"name": n,
                 "role": self.db_roles.get(n, "kỹ sư"),
                 "count": 0 if self.db_owner is None else int((self.db_owner == i).sum())}
                for i, n in enumerate(self.db_names)]

    def _rank(self, vec, topk=5):
        """vec[512] → list (name, score) MỌI người, sắp giảm dần theo score.
        Điểm mỗi người = TRUNG BÌNH top-k cosine với embedding của họ. Dùng top-k thay vì
        max để 1 embedding lạc không đủ lật kết quả → giảm nhầm tui↔nghia."""
        if self.db_embs is None:
            return []
        sims = self.db_embs @ vec
        ranked = []
        for i, name in enumerate(self.db_names):
            s_i = sims[self.db_owner == i]
            k = min(topk, s_i.numel())
            ranked.append((name, float(s_i.topk(k).values.mean())))
        ranked.sort(key=lambda x: -x[1])
        return ranked

    def _match(self, vec, threshold):
        """vec[512] → (name|None, score). => k đủ gần -> ng lạ."""
        ranked = self._rank(vec)
        if not ranked:
            return None, 0.0
        name, score = ranked[0]
        return (name, score) if score >= threshold else (None, score)

    def recognize(self, img_bgr, threshold=0.28):
        """Trả list {box, det_conf, name|None, best_name, score, runner, stranger} cho mỗi mặt.
        runner = người đứng nhì {name, score} để thấy khoảng cách (độ tự tin)."""
        boxes = self.detect(img_bgr)
        if not boxes:
            return []
        emb, keep = self.embed(img_bgr, boxes)
        out = []
        for j, i in enumerate(keep):
            ranked = self._rank(emb[j])
            best_name, best_score = ranked[0] if ranked else (None, 0.0)
            runner = ({"name": ranked[1][0], "score": round(ranked[1][1], 3)}
                      if len(ranked) > 1 else None)
            stranger = not (best_score >= threshold)
            x1, y1, x2, y2, conf = boxes[i]
            out.append({"box": (x1, y1, x2, y2), "det_conf": conf,
                        "name": None if stranger else best_name,
                        "best_name": best_name, "score": round(best_score, 3),
                        "runner": runner, "stranger": stranger})
        return out

    def embed_largest(self, img_bgr):
        """Detect → embed mặt LỚN NHẤT (enroll từ webcam). Trả vec[512] | None."""
        boxes = self.detect(img_bgr)
        if not boxes:
            return None
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        v = self._embed_crop(img_bgr, boxes[0])
        return v
