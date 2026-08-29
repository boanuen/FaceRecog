import os
import numpy as np
import cv2
import torch
import onnxruntime as ort
from ultralytics import YOLO
from insightface.app import FaceAnalysis

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "best.pt")
YOLO_OV   = os.path.join(BASE_DIR, "best_openvino_model")   # YOLO đã export sang OpenVINO (Intel)
DB_PATH   = os.path.join(BASE_DIR, "face_db.pt")

# Cho phép ép chọn thiết bị OpenVINO cho YOLO qua biến môi trường: "GPU" (iGPU) hoặc "CPU".
OV_YOLO_DEVICE = os.environ.get("OV_YOLO_DEVICE", "intel:gpu")

MARGIN = 0.30   # nới quanh box YOLO trước khi đưa vào ArcFace

class FaceRecognizer:
    def __init__(self, yolo_path=YOLO_PATH, db_path=DB_PATH, device=None,
                 det_imgsz=512, det_conf=0.25, det_iou=0.45, use_gpu_arcface=False,
                 arc_det_size=(224, 224)):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.det_imgsz = det_imgsz
        self.det_conf  = det_conf
        self.det_iou   = det_iou
        self.half      = self.device == "cuda"

        avail = ort.get_available_providers()
        self.has_openvino = "OpenVINOExecutionProvider" in avail

        # ── YOLO26 — detector khuôn mặt ──
        self.yolo_device = None
        if os.path.isdir(YOLO_OV):
            self.yolo = YOLO(YOLO_OV, task="detect")
            self.yolo_device = OV_YOLO_DEVICE
            print(f"[recognizer] YOLO: OpenVINO ({OV_YOLO_DEVICE})")
        else:
            self.yolo = YOLO(yolo_path)

        # ArcFace: căn chỉnh và embedding.
        # Chọn provider nhanh nhất sẵn có: CUDA > OpenVINO (Intel) > CPU.
        if use_gpu_arcface and "CUDAExecutionProvider" in avail:
            providers, arc_ctx = ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
        elif self.has_openvino:
            providers, arc_ctx = ["OpenVINOExecutionProvider", "CPUExecutionProvider"], 0
        else:
            providers, arc_ctx = ["CPUExecutionProvider"], -1
        self.arc_providers = providers
        try:
            self.arc = FaceAnalysis(name="buffalo_l", providers=providers,
                                    allowed_modules=["detection", "recognition"])
            self.arc.prepare(ctx_id=arc_ctx, det_size=arc_det_size)
        except Exception as e:      # provider tối ưu lỗi thì lùi về CPU
            print(f"[recognizer] ArcFace provider {providers} lỗi ({e}) -> dùng CPU")
            self.arc_providers = ["CPUExecutionProvider"]
            self.arc = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                                    allowed_modules=["detection", "recognition"])
            self.arc.prepare(ctx_id=-1, det_size=arc_det_size)
        print(f"[recognizer] ArcFace providers: {self.arc_providers}")

        # DB phẳng để so khớp bằng 1 phép nhân ma trận
        self.db_names = []      # ['nghia','quan',...]
        self.db_roles = {}      # {name: 'kỹ sư'|'sinh viên'}
        self.db_embs  = None    # tensor [M,512] (L2-normalize)
        self.db_owner = None    # tensor [M] -> chỉ số vào db_names
        self.db_path  = db_path
        if db_path and os.path.isfile(db_path):
            self.load_db(db_path)

    def detect(self, img_bgr, imgsz=None):
        """Trả list (x1,y1,x2,y2,conf) của mọi mặt."""
        if self.yolo_device is not None:            # YOLO OpenVINO (Intel): CPU/iGPU
            dev = self.yolo_device
        else:
            dev = 0 if self.device == "cuda" else "cpu"
        r = self.yolo(
            img_bgr, conf=self.det_conf, iou=self.det_iou,
            imgsz=imgsz or self.det_imgsz, half=self.half,
            device=dev, end2end=False, verbose=False,
        )[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out

    def is_blurry(self, img_bgr, box, thr=60.0):
        """True nếu vùng mặt mờ (phương sai Laplacian < thr). Dùng để bỏ qua ArcFace trên
        khung nhoè do di chuyển vì ảnh mờ cho embedding kém tin cậy. Resize 100x100 trước
        khi đo để ngưỡng không phụ thuộc mặt gần/xa camera."""
        H, W = img_bgr.shape[:2]
        x1, y1, x2, y2 = box[:4]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return True
        gray = cv2.cvtColor(cv2.resize(crop, (100, 100)), cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < thr

    def _embed_crop(self, img_bgr, box):
        """Cắt vùng box (có margin), ArcFace tự căn 5 landmark rồi embedding. Trả vec[512]."""
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
        # crop chỉ nên có 1 mặt, nếu nhiều thì lấy mặt to nhất
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return torch.tensor(f.normed_embedding, dtype=torch.float32)

    def embed(self, img_bgr, boxes):
        """boxes -> (emb [K,512] đã normalize, keep: index các box embed thành công)."""
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
        """Thêm một embedding (đã normalize) cho 'name'. Nếu chưa có thì tạo người mới."""
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
        """Xoá toàn bộ embedding của 'name'. True nếu có xoá."""
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

    def remove_embedding(self, name, local_index):
        """Xoá một mẫu cụ thể (embedding thứ local_index, 0-based) của 'name'. Dùng khi
        enroll nhầm một ảnh. True nếu xoá được. Hết mẫu thì xoá luôn người."""
        if name not in self.db_names or self.db_embs is None:
            return False
        idx = self.db_names.index(name)
        rows = (self.db_owner == idx).nonzero(as_tuple=True)[0]
        if not (0 <= local_index < len(rows)):
            return False
        keep = torch.ones(self.db_embs.shape[0], dtype=torch.bool)
        keep[int(rows[local_index])] = False
        self.db_embs, self.db_owner = self.db_embs[keep], self.db_owner[keep]
        if int((self.db_owner == idx).sum()) == 0:
            self.remove_person(name)
        return True

    def sample_diagnostics(self, name):
        """Với mỗi mẫu của 'name': điểm tự-khớp (cosine trung bình với các mẫu còn lại của
        chính người đó) và người giống nhất trong số người khác cùng điểm. Mẫu enroll nhầm
        thường tự-khớp thấp và khớp cao bất thường với người bị chụp nhầm. Trả list
        {index, self_sim, closest_other, closest_score}, sắp theo self_sim tăng dần.
        index dùng trực tiếp cho remove_embedding()."""
        if name not in self.db_names or self.db_embs is None:
            return []
        idx = self.db_names.index(name)
        mask = self.db_owner == idx
        rows = mask.nonzero(as_tuple=True)[0]
        if len(rows) == 0:
            return []
        own = self.db_embs[rows]
        other_mask  = ~mask
        other_embs  = self.db_embs[other_mask]
        other_owner = self.db_owner[other_mask]
        out = []
        for local_i in range(own.shape[0]):
            if own.shape[0] > 1:
                rest = torch.cat([own[:local_i], own[local_i + 1:]], 0)
                self_sim = float((rest @ own[local_i]).mean())
            else:
                self_sim = 1.0   # mẫu duy nhất, không có gì để so sánh
            closest_other, closest_score = None, None
            if other_embs.shape[0] > 0:
                sims = other_embs @ own[local_i]
                j = int(sims.argmax())
                closest_other, closest_score = self.db_names[int(other_owner[j])], round(float(sims[j]), 3)
            out.append({"index": local_i, "self_sim": round(self_sim, 3),
                        "closest_other": closest_other, "closest_score": closest_score})
        out.sort(key=lambda d: d["self_sim"])
        return out

    def people_summary(self):
        """List {name, role, count} theo db_names, để web hiện danh sách người và số mẫu."""
        return [{"name": n,
                 "role": self.db_roles.get(n, "kỹ sư"),
                 "count": 0 if self.db_owner is None else int((self.db_owner == i).sum())}
                for i, n in enumerate(self.db_names)]

    def _rank(self, vec, topk=5):
        """vec[512] -> list (name, score) mọi người, sắp giảm dần theo score. Điểm mỗi
        người = trung bình top-k cosine với embedding của họ, để một embedding lạc không
        đủ lật kết quả."""
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
        """vec[512] -> (name|None, score). Không đủ gần thì trả None (người lạ)."""
        ranked = self._rank(vec)
        if not ranked:
            return None, 0.0
        name, score = ranked[0]
        return (name, score) if score >= threshold else (None, score)

    def recognize(self, img_bgr, threshold=0.28):
        """Trả list {box, det_conf, name|None, best_name, score, runner, stranger} cho mỗi
        mặt. runner = người đứng nhì {name, score} để thấy khoảng cách điểm."""
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
        """Detect rồi embed mặt lớn nhất (dùng khi enroll từ webcam). Trả vec[512] hoặc None."""
        boxes = self.detect(img_bgr)
        if not boxes:
            return None
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        v = self._embed_crop(img_bgr, boxes[0])
        return v
