"""
enroll.py — Tạo DB embedding cho 5 người TỪ dữ liệu đã gán nhãn (train + valid).

Với mỗi ảnh, cắt đúng GT box theo nhãn → FaceNet embedding → gom theo người.
KHÔNG cần chụp ảnh mới, KHÔNG train lại. Kết quả: face_db.pt

Sau khi gom xong, script tự tính độ tách giữa (cùng người) và (khác người)
để GỢI Ý ngưỡng cosine hợp lý cho main.py.

"""
import os, glob, random
import cv2, torch
from recognizer import FaceRecognizer, BASE_DIR, DB_PATH

NAMES      = ['nghia', 'quan', 'son', 'tri', 'tui']
PER_PERSON = 60          # số embedding lưu mỗi người (lấy mẫu ngẫu nhiên cho đa dạng góc/ánh sáng)
SOURCES    = [("train/images", "train/labels"), ("valid/images", "valid/labels")]
random.seed(0)


def yolo_to_xyxy(cx, cy, bw, bh, W, H):
    return (int((cx - bw / 2) * W), int((cy - bh / 2) * H),
            int((cx + bw / 2) * W), int((cy + bh / 2) * H))


def gather():
    """dict name -> list (img_path, (cx,cy,bw,bh))."""
    per = {n: [] for n in NAMES}
    for img_sub, lbl_sub in SOURCES:
        img_dir = os.path.join(BASE_DIR, img_sub)
        lbl_dir = os.path.join(BASE_DIR, lbl_sub)
        for lbl in glob.glob(os.path.join(lbl_dir, "*.txt")):
            stem = os.path.splitext(os.path.basename(lbl))[0]
            img_path = next((p for ext in (".jpg", ".jpeg", ".png")
                             if os.path.isfile(p := os.path.join(img_dir, stem + ext))), None)
            if not img_path:
                continue
            with open(lbl) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    if 0 <= cid < len(NAMES):
                        per[NAMES[cid]].append((img_path, tuple(map(float, parts[1:5]))))
    return per


def main():
    print("Khởi tạo engine (YOLO + FaceNet)...")
    rec = FaceRecognizer(db_path=None)   # chưa có DB — ta đang tạo nó

    per = gather()
    print("\nSố mẫu tìm thấy theo người:")
    for n in NAMES:
        print(f"   {n:<7}: {len(per[n])}")

    embs = {}
    for name in NAMES:
        items = per[name]
        if not items:
            print(f"⚠️  {name}: 0 mẫu — bỏ qua")
            continue
        random.shuffle(items)
        vecs = []
        for img_path, (cx, cy, bw, bh) in items:
            if len(vecs) >= PER_PERSON:
                break                       # đã đủ embedding tốt
            img = cv2.imread(img_path)
            if img is None:
                continue
            H, W = img.shape[:2]
            box = yolo_to_xyxy(cx, cy, bw, bh, W, H)
            emb, keep = rec.embed(img, [box])   # ArcFace có thể bỏ qua frame khó
            if len(keep):
                vecs.append(emb[0])
        if vecs:
            embs[name] = torch.stack(vecs)
            print(f"   {name:<7}: {len(vecs)} embedding")

    if not embs:
        print("Không tạo được embedding nào — kiểm tra lại đường dẫn ảnh/nhãn.")
        return

    names = [n for n in NAMES if n in embs]
    torch.save({"names": names, "embs": embs}, DB_PATH)
    print(f"\n✓ Đã lưu DB: {DB_PATH}")

    # ── HIỆU CHỈNH NGƯỠNG ──────────────────────────────────────────
    all_emb = torch.cat([embs[n] for n in names], 0)
    owner   = torch.cat([torch.full((embs[n].shape[0],), i) for i, n in enumerate(names)])
    sims = all_emb @ all_emb.T
    M = sims.shape[0]
    iu = torch.triu_indices(M, M, offset=1)
    same = owner[iu[0]] == owner[iu[1]]
    s = sims[iu[0], iu[1]]
    intra, inter = s[same], s[~same]

    def pct(t, q):
        return torch.quantile(t, q).item()

    print("\n── Độ tách cosine (để chọn ngưỡng) ──")
    print(f"   CÙNG người : trung bình {intra.mean():.3f} | thấp nhất(p5) {pct(intra,0.05):.3f}")
    print(f"   KHÁC người : trung bình {inter.mean():.3f} | cao nhất(p95) {pct(inter,0.95):.3f}")
    thr = round((pct(intra, 0.05) + pct(inter, 0.95)) / 2, 2)
    gap = pct(intra, 0.05) - pct(inter, 0.95)
    print(f"\n   → NGƯỠNG gợi ý: {thr}")
    if gap > 0:
        print(f"     (tách sạch, biên {gap:.3f} — nên hoạt động tốt)")
    else:
        print(f"     (hai phân phối chồng nhau {abs(gap):.3f} — quan/tri có thể còn khó; "
              f"tăng PER_PERSON hoặc nâng lên B2 có align nếu cần)")
    print(f"\n   Đặt vào main.py:  THRESHOLD = {thr}")


if __name__ == "__main__":
    main()
