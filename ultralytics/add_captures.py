"""Gộp ảnh webcam đã chụp (captures/) vào tập train -> tạo data_plus.yaml để train lại.
Zero-copy: chỉ sinh 1 file .txt liệt kê (ảnh train gốc + ảnh webcam mới) + 1 .yaml.
Nhãn tự tìm theo quy ước YOLO (images/ -> labels/), cả 2 nguồn đều có sẵn labels.

Chạy:  python add_captures.py     (sau khi đã chụp bằng nút 📸 trên web)
"""
import os, glob
from collections import Counter

ROOT = r"C:\Users\Admin\detect\ultralytics"
NAMES = ['nghia', 'quan', 'son', 'tri', 'tui']

train_imgs = glob.glob(os.path.join(ROOT, "train", "images", "*.jpg"))
cap_imgs   = glob.glob(os.path.join(ROOT, "captures", "images", "*.jpg"))

if not cap_imgs:
    print("⚠️  Chưa có ảnh nào trong captures/images — hãy chụp bằng nút 📸 trên web trước đã.")
    raise SystemExit

# đếm ảnh webcam theo người (tên file dạng <person>_<timestamp>.jpg)
by_person = Counter()
for p in cap_imgs:
    who = os.path.basename(p).split("_")[0]
    if who in NAMES:
        by_person[who] += 1

all_imgs = train_imgs + cap_imgs
txt = os.path.join(ROOT, "train_plus.txt")
with open(txt, "w", encoding="utf-8") as f:
    f.write("\n".join(all_imgs))

yaml = os.path.join(ROOT, "data_plus.yaml")
with open(yaml, "w", encoding="utf-8") as f:
    f.write(
        "# Train = ảnh gốc + ảnh webcam mới (captures/). val/test giữ nguyên.\n"
        f"train: {txt}\n"
        f"val: {os.path.join(ROOT, 'valid', 'images')}\n"
        f"test: {os.path.join(ROOT, 'test', 'images')}\n\n"
        "nc: 5\n"
        f"names: {NAMES}\n"
    )

print(f"Ảnh train gốc : {len(train_imgs)}")
print(f"Ảnh webcam mới: {len(cap_imgs)}")
for n in NAMES:
    print(f"    {n:<7}: {by_person.get(n, 0)} ảnh webcam")
print(f"Tổng train    : {len(all_imgs)}")
print(f"-> {txt}")
print(f"-> {yaml}")
print("\nTrain lại (trong cmd RIÊNG, tắt Docker):")
print('   sửa train.py:  data="data_plus.yaml"   rồi  python train.py')
