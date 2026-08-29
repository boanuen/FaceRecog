"""
Thử nghiệm: train yolo26s trên dataset đã lọc (bỏ ảnh quan/tri có mặt < 10% khung).
Chạy sau khi train.py xong vì GPU 8GB chỉ chạy được 1 train một lúc.
Khác train.py đúng 2 chỗ: data="data_filtered.yaml" và name="...v6_filtered".
"""
from ultralytics import YOLO
import torch, sys

MODEL_SIZE = "s"
IMGSZ = 512

# Cấu hình y hệt config "s" của train.py, chỉ đổi name để không đè run cũ.
_COMMON = dict(
    optimizer="MuSGD",
    lr0=0.002, lrf=0.01,
    momentum=0.949, weight_decay=0.0005, warmup_epochs=3.0,
    box=7.5, cls=1.5, dfl=1.5, cls_pw=0.0,
    mosaic=0.5, close_mosaic=15,
    mixup=0.0, cutmix=0.0, copy_paste=0.0,
    scale=0.5, translate=0.1, degrees=8.0, shear=0.0,
    fliplr=0.5, flipud=0.0,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, bgr=0.0,
    erasing=0.4,
)
cfg = dict(model="yolo26s.pt", name="yolo26s_face_v6_filtered", batch=8, **_COMMON)

if __name__ == '__main__':
    print(f"YOLO26{MODEL_SIZE.upper()} Face - thử nghiệm (dataset đã lọc)")
    if not torch.cuda.is_available():
        print("Không có GPU!")
        sys.exit(1)

    gpu  = torch.cuda.get_device_properties(0)
    print(f"GPU  : {gpu.name}  ({gpu.total_memory/1e9:.1f} GB VRAM)")
    print(f"Model: YOLO26{MODEL_SIZE.upper()} | batch={cfg['batch']} | imgsz={IMGSZ}")
    print(f"Data : data_filtered.yaml (train bỏ ảnh quan/tri mặt <10%)\n")

    model = YOLO(cfg["model"])

    results = model.train(
        data="data_filtered.yaml",     # khác train.py
        epochs=110,
        patience=50,
        imgsz=IMGSZ,
        batch=cfg["batch"],
        rect=False,
        freeze=5,

        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"], lrf=cfg["lrf"],
        momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],

        box=cfg["box"], cls=cfg["cls"], dfl=cfg["dfl"], cls_pw=cfg["cls_pw"],

        mosaic=cfg["mosaic"], close_mosaic=cfg["close_mosaic"],
        mixup=cfg["mixup"], cutmix=cfg["cutmix"], copy_paste=cfg["copy_paste"],
        scale=cfg["scale"], translate=cfg["translate"],
        degrees=cfg["degrees"], shear=cfg["shear"],
        fliplr=cfg["fliplr"], flipud=cfg["flipud"],
        hsv_h=cfg["hsv_h"], hsv_s=cfg["hsv_s"], hsv_v=cfg["hsv_v"], bgr=cfg["bgr"],
        erasing=cfg["erasing"],

        device=0,
        amp=True,
        cache="disk",
        workers=2,

        project="runs/face_yolo26",
        name=cfg["name"],              # v6_filtered, không đè run cũ
        save_period=10,
        plots=True,
        verbose=True,
    )

    rd = results.results_dict
    print("\n" + "=" * 60)
    print("Xong (thử nghiệm dataset đã lọc).")
    print(f"   Best model : {results.save_dir}/weights/best.pt")
    print(f"   mAP50      : {rd.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   Precision  : {rd.get('metrics/precision(B)', 0):.4f}")
    print(f"   Recall     : {rd.get('metrics/recall(B)', 0):.4f}")

    print("\n   Per-class mAP50 (chú ý quan & tri):")
    for i, cls_name in enumerate(['nghia', 'quan', 'son', 'tri', 'tui']):
        ap = rd.get(f'metrics/mAP50(B){i}')
        if ap is not None:
            print(f"     {cls_name}: {ap:.4f}")

    print("\n   -> So confusion_matrix_normalized.png của run này với run cũ (ô quan / tri).")
