"""
Train YOLO26 detect 5 lớp khuôn mặt. Tinh chỉnh loss gain và augmentation để giảm
nhầm giữa quan và tri (2 người có nét mặt gần giống nhau).
"""
from ultralytics import YOLO
import torch, sys

MODEL_SIZE = "m"   # "m" phân biệt 2 mặt giống nhau tốt hơn "s"
IMGSZ = 512        # ảnh gốc ~480px, 512 là đủ

# Mọi tham số train lấy từ đây. Hyperparams giống nhau cho m/s, chỉ khác model và batch.
_COMMON = dict(
    # optimizer
    optimizer="MuSGD",
    lr0=0.002, lrf=0.01,              # lrf=0.01 cho cosine decay về gần 0
    momentum=0.949, weight_decay=0.0005, warmup_epochs=3.0,

    # loss gains
    box=7.5,                          # giảm từ 9.83 để nhánh cls có trọng số tương đối cao hơn
    cls=1.5,                          # tăng từ mặc định 0.5 để phân biệt quan/tri
    dfl=1.5,
    cls_pw=0.0,                       # tắt inverse-freq weighting (đang vô tình hạ trọng số 'tri')

    # augmentation
    # HSV mạnh để model bớt phụ thuộc màu/ánh sáng của ảnh điện thoại khi chạy trên webcam.
    hsv_h=0.02, hsv_s=0.7, hsv_v=0.5, bgr=0.0,
    mosaic=0.5, close_mosaic=15,
    mixup=0.1, cutmix=0.0, copy_paste=0.0,   # mixup nhẹ để khái quát hoá tốt hơn
    scale=0.5, translate=0.1, degrees=8.0, shear=0.0,
    fliplr=0.5, flipud=0.0,           # mặt gần đối xứng nên lật ngang được, không lật dọc
    erasing=0.4,                      # random erasing: bền hơn với che khuất/góc nghiêng
)

CONFIGS = {
    "m": dict(model="yolo26m.pt", name="yolo26m_face_v6", batch=4,  **_COMMON),   # RTX 3050 4GB @ 512
    "s": dict(model="yolo26s.pt", name="yolo26s_face_v6", batch=8,  **_COMMON),
}

if __name__ == '__main__':

    cfg = CONFIGS[MODEL_SIZE]

    print(f"YOLO26{MODEL_SIZE.upper()} Face Recognition")
    if not torch.cuda.is_available():
        print("Không có GPU!")
        sys.exit(1)

    gpu  = torch.cuda.get_device_properties(0)
    vram = gpu.total_memory / 1e9
    print(f"GPU  : {gpu.name}  ({vram:.1f} GB VRAM)")
    print(f"Model: YOLO26{MODEL_SIZE.upper()} | batch={cfg['batch']} | imgsz={IMGSZ}")
    print()

    model = YOLO(cfg["model"])

    results = model.train(
        data="data.yaml",
        epochs=120,
        patience=50,
        imgsz=IMGSZ,
        batch=cfg["batch"],
        rect=False,
        freeze=5,               

        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        lrf=cfg["lrf"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],

        box=cfg["box"],
        cls=cfg["cls"],       
        dfl=cfg["dfl"],
        cls_pw=cfg["cls_pw"],

        mosaic=cfg["mosaic"],
        close_mosaic=cfg["close_mosaic"],
        mixup=cfg["mixup"],
        cutmix=cfg["cutmix"],
        copy_paste=cfg["copy_paste"],
        scale=cfg["scale"],
        translate=cfg["translate"],
        degrees=cfg["degrees"],
        shear=cfg["shear"],
        fliplr=cfg["fliplr"],
        flipud=cfg["flipud"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        bgr=cfg["bgr"],
        erasing=cfg["erasing"],

        device=0,
        amp=True,
        cache="disk",
        workers=2,

        project="runs/face_yolo26",
        name=cfg["name"],
        save_period=10,
        plots=True,
        verbose=True,
    )

    rd = results.results_dict
    print("\n" + "=" * 60)
    print("xong!")
    print(f"   Best model : {results.save_dir}/weights/best.pt")
    print(f"   mAP50      : {rd.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   Precision  : {rd.get('metrics/precision(B)', 0):.4f}")
    print(f"   Recall     : {rd.get('metrics/recall(B)', 0):.4f}")

    print("\n   Per-class mAP50 (so với v2 test):")
    v2_test = {'nghia': 0.935, 'quan': 0.891, 'son': 0.995, 'tri': 0.995, 'tui': 0.995}
    for i, cls_name in enumerate(['nghia', 'quan', 'son', 'tri', 'tui']):
        ap = rd.get(f'metrics/mAP50(B){i}')
        if ap is not None:
            diff = ap - v2_test[cls_name]
            arrow = "↑" if diff > 0 else "↓"
            print(f"     {cls_name}: {ap:.4f}  (v2={v2_test[cls_name]:.3f} {arrow}{abs(diff):.3f})")

    print("\nTest set:")
    best = YOLO(f"{results.save_dir}/weights/best.pt")
    tr = best.val(data="data.yaml", split="test", conf=0.5,
                  iou=0.45, imgsz=IMGSZ, plots=True, workers=0)
    rd_t = tr.results_dict
    print(f"   mAP50    : {rd_t.get('metrics/mAP50(B)', 0):.4f}  (v2: 0.9623)")
    print(f"   mAP50-95 : {rd_t.get('metrics/mAP50-95(B)', 0):.4f}  (v2: 0.7661)")

    bm = f"{results.save_dir}/weights/best.pt"
    print(f"\nExport:")
    print(f"   TensorRT : yolo export model={bm} format=engine imgsz={IMGSZ} half=True")
    print(f"   ONNX     : yolo export model={bm} format=onnx   imgsz={IMGSZ} opset=12")