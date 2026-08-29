"""
Demo camera thật + nhận diện thật (cùng model YOLO + ArcFace với main.py): khi
nhận ra người (hoặc người lạ) thì gửi tín hiệu đèn qua plc_simulator. Dùng để
trình diễn toàn bộ luồng nhận diện + logic PLC khi chưa lắp đèn tháp thật.

Script không dùng cv2.imshow() (nhiều máy cài lẫn opencv-python và
opencv-python-headless nên mất GUI). Thay vào đó, kết quả in ra terminal và
tuỳ chọn lưu ảnh chụp có vẽ khung/tên ra file.

Cách dùng:
    1) Terminal 1, chạy simulator trước:
         python plc_simulator.py --port 5007

    2) Terminal 2, chạy demo (webcam mặc định là 0):
         python demo_camera_to_plc.py
       hoặc chỉ rõ webcam / vai trò / IP-port simulator:
         python demo_camera_to_plc.py --cam 0 --role vao --plc-port 5007
       lưu ảnh chụp liên tục ra file để xem overlay:
         python demo_camera_to_plc.py --snapshot demo_snapshot.jpg

    3) Đưa mặt vào webcam. Khi một danh tính ổn định qua N frame liên tiếp,
       script gọi plc.signal(...) và simulator in ra "GHI Y0 = 1 (BẬT ĐÈN)".

    Nhấn Ctrl+C để dừng.
"""
import argparse
import time
from collections import deque, Counter

import cv2

from recognizer import FaceRecognizer
from plc_light import PLCLight

CONFIRM_FRAMES = 5      # số frame liên tiếp phải cùng một danh tính mới coi là chắc
COOLDOWN_SEC = 3.0      # sau khi báo 1 lần, chờ ít nhất bấy nhiêu giây mới báo lại
PRINT_EVERY_SEC = 0.5   # giãn cách in trạng thái frame ra terminal


# Chữ vẽ đè lên video dùng tiếng Anh vì cv2.putText không hiển thị được tiếng Việt có dấu.
def draw_overlay(frame, results, status_text):
    for r in results:
        x1, y1, x2, y2 = r["box"]
        if r["stranger"]:
            label, color = "Stranger", (0, 0, 255)
        else:
            label, color = f"{r['name']} ({r['score']:.2f})", (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, status_text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)


def main():
    ap = argparse.ArgumentParser(description="Demo camera thật + nhận diện thật -> PLC giả lập")
    ap.add_argument("--cam", type=int, default=0, help="Chỉ số webcam (cv2.VideoCapture)")
    ap.add_argument("--role", choices=["vao", "ra"], default="vao",
                     help="Camera đóng vai cổng vào (xanh) hay cổng ra (vàng)")
    ap.add_argument("--threshold", type=float, default=0.28, help="Ngưỡng cosine người quen/lạ")
    ap.add_argument("--plc-ip", default="127.0.0.1")
    ap.add_argument("--plc-port", type=int, default=5007)
    ap.add_argument("--plc-type", default="Q", choices=["Q", "iQ-R"])
    ap.add_argument("--snapshot", default=None,
                     help="Đường dẫn file .jpg để liên tục lưu ảnh có vẽ khung/tên. Vd: --snapshot snap.jpg")
    ap.add_argument("--show", action="store_true",
                     help="Thử mở cửa sổ video bằng cv2.imshow (chỉ khi máy chắc chắn có GUI opencv)")
    args = ap.parse_args()

    print("[demo] Đang khởi tạo YOLO + ArcFace (lần đầu có thể mất vài giây)...")
    rec = FaceRecognizer(det_conf=0.25)
    print(f"[demo] DB hiện có {len(rec.db_names)} người: {rec.db_names}")

    plc = PLCLight(args.plc_ip, args.plc_port,
                    {"green": "Y0", "yellow": "Y1", "red": "Y2"},
                    plctype=args.plc_type, pulse_sec=2.5)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[demo] Không mở được webcam index {args.cam}")
        return

    gui_ok = args.show   # nếu imshow lỗi ngay lần đầu thì tự tắt cho các frame sau
    hist = deque(maxlen=CONFIRM_FRAMES)
    last_signalled = None
    last_signal_time = 0.0
    last_print_time = 0.0

    print("[demo] Đang chạy. Nhấn Ctrl+C để dừng.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[demo] Không đọc được frame từ webcam.")
                break

            results = rec.recognize(frame, threshold=args.threshold)

            frame_label = None
            for r in results:
                if r["stranger"]:
                    frame_label = frame_label or "LA"
                else:
                    frame_label = frame_label or r["name"]

            hist.append(frame_label)   # None nếu frame này không thấy mặt nào

            # Ổn định: CONFIRM_FRAMES frame liên tiếp cùng một giá trị (bỏ qua None)
            votes = Counter(v for v in hist if v is not None)
            stable = None
            if len(hist) == CONFIRM_FRAMES and votes:
                top, count = votes.most_common(1)[0]
                if count == CONFIRM_FRAMES:
                    stable = top

            now = time.time()
            if stable is not None and (stable != last_signalled or now - last_signal_time > COOLDOWN_SEC):
                if stable == "LA":
                    plc.signal("red")
                    print("[demo] Xác nhận: người lạ -> gửi PLC 'red'")
                else:
                    color = "green" if args.role == "vao" else "yellow"
                    plc.signal(color)
                    print(f"[demo] Xác nhận: {stable} -> gửi PLC '{color}'")
                last_signalled = stable
                last_signal_time = now

            status = f"PLC {'online' if plc.connected else 'offline'} | role={args.role}"

            # In trạng thái từng frame (đã giãn cách) để thấy hệ thống đang chạy
            if now - last_print_time > PRINT_EVERY_SEC:
                if results:
                    seen = ", ".join(
                        ("Người lạ" if r["stranger"] else f"{r['name']}({r['score']:.2f})")
                        for r in results
                    )
                else:
                    seen = "(không thấy mặt)"
                print(f"[demo] đang thấy: {seen}   |   {status}")
                last_print_time = now

            if args.snapshot:
                draw_overlay(frame, results, status)
                cv2.imwrite(args.snapshot, frame)

            if gui_ok:
                try:
                    draw_overlay(frame, results, status)
                    cv2.imshow("Demo (press Ctrl+C to quit)", frame)
                    cv2.waitKey(1)
                except cv2.error as e:
                    print(f"[demo] cv2.imshow lỗi ({e}) — opencv không có GUI. "
                          f"Tắt cửa sổ, tiếp tục chạy không GUI.")
                    gui_ok = False
    except KeyboardInterrupt:
        print("\n[demo] Đã nhận Ctrl+C, đang dừng...")
    finally:
        cap.release()
        if gui_ok:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        plc.close()
        print("[demo] Đã đóng.")


if __name__ == "__main__":
    main()
