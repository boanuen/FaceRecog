"""
Test riêng đèn tháp qua PLC, chạy trước khi tích hợp vào main.py để xác nhận đúng
địa chỉ Y và đấu dây.

Cách dùng:
    python test_plc_light.py                 # bật lần lượt xanh -> vàng -> đỏ, mỗi màu 2s
    python test_plc_light.py green            # chỉ bật xanh 2s rồi tắt
    python test_plc_light.py red --duration 5 # bật đỏ 5s

Trong lúc chạy, mở GX Works ở chế độ Online Monitor để xem bit Y có đổi trạng thái
không. Nếu Y đổi mà đèn không sáng thì lỗi nằm ở phần cứng/đấu dây.
"""
import argparse
import time

from plc_light import PLCLight

# Đổi 4 giá trị này cho khớp main.py / tủ điện thật trước khi chạy.
PLC_IP    = "192.168.1.10"
PLC_PORT  = 5000
PLC_TYPE  = "Q"
PLC_COILS = {"green": "Y0", "yellow": "Y1", "red": "Y2"}


def main():
    parser = argparse.ArgumentParser(description="Test đèn tháp qua PLC")
    parser.add_argument("color", nargs="?", choices=["green", "yellow", "red"],
                        help="Bỏ trống để test lần lượt cả 3 màu")
    parser.add_argument("--duration", type=float, default=2.0, help="Số giây đèn sáng")
    args = parser.parse_args()

    plc = PLCLight(PLC_IP, PLC_PORT, PLC_COILS, plctype=PLC_TYPE, pulse_sec=args.duration)
    time.sleep(0.5)   # đợi thread nền thử kết nối lần đầu

    colors = [args.color] if args.color else ["green", "yellow", "red"]
    for c in colors:
        print(f"--> Bật {c} ({PLC_COILS[c]}) trong {args.duration}s ...")
        plc.signal(c, duration=args.duration)
        time.sleep(args.duration + 0.5)   # đợi xử lý xong trước khi sang màu kế

    print(f"Kết nối PLC: {'OK' if plc.connected else 'thất bại - kiểm tra IP/port/mạng LAN'}")
    plc.close()


if __name__ == "__main__":
    main()
