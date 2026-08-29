"""
Giả lập PLC Mitsubishi (MC Protocol, khung 3E, binary) để chạy thử luồng
main.py -> plc_light.py -> PLC khi chưa có phần cứng thật.

Mở một TCP server, đọc gói tin pymcprotocol gửi lên khi gọi
batchwrite_bitunits() / batchread_bitunits() / connect(), in log mỗi lệnh ghi
bit (vd "Y0 = 1") và trả về khung phản hồi hợp lệ để pymcprotocol coi là thành
công.

Cách chạy:
    python plc_simulator.py                # lắng nghe 0.0.0.0:5007
    python plc_simulator.py --port 5000     # đổi port cho khớp PLC_PORT

Rồi trỏ PLC_IP="127.0.0.1", PLC_PORT=5007 trong main.py hoặc test_plc_light.py.
"""
import argparse
import socket
import sys
import threading

# Ép stdout/stderr sang UTF-8 để log tiếng Việt không bị lỗi trên console Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Bảng mã vùng nhớ (device code), lấy từ pymcprotocol/mcprotocolconst.py.
DEVICE_CODES = {
    0x9C: "X", 0x9D: "Y", 0x90: "M", 0x92: "L", 0x93: "F", 0x94: "V",
    0xA0: "B", 0xA8: "D", 0xB4: "W", 0xAF: "R", 0xB0: "ZR",
    0x91: "SM", 0xA9: "SD",
}

# Bố cục khung 3E (binary), theo type3e.py:
#   0-1 subheader | 2 network | 3 pc | 4-5 dest_moduleio | 6 dest_modulesta
#   7-8 data length | 9-10 timer | 11.. requestdata (command, subcommand, device...)
HEADER_LEN = 11


def parse_write_bit_command(payload: bytes):
    """payload = requestdata (từ offset 11). Trả (device_name, values), hoặc None
    nếu không phải lệnh ghi bit (lệnh khác vẫn được ACK, chỉ không log chi tiết)."""
    if len(payload) < 4:
        return None
    command = int.from_bytes(payload[0:2], "little")
    subcommand = int.from_bytes(payload[2:4], "little")
    if command != 0x1401:          # 0x1401 = batch write
        return None
    if subcommand not in (0x0001, 0x0003):   # 0x0001 = bit (Q/L/QnA/iQ-L), 0x0003 = bit (iQ-R)
        return None

    is_iqr = subcommand == 0x0003
    idx = 4
    if is_iqr:
        devicenum = int.from_bytes(payload[idx:idx + 4], "little"); idx += 4
        devicecode = int.from_bytes(payload[idx:idx + 2], "little"); idx += 2
    else:
        devicenum = int.from_bytes(payload[idx:idx + 3], "little"); idx += 3
        devicecode = payload[idx]; idx += 1
    write_size = int.from_bytes(payload[idx:idx + 2], "little"); idx += 2
    bit_bytes = payload[idx:]

    values = []
    for i in range(write_size):
        b = bit_bytes[i // 2] if i // 2 < len(bit_bytes) else 0
        bit = (b >> 4) & 1 if i % 2 == 0 else b & 1
        values.append(bit)

    dev_name = DEVICE_CODES.get(devicecode, f"0x{devicecode:02X}")
    device_str = f"{dev_name}{devicenum:X}" if dev_name in ("X", "Y", "B", "W") else f"{dev_name}{devicenum}"
    return device_str, values


def make_ack(status: int = 0) -> bytes:
    """Khung phản hồi 3E tối thiểu cho pymcprotocol. Chỉ cần end code ở offset 9-10;
    pymcprotocol không kiểm tra subheader/network/pc của phản hồi."""
    resp = bytearray(11)
    resp[0:2] = (0xD000).to_bytes(2, "big")   # subheader phản hồi
    resp[2] = 0x00                            # network
    resp[3] = 0xFF                            # pc
    resp[4:6] = (0x03FF).to_bytes(2, "little")  # dest_moduleio
    resp[6] = 0x00                            # dest_modulesta
    resp[7:9] = (2).to_bytes(2, "little")     # độ dài dữ liệu phản hồi
    resp[9:11] = status.to_bytes(2, "little")  # end code: 0 = thành công
    return bytes(resp)


def handle_client(conn: socket.socket, addr):
    print(f"[simulator] PLC giả lập: đã kết nối từ {addr}")
    with conn:
        while True:
            try:
                data = conn.recv(4096)
            except (ConnectionResetError, ConnectionAbortedError):
                break
            if not data:
                break
            if len(data) < HEADER_LEN:
                conn.sendall(make_ack(0))
                continue

            payload = data[HEADER_LEN:]
            parsed = parse_write_bit_command(payload)
            if parsed is not None:
                device_str, values = parsed
                for off, v in enumerate(values):
                    print(f"[simulator] GHI  {device_str}  =  {v}   "
                          f"({'BẬT ĐÈN' if v == 1 else 'TẮT ĐÈN'})")
            else:
                print(f"[simulator] (lệnh khác, {len(payload)} byte payload) -> ACK")

            conn.sendall(make_ack(0))   # luôn trả về thành công (status=0)
    print(f"[simulator] {addr} đã ngắt kết nối")


def main():
    ap = argparse.ArgumentParser(description="Giả lập PLC Mitsubishi qua MC Protocol (khung 3E)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5007)
    args = ap.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(5)
    print(f"[simulator] Đang lắng nghe {args.host}:{args.port} — Ctrl+C để dừng")
    print(f"[simulator] Trỏ PLC_IP=\"127.0.0.1\", PLC_PORT={args.port} trong main.py / test_plc_light.py")
    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\n[simulator] Dừng.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
