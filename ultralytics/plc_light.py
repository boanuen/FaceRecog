"""
Giao tiếp với PLC Mitsubishi (Q series / iQ-R) qua MC Protocol (Ethernet, khung 3E)
để điều khiển đèn tháp báo hiệu (xanh / vàng / đỏ) — dùng thư viện pymcprotocol.

Ý nghĩa 3 màu (gọi từ main.py, xem _log_scan()):
    xanh (green)  = nhận diện thành công lúc VÀO   (cam1, status OK)
    vàng (yellow) = nhận diện thành công lúc RA/CHECK-OUT (cam2, status OK)
    đỏ   (red)    = nhận diện FAIL hoặc NGƯỜI LẠ    (status FAIL/LẠ, cả 2 cam)

Cách hoạt động:
- 1 thread nền giữ kết nối TCP tới PLC, xử lý lệnh bật đèn TUẦN TỰ qua Queue —
  main.py gọi signal() chỉ đẩy lệnh vào hàng đợi rồi trả về NGAY (không block
  vòng lặp nhận diện/FastAPI).
- Mỗi lệnh: tắt các màu khác (tránh 2 đèn cùng sáng) → bật màu cần báo → giữ
  `pulse_sec` giây → tắt lại.
- Mất kết nối / lỗi ghi: log cảnh báo, KHÔNG làm crash server nhận diện — hệ
  thống chấm công vẫn chạy dù đèn báo tạm thời không hoạt động; lần signal()
  kế tiếp sẽ tự thử kết nối lại.

Cần cài: pip install pymcprotocol
"""
import queue
import threading
import time

import pymcprotocol


class PLCLight:
    def __init__(self, ip: str, port: int, coils: dict, plctype: str = "Q", pulse_sec: float = 1.5):
        """
        ip, port  : địa chỉ Ethernet module của PLC (QJ71E71 / cổng Ethernet built-in iQ-R),
                    port là port MC Protocol đã cấu hình trên module (không phải port PLC lập trình).
        coils     : map màu -> địa chỉ ngõ ra Y,  vd {"green": "Y0", "yellow": "Y1", "red": "Y2"}
        plctype   : "Q" cho dòng Q, "iQ-R" cho dòng iQ-R (đổi theo PLC thật).
        pulse_sec : thời gian đèn sáng mỗi lần báo (giây).
        """
        self.ip, self.port, self.coils = ip, port, coils
        self.plctype = plctype
        self.pulse_sec = pulse_sec
        self._q: "queue.Queue" = queue.Queue()
        self._pymc = None
        self._connected = False
        self._stop = False
        threading.Thread(target=self._worker, daemon=True, name="plc-light").start()

    # ── kết nối (tự thử lại mỗi khi có lệnh mới, không giữ worker chờ quá lâu) ──
    def _ensure_connected(self) -> bool:
        if self._connected:
            return True
        try:
            pymc = pymcprotocol.Type3E(plctype=self.plctype)
            pymc.connect(self.ip, self.port)
            self._pymc = pymc
            self._connected = True
            print(f"[plc_light] Đã kết nối PLC {self.ip}:{self.port} ({self.plctype})")
        except Exception as e:
            print(f"[plc_light] Không kết nối được PLC {self.ip}:{self.port} — {e}")
            self._connected = False
        return self._connected

    def _write_coil(self, device: str, value: int):
        if not self._ensure_connected():
            return
        try:
            self._pymc.batchwrite_bitunits(headdevice=device, values=[value])
        except Exception as e:
            print(f"[plc_light] Lỗi ghi {device}={value}: {e}")
            self._connected = False   # buộc reconnect ở lệnh kế tiếp
            try:
                self._pymc.close()
            except Exception:
                pass

    def _worker(self):
        while not self._stop:
            color, duration = self._q.get()
            if color is None:   # sentinel để dừng thread khi close()
                break
            device = self.coils.get(color)
            if device is None:
                print(f"[plc_light] Màu '{color}' chưa cấu hình trong PLC_COILS")
                continue
            for c, d in self.coils.items():   # tắt màu khác trước, tránh 2 đèn cùng sáng
                if c != color:
                    self._write_coil(d, 0)
            self._write_coil(device, 1)
            time.sleep(duration)
            self._write_coil(device, 0)

    # ── API gọi từ main.py: fire-and-forget, KHÔNG block request ──
    def signal(self, color: str, duration: float = None):
        self._q.put((color, duration if duration is not None else self.pulse_sec))

    @property
    def connected(self) -> bool:
        return self._connected

    def close(self):
        self._stop = True
        self._q.put((None, 0))
        if self._pymc is not None:
            try:
                self._pymc.close()
            except Exception:
                pass
