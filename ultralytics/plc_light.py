"""
Điều khiển đèn tháp (xanh/vàng/đỏ) trên PLC Mitsubishi Q/iQ-R qua MC Protocol
(Ethernet, khung 3E), dùng thư viện pymcprotocol.

Ý nghĩa 3 màu (xem _log_scan trong main.py):
    xanh  = nhận diện OK (check-in / check-out thành công)
    vàng  = nhận diện FAIL (người quen bị nhận thành người lạ)
    đỏ    = người lạ

Một thread nền giữ kết nối tới PLC và xử lý lệnh bật đèn lần lượt qua Queue.
main.py chỉ gọi signal() để đẩy lệnh vào hàng đợi rồi trả về ngay, không chặn
vòng nhận diện. Mỗi lệnh: bật màu cần báo, tắt các màu khác, giữ pulse_sec giây
rồi tắt. Khi mất kết nối, hệ thống chấm công vẫn chạy và lần signal() sau sẽ tự
kết nối lại.

Cài đặt: pip install pymcprotocol
"""
import queue
import sys
import threading
import time

import pymcprotocol

# Console Windows mặc định không in được tiếng Việt có dấu. Ép stdout/stderr sang
# UTF-8 để print() trong thread nền không ném UnicodeEncodeError làm dừng thread.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _safe_print(msg: str):
    """In an toàn, không để lỗi encoding console làm dừng thread nền."""
    try:
        print(msg)
    except Exception:
        try:
            print(msg.encode("ascii", errors="replace").decode("ascii"))
        except Exception:
            pass


class PLCLight:
    def __init__(self, ip: str, port: int, coils: dict, plctype: str = "Q", pulse_sec: float = 2.0):
        """
        ip, port  : địa chỉ IP và port MC Protocol của module Ethernet trên PLC.
        coils     : map màu -> địa chỉ ngõ ra Y, vd {"green": "Y0", "yellow": "Y1", "red": "Y2"}.
        plctype   : "Q" hoặc "iQ-R" tuỳ dòng PLC.
        pulse_sec : thời gian đèn sáng mỗi lần báo (giây).
        """
        self.ip, self.port, self.coils = ip, port, coils
        self.plctype = plctype
        self.pulse_sec = pulse_sec
        self._q: "queue.Queue" = queue.Queue()
        self._pymc = None
        self._connected = False
        self._stop = False
        self._current = None   # màu đèn đang sáng: "green"/"yellow"/"red" hoặc None
        threading.Thread(target=self._worker, daemon=True, name="plc-light").start()

    # Kết nối tới PLC, tự thử lại mỗi khi có lệnh mới.
    def _ensure_connected(self) -> bool:
        if self._connected:
            return True
        try:
            pymc = pymcprotocol.Type3E(plctype=self.plctype)
            pymc.connect(self.ip, self.port)
            self._pymc = pymc
            self._connected = True
            _safe_print(f"[plc_light] Đã kết nối PLC {self.ip}:{self.port} ({self.plctype})")
        except Exception as e:
            _safe_print(f"[plc_light] Không kết nối được PLC {self.ip}:{self.port} — {e}")
            self._connected = False
        return self._connected

    def _write_coil(self, device: str, value: int):
        if not self._ensure_connected():
            return
        try:
            self._pymc.batchwrite_bitunits(headdevice=device, values=[value])
        except Exception as e:
            _safe_print(f"[plc_light] Lỗi ghi {device}={value}: {e}")
            self._connected = False   # buộc kết nối lại ở lệnh sau
            try:
                self._pymc.close()
            except Exception:
                pass

    def _worker(self):
        while not self._stop:
            color, duration = self._q.get()
            if color is None:   # tín hiệu dừng thread khi close()
                break
            device = self.coils.get(color)
            if device is None:
                _safe_print(f"[plc_light] Màu '{color}' chưa cấu hình trong PLC_COILS")
                continue
            # Bật màu cần báo trước rồi mới tắt các màu khác, để giảm độ trễ cảm nhận.
            self._write_coil(device, 1)
            self._current = color   # web UI đọc qua /plc-status
            for c, d in self.coils.items():
                if c != color:
                    self._write_coil(d, 0)
            time.sleep(duration)
            self._write_coil(device, 0)
            self._current = None

    # main.py gọi hàm này: chỉ đẩy lệnh vào hàng đợi rồi trả về ngay.
    def signal(self, color: str, duration: float = None):
        self._q.put((color, duration if duration is not None else self.pulse_sec))

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def current(self):
        """Màu đèn đang sáng ("green"/"yellow"/"red") hoặc None nếu đèn tắt."""
        return self._current

    def close(self):
        self._stop = True
        self._q.put((None, 0))
        if self._pymc is not None:
            try:
                self._pymc.close()
            except Exception:
                pass
