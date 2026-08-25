# Giao tiếp PC ↔ PLC Mitsubishi qua Ethernet (MC Protocol) — Kiến thức nền tảng

Tài liệu này giải thích từ gốc: PLC là gì, tại sao cần giao tiếp với nó, các cách giao tiếp phổ biến, và đi sâu vào MC Protocol — giao thức bạn sẽ dùng để cho hệ thống nhận diện khuôn mặt điều khiển đèn tháp qua PLC Mitsubishi Q/iQ-R. Không có code ở đây, chỉ có khái niệm và cách tư duy.

## 1. PLC là gì và vì sao nó "khó nói chuyện" hơn một phần mềm bình thường

PLC (Programmable Logic Controller) là một máy tính công nghiệp chuyên dụng, chạy một chương trình logic cố định (thường viết bằng ladder diagram) theo chu kỳ quét (scan cycle) lặp đi lặp lại hàng chục đến hàng trăm lần mỗi giây: đọc toàn bộ ngõ vào → chạy logic → ghi toàn bộ ngõ ra → lặp lại. Nó không giống một chương trình Python hay web server vốn "ngồi chờ" request rồi xử lý — PLC luôn bận chạy vòng lặp của chính nó, không có khái niệm dừng lại để chờ ai đó gọi vào.

Vì vậy khi bạn muốn một máy tính bên ngoài (như PC chạy `main.py`) "nói chuyện" được với PLC, không có chuyện PLC chủ động mở kết nối ra ngoài hay gọi API như web thông thường. Thay vào đó, nhà sản xuất PLC (ở đây là Mitsubishi) thiết kế sẵn một cơ chế: PLC (thực ra là module Ethernet gắn trên nó) đóng vai trò server thụ động, luôn lắng nghe ở một cổng mạng cố định, sẵn sàng nhận các gói tin theo đúng định dạng mà Mitsubishi quy định. Bất kỳ ai gửi đúng định dạng gói tin đó tới đúng địa chỉ, PLC sẽ đọc/ghi dữ liệu tương ứng trong bộ nhớ của nó — không hiểu Python là gì, không quan tâm ai gửi, chỉ đơn thuần diễn giải gói tin theo giao thức đã định sẵn.

Đây chính là bản chất của việc "giao tiếp với PLC": không phải PLC học hiểu chương trình của bạn, mà chương trình của bạn phải nói đúng "ngôn ngữ" mà PLC đã được lập trình sẵn để hiểu.

## 2. Các phương thức giao tiếp PC ↔ PLC phổ biến

Tùy hãng PLC mà có giao thức riêng, nhưng đều xoay quanh 2 lớp vật lý: **nối tiếp** (RS232/RS485, tốc độ chậm, dùng cho khoảng cách xa hoặc PLC đời cũ) và **Ethernet** (TCP/IP, tốc độ cao, phổ biến trên PLC hiện đại — trường hợp của bạn).

| Hãng PLC | Giao thức thường dùng qua Ethernet |
|---|---|
| Mitsubishi | MC Protocol (MELSEC Communication Protocol) |
| Siemens | S7comm |
| Omron | FINS |
| Đa số hãng (chuẩn mở) | Modbus TCP |

Ngoài ra còn có các chuẩn mở không phụ thuộc hãng như **Modbus** (đơn giản, phổ biến nhất trong công nghiệp) và **OPC UA** (hiện đại hơn, bảo mật tốt hơn, dùng nhiều trong nhà máy thông minh/IoT công nghiệp). Nhiều PLC Mitsubishi đời mới cũng hỗ trợ Modbus TCP song song với MC Protocol — nhưng vì bạn dùng dòng Q/iQ-R với module Ethernet gốc của Mitsubishi, MC Protocol là lựa chọn tự nhiên nhất: được hỗ trợ đầy đủ, tài liệu chính hãng rõ ràng, và không cần cấu hình thêm gì trên PLC ngoài việc bật module Ethernet.

## 3. Lớp mạng bên dưới: Ethernet và TCP/IP hoạt động thế nào trong trường hợp này

Trước khi nói tới MC Protocol, cần hiểu tầng bên dưới nó là gì, vì mọi lỗi kết nối thực tế đều nằm ở tầng này chứ không phải ở giao thức.

Module Ethernet trên PLC (ví dụ QJ71E71 gắn rời, hoặc cổng Ethernet tích hợp sẵn trên CPU iQ-R) được gán một địa chỉ IP tĩnh, cấu hình từ trước bằng phần mềm lập trình PLC (GX Works2/GX Works3), y hệt như bạn gán IP tĩnh cho một máy tính. PC chạy `main.py` và PLC phải nằm chung một mạng LAN (cùng dải subnet, ví dụ cả hai đều 192.168.1.x) hoặc có route mạng hợp lệ tới nhau — nếu không, gói tin sẽ không bao giờ tới nơi, giống như gửi thư mà sai địa chỉ nhà.

Trên nền đó, PLC mở một cổng TCP cố định (port) chuyên để nhận lệnh MC Protocol — cổng này **khác** với cổng bạn dùng khi nạp chương trình ladder từ GX Works. Đây là một "cổng dịch vụ" riêng, do người cấu hình PLC chọn số hiệu (thường gặp 5000, 5001... nhưng không cố định, tùy tủ điện đã cấu hình sẵn). PC mở một kết nối TCP tới `IP:port` đó — giống hệt việc trình duyệt mở kết nối tới một web server ở `IP:80`, chỉ khác là nội dung gói tin không phải HTTP mà là định dạng nhị phân riêng của Mitsubishi.

Vì là kết nối TCP thông thường, mọi nguyên tắc mạng cơ bản đều áp dụng: nếu có tường lửa (firewall) trên PC hoặc trên switch chặn port đó, kết nối sẽ thất bại; nếu PLC và PC khác dải mạng mà không có router định tuyến, cũng thất bại; nếu cáp mạng lỏng hoặc switch lỗi, cũng thất bại. Đây là nhóm nguyên nhân chiếm phần lớn lỗi "không kết nối được PLC" trong thực tế — trước khi nghi ngờ giao thức, luôn kiểm tra tầng mạng vật lý trước (ping thử IP của PLC từ PC là bước đầu tiên nên làm).

## 4. Bộ nhớ PLC: các vùng địa chỉ và ý nghĩa (Y, X, M, D...)

PLC tổ chức dữ liệu của nó thành nhiều "vùng nhớ" (device), mỗi vùng có một chữ cái ký hiệu và một dải số. Đây là phần quan trọng nhất cần hiểu vì Y0/Y1/Y2 mà bạn dùng cho đèn tháp chỉ là một trong nhiều loại vùng nhớ.

| Ký hiệu | Tên gọi | Ý nghĩa | Đọc/ghi từ bên ngoài |
|---|---|---|---|
| **X** | Input (ngõ vào) | Trạng thái vật lý của cảm biến, nút nhấn nối vào PLC | Chỉ đọc — không ghi được vì nó phản ánh phần cứng thật |
| **Y** | Output (ngõ ra) | Trạng thái vật lý điều khiển thiết bị ngoài (đèn, relay, van...) | Đọc/ghi được — ghi Y tức là tác động thẳng lên thiết bị nối vào chân đó |
| **M** | Internal relay (rơ-le nội) | Biến trung gian dùng trong logic ladder, không nối ra phần cứng | Đọc/ghi được — an toàn để bên ngoài ghi vào vì không đụng trực tiếp phần cứng |
| **D** | Data register (thanh ghi dữ liệu) | Lưu số nguyên/thực (16-bit hoặc 32-bit), dùng cho giá trị như nhiệt độ, đếm số lượng | Đọc/ghi được |
| **T, C** | Timer, Counter | Bộ định thời, bộ đếm nội bộ của ladder | Thường chỉ đọc giá trị hiện tại |

Điểm mấu chốt: **X và Y là các vùng nhớ ánh xạ trực tiếp tới phần cứng vật lý** (input/output module), trong khi **M và D là vùng nhớ "ảo"**, chỉ tồn tại trong logic của PLC. Khi bạn ghi bit vào Y0, PLC lập tức đặt điện áp ra chân vật lý Y0 (bật rơ-le/đèn nối vào đó) — tác động ngay tức thì, không qua bất kỳ logic trung gian nào.

Đây cũng là lý do mình từng lưu ý ở phần trước: nếu PLC của bạn có sẵn một đoạn ladder khác cũng đang ghi vào Y0/Y1/Y2 (ví dụ logic gốc điều khiển đèn theo trạng thái máy sản xuất), thì việc PC ghi từ bên ngoài sẽ "đánh nhau" với ladder — PLC chạy scan cycle hàng chục lần mỗi giây, ladder có thể ghi đè lại giá trị PC vừa gửi trong tích tắc. Cách làm an toàn theo đúng chuẩn công nghiệp là PC ghi vào một bit M (ví dụ M100/M101/M102), rồi ladder đọc M đó và tự quyết định bật Y — tách biệt "tín hiệu từ bên ngoài" và "quyền điều khiển phần cứng thật" vẫn nằm trong tay logic PLC. Vì bạn đã xác nhận đèn tháp đấu thẳng Y0/Y1/Y2 và không có ladder nào khác động vào, ghi thẳng Y là chấp nhận được cho hệ thống hiện tại — nhưng nếu sau này ai đó thêm logic ladder mới đụng tới 3 ngõ này, cần nhớ lại nguyên tắc này để tránh xung đột.

## 5. MC Protocol: giao thức cụ thể bạn đang dùng

MC Protocol (viết tắt của MELSEC Communication Protocol) là giao thức độc quyền của Mitsubishi, cho phép thiết bị bên ngoài đọc/ghi trực tiếp vào các vùng nhớ (X, Y, M, D...) của PLC mà không cần PLC chạy thêm bất kỳ đoạn code đặc biệt nào để "phục vụ" request đó — module Ethernet tự xử lý toàn bộ việc diễn giải gói tin, PLC ladder hoàn toàn không biết có ai đang đọc/ghi từ bên ngoài (trừ khi bạn chủ động thêm logic đọc M như mục 4).

**Các định dạng khung (frame):**
- **Khung 1E**: định dạng cũ, đơn giản hơn, dùng cho các PLC đời cũ (FX cổ, module Ethernet thế hệ đầu).
- **Khung 3E**: định dạng phổ biến nhất hiện nay, dùng cho Q series và iQ-R — đây là loại bạn sẽ dùng. Hỗ trợ cả 2 kiểu mã hóa: **ASCII** (gói tin là chuỗi ký tự đọc được, dễ debug bằng mắt nhưng dung lượng lớn hơn) và **Binary** (gói tin là dữ liệu nhị phân thuần, nhỏ gọn và nhanh hơn, là lựa chọn mặc định trong hầu hết thư viện lập trình).
- **Khung 4E**: mở rộng của 3E, thêm số serial để theo dõi từng gói tin trong hệ thống phức tạp — ít dùng cho ứng dụng đơn giản như đèn báo.

**Cấu trúc khái quát của một gói tin khung 3E (binary) khi ghi 1 bit vào Y0:**

Gói tin gửi đi gồm các phần: (1) header xác định đây là khung 3E và kiểu binary, (2) mã lệnh cho biết đây là thao tác "ghi bit" (khác với "đọc bit" hay "đọc/ghi word"), (3) tên vùng nhớ đích — mã hóa chữ "Y" thành một mã số cố định theo bảng quy ước của Mitsubishi, (4) số thứ tự thiết bị (0 cho Y0), (5) giá trị cần ghi (1 = bật). PLC nhận gói, kiểm tra header hợp lệ, thực hiện lệnh, rồi gửi lại một gói phản hồi ngắn xác nhận đã ghi thành công (hoặc mã lỗi nếu địa chỉ sai/không hợp lệ). Toàn bộ quá trình này diễn ra trong vài mili-giây.

Bạn không cần tự tay xây dựng gói tin nhị phân này — đó chính là việc thư viện `pymcprotocol` làm thay bạn: nó cung cấp các hàm ở mức cao (như "ghi bit vào Y0"), tự động đóng gói thành đúng định dạng khung 3E, gửi qua socket TCP, nhận phản hồi, và báo lại cho chương trình Python biết thành công hay thất bại. Nhưng hiểu được cấu trúc bên dưới giúp bạn đọc hiểu tài liệu kỹ thuật của Mitsubishi và tự chẩn đoán lỗi khi cần (ví dụ dùng Wireshark bắt gói tin để xem PLC có nhận được đúng lệnh không).

## 6. Cách xác định các thông số thực tế cần cho kết nối

Để kết nối thành công, có 4 thông tin bắt buộc phải lấy đúng từ chính PLC, không thể đoán:

**IP và port của module Ethernet**: mở phần mềm GX Works2/GX Works3 (đã dùng để lập trình PLC), vào mục cấu hình module Ethernet (Network Parameter / Ethernet Port setting). Ở đó sẽ thấy IP tĩnh đã gán, và mục cấu hình cổng dịch vụ (thường ghi là "MC Protocol" hoặc "Open Setting" với loại giao thức chọn là "MC Protocol", kèm số port). Nếu không có quyền truy cập file dự án PLC, cần hỏi người đã lắp đặt tủ điện — đây là thông tin cấu hình, không thể dò ra từ bên ngoài một cách dễ dàng.

**Loại PLC (Q hay iQ-R)**: nhìn nhãn dán trên chính thân CPU của PLC (ví dụ Q series ghi rõ "Q0xUDV..." còn iQ-R ghi "R0xCPU..."), hoặc xem trong project GX Works3 (iQ-R dùng GX Works3, Q series cũ hơn có thể dùng GX Works2). Thông tin này quyết định thư viện đóng gói header đúng định dạng — chọn sai, PLC sẽ từ chối gói tin ngay ở bước đầu.

**Địa chỉ Y thật của đèn tháp**: đây không phải thứ đoán được, mà phải đọc từ sơ đồ đấu dây tủ điện (bản vẽ điện) hoặc mở chương trình ladder trong GX Works xem chân Y nào thực sự nối ra rơ-le điều khiển đèn tháp — nhìn vật lý dây nối từ module output của PLC ra tới đèn, lần theo số hiệu ghi trên terminal block.

**Mạng LAN chung**: xác nhận PC và PLC join chung 1 mạng — cách nhanh nhất là từ PC mở Command Prompt, gõ lệnh `ping <IP của PLC>`, nếu có phản hồi (reply) tức là tầng mạng đã thông, phần còn lại chỉ là vấn đề đúng giao thức/port.

## 7. Vòng đời một lần giao tiếp — tổng hợp lại toàn bộ quy trình

Khi hệ thống nhận diện khuôn mặt phát hiện một sự kiện (ví dụ nhận diện thành công lúc vào), toàn bộ chuỗi sự kiện diễn ra như sau: chương trình Python quyết định cần bật đèn xanh → mở (hoặc dùng lại) kết nối TCP tới IP:port của module Ethernet trên PLC → đóng gói yêu cầu "ghi bit Y0 = 1" theo đúng cấu trúc khung 3E binary mà thư viện `pymcprotocol` xử lý sẵn → gửi gói tin qua mạng LAN → module Ethernet trên PLC nhận, xác thực gói tin hợp lệ, ghi giá trị 1 vào vùng nhớ Y0 → phần cứng module output tương ứng đóng mạch, dòng điện chạy tới đèn xanh, đèn sáng → PLC gửi phản hồi xác nhận về PC → sau khoảng thời gian định trước (1–2 giây), chương trình Python gửi tiếp một gói ghi Y0 = 0 để tắt đèn.

Toàn bộ chuỗi này thường chỉ mất vài đến vài chục mili-giây cho mỗi lượt ghi — đủ nhanh để không ảnh hưởng tới trải nghiệm nhận diện thời gian thực.

## 8. Những rủi ro và cách tư duy khi thiết kế hệ thống thật

**Mất kết nối tạm thời**: mạng công nghiệp có thể chập chờn (nhiễu điện, cáp lỏng). Chương trình phía PC cần có cơ chế phát hiện lỗi khi gửi gói tin thất bại, và tự động thử kết nối lại ở lần gửi kế tiếp, thay vì để cả chương trình treo hoặc crash chỉ vì đèn báo không gửi được — hệ thống chấm công chính vẫn phải hoạt động độc lập với việc đèn có sáng hay không.

**Xung đột ghi đè giữa PC và ladder** (đã nói ở mục 4): nếu tương lai có ai chỉnh sửa chương trình PLC và vô tình để ladder cũng điều khiển Y0/Y1/Y2, cần chuyển sang ghi qua vùng M trung gian.

**Tốc độ gửi lệnh dồn dập**: nếu nhiều sự kiện xảy ra liên tiếp trong thời gian ngắn (nhiều người quét mặt liền nhau), cần có cơ chế xếp hàng (queue) để các lệnh bật/tắt đèn được xử lý tuần tự, tránh việc gửi 2 lệnh ghi đè lẫn nhau gây ra trạng thái đèn sai (ví dụ vừa bật xanh xong bị lệnh đỏ ghi đè ngay lập tức mà chưa kịp hiển thị).

**Bảo mật mạng**: MC Protocol nguyên bản không có xác thực hay mã hóa — bất kỳ máy nào trong cùng mạng LAN cũng có thể gửi lệnh ghi tới PLC nếu biết đúng IP/port. Trong môi trường xưởng sản xuất kín, đây thường chấp nhận được, nhưng nếu mạng đó có kết nối ra ngoài (Internet, wifi công cộng...) cần cách ly bằng VLAN riêng hoặc firewall, không để PLC lộ ra mạng ngoài.

## 9. Công cụ hỗ trợ chẩn đoán khi có sự cố

Khi kết nối không hoạt động, thứ tự kiểm tra hợp lý là: (1) `ping` IP của PLC từ PC để xác nhận tầng mạng — nếu không ping được, dừng lại kiểm tra cáp/switch/IP trước khi đi tiếp; (2) dùng chính GX Works kết nối thử vào PLC qua Ethernet để xác nhận module Ethernet còn sống và đúng cấu hình MC Protocol đang bật; (3) nếu ping được nhưng chương trình Python vẫn báo lỗi kết nối, dùng công cụ bắt gói tin mạng như Wireshark trên PC, lọc theo IP của PLC, xem có gói tin nào được gửi đi và PLC có phản hồi không — nếu không có phản hồi, khả năng cao sai port hoặc PLC đang chặn kết nối đó; (4) nếu kết nối thành công nhưng ghi Y không có tác dụng, kiểm tra lại đúng địa chỉ Y bằng cách quan sát trực tiếp trạng thái Y trên GX Works (chế độ online monitor) trong lúc chương trình Python gửi lệnh — nếu thấy Y đổi trạng thái trên GX Works nhưng đèn thật không sáng, vấn đề chuyển sang phần cứng (dây đấu, rơ-le, nguồn cấp đèn) chứ không còn là phần mềm nữa.

## 10. Tóm lại — bức tranh tổng thể

Bạn đang xây một hệ thống có 3 lớp tách biệt rõ ràng, và hiểu rõ ranh giới giữa 3 lớp này là chìa khóa để debug đúng chỗ khi có lỗi:

Lớp nhận diện (Python/YOLO/ArcFace) chỉ quan tâm "ai vừa quét mặt, kết quả là gì" — hoàn toàn không biết gì về PLC. Lớp giao tiếp (`plc_light.py`/MC Protocol/Ethernet) chỉ có nhiệm vụ nhận một tín hiệu đơn giản ("bật xanh", "bật đỏ"...) và dịch nó thành đúng gói tin mạng để PLC hiểu — không quan tâm tín hiệu đó tới từ đâu hay có ý nghĩa gì trong nghiệp vụ chấm công. Lớp phần cứng (PLC + đèn tháp) chỉ biết đọc bit ở Y0/Y1/Y2 và bật/tắt điện tương ứng — hoàn toàn không biết gì về khuôn mặt hay AI.

Khi có sự cố, việc đầu tiên là xác định lỗi nằm ở lớp nào (nhận diện sai người? gói tin không gửi được? Y đúng nhưng đèn không sáng vì hỏng dây?) — nhờ 3 lớp độc lập, mỗi lớp có thể kiểm tra và sửa riêng mà không ảnh hưởng 2 lớp còn lại.
