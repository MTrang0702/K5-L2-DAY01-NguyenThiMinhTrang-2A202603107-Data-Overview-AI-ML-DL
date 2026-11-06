# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** *11/06/2026*
**Runtime Colab:** GPU (Tesla T4 / CUDA)
**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145
**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`
**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`
**Thay đổi so với notebook nguồn:** Không

---

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

---

## 1. Phân loại ảnh

1. **Chép record hạng 1:** `class_id`=919 ; `class_name`=streetcar ; `rank`=1 ; `score`=0.41 ; `taxonomy_name`=ImageNet-1K
2. **Phạm vi gán nhãn:** Record này nói về **cả bức ảnh**. Vì đây là bài toán Image Classification, mô hình đầu ra gán một nhãn duy nhất đại diện cho toàn bộ ngữ cảnh của ảnh chứ không khoanh vùng từng đối tượng đơn lẻ.
3. **Nguồn gốc danh sách lớp:** Danh sách lớp đi kèm theo **taxonomy/checkpoint** (`yolo11n-cls.pt` huấn luyện trên tập dữ liệu ImageNet-1K), không phải do mô hình tự nghĩ ra.
4. **Lý do cần ghi đủ 3 thông số:** `class_id` để máy tính xử lý định danh, `class_name` giúp con người dễ đọc hiểu, và `taxonomy_name` giúp xác định ngữ cảnh danh mục lớp (tránh nhầm lẫn giữa các bộ dữ liệu khác nhau như ImageNet và COCO).
5. **Quy tắc Guideline khi ảnh có nhiều vật:** Guideline cần quy định rõ tiêu chí ưu tiên chọn nhãn: chọn đối tượng chính chiếm diện tích lớn nhất ở trung tâm ảnh, hoặc chọn lớp đại diện cho ngữ cảnh tổng thể chung (context).
6. **Không lấy model score làm ground truth:** Vì `score` chỉ thể hiện độ tự tin mang tính xác suất của mô hình toán học (có thể đoán sai hoặc bị nhiễu), còn ground truth đòi hỏi độ chính xác tuyệt đối dựa trên thực tế do con người xác minh.

---

## 2. Phát hiện vật thể

1. **Chép một record (`sample_id = kitchen`):** `class_name`=refrigerator ; `score`=0.88 ; `bbox_xyxy`=[20, 100, 300, 500] ; `bbox_width`=280 ; `bbox_height`=400
2. **Giải thích box bằng lời:** Bounding box có góc trên-trái tại tọa độ pixel (20, 100), góc dưới-phải tại tọa độ pixel (300, 500), chiều rộng box là 280 pixel và chiều cao là 400 pixel.
3. **Số lượng prediction theo ngưỡng:**
   - Ngưỡng 0.20: 12 predictions (giữ nhiều box, dễ bị box thừa/nhiễu).
   - Ngưỡng 0.60: 4 predictions (chỉ giữ box rất chắc chắn, dễ bỏ sót vật bị che).
4. **Nhiệm vụ của Reviewer khi hạ ngưỡng:** Reviewer phải tốn thời gian lọc, kiểm tra kỹ và xóa bỏ các bounding box rác/dự đoán sai (false positives) do mô hình giữ lại quá nhiều.
5. **Quy tắc gán nhãn box:** "Khung chữ nhật (bounding box) phải bao sát hoàn toàn mép ngoài cùng của vật thể, không cắt lẹm vật thể và giữ khoảng trống thừa nhỏ hơn 5% diện tích box."
6. **Vật bị che/cắt mép và xử lý:** Trên ảnh `kitchen`, chiếc bàn/ghế bị che khuất một phần. Trường hợp này sẽ **đưa lên hỏi Lab Coach (escalation)** nếu guideline chưa quy định rõ phần trăm che khuất tối đa cần khoanh vùng, tránh việc người gán nhãn tự bịa quy tắc.

---

## 3. Phân đoạn theo từng đối tượng

1. **Chép một record (`sample_id = kitchen`):** `instance_id`=inst_01 ; `class_name`=cup ; `score`=0.82 ; `polygon_point_count`=18 ; `polygon_xy`=[[120, 240], [125, 238], [130, 245], [128, 255]]
2. **Thông tin Polygon cung cấp so với Box:** Polygon cung cấp chi tiết đường viền chính xác hình dạng thực tế của vật thể (pixel-level mask), thay vì chỉ là một khung hình vuông/chữ nhật chứa nhiều diện tích nền thừa.
3. **Hiểu về `instance_id`:** `instance_id` là mã định danh duy nhất dùng để phân biệt các cá thể vật thể riêng biệt trong cùng một bức ảnh (ví dụ: cái cốc A và cái cốc B), nó không thay thế cho tên lớp và không theo dõi vật thể qua nhiều khung hình video khác nhau.
4. **Quy tắc vẽ biên mask:** "Vẽ các điểm polygon bám sát theo đường viền thực của vật thể, khoảng cách giữa các điểm nối không quá 5 pixel ở các đoạn cong và viền nét."
5. **Biên mờ/vật chạm nhau:** Sử dụng **Guideline** nếu trong quy trình đã có quy định về việc tách viền đối tượng đè lên nhau; nếu đối tượng quá mờ không thể phân biệt viền thì dùng **Escalation** để báo cáo.

---

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | Một mã/tên lớp đại diện cho toàn bộ ảnh[cite: 10] | Ảnh có nhiều vật thể khác nhau dẫn đến gán sai nhãn tổng thể | Chọn 1 lớp chuẩn nhất theo thứ tự ưu tiên trong guideline | Kiểm tra tên lớp đã khớp với quy tắc chọn nhãn chưa |
| **Phát hiện vật thể** | Bounding box (tọa độ xyxy) + Tên lớp | Box bao lỏng lẻo, bỏ sót vật thể bị che khuất một phần | Vẽ khung chữ nhật ôm sát biên ngoài của từng vật | Kiểm tra độ ôm sát của box và nhãn lớp tương ứng |
| **Instance segmentation** | Đa giác (Polygon mask) + Instance ID + Tên lớp | Biên vật thể bị mờ, hai vật nằm chồng lấn khó tách viền | Nhấp từng điểm viền chi tiết quanh ranh giới vật thể | Kiểm tra đường viền mask có bị đè viền hoặc lẹm sang vật khác |

---

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu:** "Không đăng tải, chia sẻ dữ liệu bài lab hoặc ghi thông tin cá nhân (họ tên, MSSV, email, SĐT) vào báo cáo và các tệp kết quả."[cite: 10, 12]
2. **Xử lý dữ liệu không đúng phạm vi:** Nếu gặp dữ liệu/ảnh nhạy cảm hoặc không đúng phạm vi bài tập, tôi sẽ dừng thực hiện và **báo cáo ngay cho Lab Coach / Giảng viên hướng dẫn**.

---

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output[cite: 10]
- [x] Đã nộp **link repository** trên LMS[cite: 10]