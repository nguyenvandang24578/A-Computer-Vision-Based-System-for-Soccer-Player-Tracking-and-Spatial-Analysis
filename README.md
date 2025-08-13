# A-Computer-Vision-Based-System-for-Soccer-Player-Tracking-and-Spatial-Analysis

# ⚽ Soccer Player Tracking & Spatial Analysis System

## 📌 Giới thiệu
Dự án này phát triển một hệ thống **thị giác máy tính** kết hợp **học sâu** để **phát hiện – theo dõi – phân tích chuyển động** của cầu thủ bóng đá từ **video quay đơn giản**, không yêu cầu thiết bị chuyên dụng đắt tiền.  
Kết quả đầu ra bao gồm:
- **Bản đồ nhiệt (heatmap)** thể hiện phạm vi hoạt động.
- **Đường di chuyển** của từng cầu thủ.
- **Thông tin thống kê** hỗ trợ phân tích chiến thuật.

Hệ thống đặc biệt phù hợp cho:
- Các đội bóng học đường / phong trào.
- Huấn luyện viên và nhà phân tích chiến thuật.
- Các dự án nghiên cứu và ứng dụng giáo dục thể thao.

---

## 🎯 Mục tiêu & Động lực
- Tự động hóa phân tích chiến thuật bóng đá.
- Giảm thời gian và chi phí so với phương pháp thủ công.
- Tạo ra kết quả phân tích trực quan, dễ hiểu và dễ áp dụng.
- Mở rộng cơ hội tiếp cận công nghệ phân tích bóng đá hiện đại.

---

## 📂 Dữ liệu sử dụng
1. **Phát hiện cầu thủ & bóng**: Video bóng đá sân 5 (Pixellot AI), 25 FPS, gán nhãn theo chuẩn YOLO.
2. **Nhận diện số áo**: Dataset *Jersey Number Detection* (Roboflow), 24.315 ảnh train + 1.205 ảnh val.
3. **Tăng cường dữ liệu**: Xoay, thay đổi độ sáng, phóng to/thu nhỏ, lật ảnh.

---

## 🛠 Phương pháp
### 1. **Phát hiện cầu thủ và bóng**
- Mô hình **YOLOv8** cho phát hiện đối tượng thời gian thực.
- Bounding box cho từng cầu thủ và bóng.

### 2. **Nhận diện số áo & phân loại đội**
- Cắt vùng bounding box từ Player Detection.
- YOLOv8 nhận diện số áo.
- Trích xuất đặc trưng màu (HSV, LAB, RGB) + KMeans để phân loại đội.

### 3. **Tracking nâng cao – TeamAwareByteTracker**
- Cải tiến từ **ByteTrack**:
  - Lịch sử phân loại đội (Team Consistency).
  - So khớp đặc trưng màu (Feature Similarity).
  - Nhất quán vị trí (Position Consistency).
  - Ổn định số áo (Jersey Stabilization).
- Giảm **ID Switch** tới **46.2%** so với ByteTrack gốc.

### 4. **Nội suy vị trí bóng**
- Kết hợp nội suy tuyến tính và ngoại suy vận tốc.
- Duy trì quỹ đạo bóng liên tục.

### 5. **Chuyển đổi tọa độ & trực quan hóa**
- Áp dụng **perspective transformation** để ánh xạ tọa độ pixel → sân 2D.
- Sinh **heatmap**, **movement trails**, so sánh hoạt động giữa hai đội.

---

## 📊 Kết quả nổi bật
- **MOTA**: tăng từ 0.685 → **0.785** (+14.6%).
- **ID Switch**: giảm từ 52 → **28** (-46.2%).
- Nhận diện số áo: Precision ≈ 0.95, Recall ≈ 0.94, mAP50 ≈ 0.965.
- Hệ thống hoạt động ổn định trong môi trường thực tế.

### Overall Performance
- **Best Configuration**: FullTeamAware
- **MOTA Range**: 0.6850 - 0.7850
- **Average MOTA**: 0.7257

### ID Switch Reduction
- **Baseline ID Switches**: 52
- **Best Performance**: 28 switches
- **Reduction**: 24 switches (46.2% improvement)

### Object 4 Specific Tracking
- **Baseline Switch Rate**: 0.1923
- **Best Switch Rate**: 0.0923
- **Improvement**: 52.0%

---

## 📈 Configuration Performance

| Configuration | MOTA | ID Switches | Obj4 Switch Rate | Improvement |
|---------------|------|-------------|------------------|-------------|
| BaselineByteTrack | 0.6850 | 52 | 0.1923 | +0.0% |
| ByteTrack+TeamHistory | 0.7120 | 45 | 0.1615 | +3.9% |
| ByteTrack+Features | 0.7045 | 48 | 0.1731 | +2.8% |
| ByteTrack+Position | 0.6980 | 49 | 0.1808 | +1.9% |
| ByteTrack+Jersey | 0.7200 | 42 | 0.1538 | +5.1% |
| ByteTrack+Team+Features | 0.7380 | 38 | 0.1385 | +7.7% |
| ByteTrack+Team+Position | 0.7290 | 40 | 0.1462 | +6.4% |
| ByteTrack+Team+Jersey | 0.7450 | 35 | 0.1231 | +8.8% |
| ByteTrack+Features+Position | 0.7250 | 41 | 0.1500 | +5.8% |
| ByteTrack+Features+Jersey | 0.7350 | 37 | 0.1346 | +7.3% |
| ByteTrack+Position+Jersey | 0.7320 | 39 | 0.1423 | +6.9% |
| **FullTeamAware** | **0.7850** | **28** | **0.0923** | **+14.6%** |

---

## 🚀 Hướng phát triển
- Cải thiện nhận diện trong điều kiện ánh sáng kém.
- Xử lý tốt hơn các tình huống che khuất phức tạp.
- Tối ưu hóa tốc độ xử lý để đạt real-time.
- Nâng cấp mô hình AI (YOLOv11, transformer-based tracking).
- Xây dựng giao diện web/desktop cho người dùng.

---

## 📧 Liên hệ
- **Leader**: Nguyễn Văn Đăng  
- Email: *nguyendangdh1@gmail.com*  
- GitHub: **https://github.com/nguyenvandang24578**
