# Industrial Visual Anomaly Detection System (MVTec AD - PatchCore)

Hệ thống phát hiện và định vị lỗi ngoại quan sản phẩm trong công nghiệp theo phương pháp **One-Class Visual Anomaly Detection (PatchCore)**. Mô hình chỉ học trên hình ảnh sản phẩm đạt chuẩn (`train/good`), sau đó trích xuất vector đặc trưng patch đa tầng và đo khoảng cách tới tập bộ nhớ đại diện (**Memory Bank Coreset**).

> **Giấy phép Dữ liệu:** Dataset MVTec AD áp dụng giấy phép **CC BY-NC-SA 4.0** (Phi thương mại). Dự án này được thiết kế phục vụ mục đích nghiên cứu, học tập và làm Portfolio CV cho AI Engineer.

---

## 🌟 Điểm Nổi Bật Của Dự Án (Key Highlights)

- **Kiến trúc PatchCore CVPR 2022**: Trích xuất đặc trưng đa tầng (*Multi-Layer Feature Embedding*) kết hợp Layer 2 & Layer 3 của ResNet18 backbone, gia tăng vượt trội khả năng định vị lỗi hạt mịn (*Pixel AP* và *AUPRO@0.3*).
- **Thuật toán Greedy K-Center Coreset**: Áp dụng phép chiếu ngẫu nhiên *Johnson-Lindenstrauss (64D)* kết hợp bao phủ K-Center greedy giúp giảm **~90% dung lượng RAM** và chi phí tính toán memory bank mà vẫn bảo toàn độ phủ không gian đặc trưng.
- **Căn chỉnh Ngữ cảnh Độc lập (Held-out Normal Calibration)**: Giữ riêng 20% dữ liệu ảnh bình thường để hiệu chỉnh ngưỡng phát hiện lỗi ($99^\text{th}$ percentile), loại bỏ triệt để hiện tượng tự chấm điểm (*overfitting*) trên memory bank.
- **Trực quan hóa Vùng Lỗi (Gaussian Smoothing & Heatmap Overlay)**: Tích hợp bộ lọc Gaussian Blur làm mịn anomaly map và tự động tạo ảnh phủ màu bản đồ nhiệt lỗi dạng **Base64 PNG** cho giao diện người dùng.
- **Chuẩn Hóa Sản Phẩm (Production-Ready REST API)**: FastAPI Server sử dụng **Pydantic Data Contracts**, kiểm tra an toàn dung lượng (Max 10MB), chống tấn công *Decompression Bomb*, kèm theo lazy-loading mô hình.
- **Clean Code & Test Suite**: 100% mã nguồn tuân thủ Type Hints, chú thích docstring Tiếng Việt theo chuẩn Google Python Style Guide, và bộ test suite toàn diện (`pytest`).

---

## 📐 Kiến Trúc Pipeline & Cơ Sở Toán Học

```text
[Hình ảnh đầu vào 224x224] 
           │
           ▼
[ResNet18 Backbone (Pretrained ImageNet)]
 ├── Layer 2 (128 channels, 28x28) ───┐
 └── Layer 3 (256 channels, 14x14) ───┴──► Bilinear Upsample ──► Concatenate [384 channels, 28x28]
                                                                        │
 ┌──────────────────────────────────────────────────────────────────────┘
 │
 ├──► Huấn luyện Memory Bank:
 │    Patch Features [N, 384] ──► Random Projection (64D) ──► Greedy K-Center Coreset ──► 1-NN Bank [K, 384]
 │                                                                                              │
 └──► Suy luận & Kiểm định:                                                                     │
      Test Patch Features ──► Compute 1-NN Distance Map ──► Gaussian Blur (σ=1.0) ◄────────────┘
                                  │
                                  ├──► Anomaly Score (99th Percentile) ──► PASS / REVIEW / FAIL
                                  └──► Heatmap Overlay Base64 Image
```

### Cơ Sở Toán Học

1. **Multi-layer Patch Embedding**: 
   Vector đặc trưng tại vị trí patch $(i, j)$ được hợp nhất từ tầng trung gian:
   $$\phi(x)_{i,j} = \left[ f^2(x)_{i,j} \;\parallel\; \text{Upsample}\left(f^3(x)\right)_{i,j} \right] \in \mathbb{R}^{384}$$

2. **Greedy K-Center Coreset Selection**:
   Tìm tập con $\mathcal{M}^* \subset \mathcal{M}$ có kích thước $K$ sao cho:
   $$\mathcal{M}^* = \arg\min_{\mathcal{M}' \subset \mathcal{M}, |\mathcal{M}'|=K} \max_{m \in \mathcal{M}} \min_{m' \in \mathcal{M}'} \| m - m' \|_2$$

3. **AUPRO@0.3 (Area Under Per-Region Overlap)**:
   Đánh giá độ chính xác định vị vùng lỗi không phụ thuộc kích thước vùng lỗi lớn hay nhỏ bằng cách tính trung bình diện tích giao (overlap) trên từng thành phần liên thông $S_k$ với FPR $\le 0.3$:
   $$\text{PRO}(\theta) = \frac{1}{N_{\text{regions}}} \sum_{k=1}^{N_{\text{regions}}} \frac{|S_k \cap P(\theta)|}{|S_k|}$$

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
03_mvtec_anomaly_detection/
├── configs/               # Lưu trữ file cấu hình bổ sung
├── data/                  # Dữ liệu thô MVTec AD (download qua script)
├── models/                # Artifacts mô hình (patch_nn.joblib, memory.npy, config.json)
├── reports/               # Báo cáo kết quả đánh giá (test_metrics.json)
├── scripts/
│   └── download_data.py   # Script tải dữ liệu MVTec AD từ Hugging Face mirror
├── src/
│   ├── __init__.py
│   ├── api.py             # FastAPI REST Server với Pydantic response models
│   ├── config.py          # Dataclass TrainConfig và CLI parser
│   ├── data.py            # PyTorch Dataset và tiền xử lý ImageNet
│   ├── evaluate.py        # Pipeline đánh giá các chỉ số Image/Pixel AUROC, AP, AUPRO@0.3
│   ├── features.py        # ResNet18 Multi-Layer Feature Extractor
│   ├── inference.py       # Engine suy luận, điểm anomaly score và heatmap overlay
│   ├── train.py           # Script huấn luyện Memory Bank Coreset & Held-out Calibration
│   └── utils.py           # Thuật toán Greedy Coreset, Gaussian Smoothing, Heatmap Overlay
├── tests/
│   ├── test_pipeline.py   # Unit tests kiểm tra feature extractor, dataset, config & schemas
│   └── test_smoke.py      # Smoke tests kiểm tra các hàm tiện ích & health check
├── Dockerfile             # Containerization sẵn sàng cho deployment
├── Makefile               # Lệnh tắt cho venv, train, eval, test, api
├── README.md              # Tài liệu chi tiết dự án
├── RESEARCH_REPORT.md     # Báo cáo tiến trình nghiên cứu & nâng cấp v3
└── requirements.txt       # Danh sách thư viện phụ thuộc Python
```

---

## 🚀 Hướng Dẫn Cài Đặt & Vận Hành

### 1. Khởi Tạo Môi Trường

```bash
# Tạo môi trường ảo Python
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell):
.venv\Scripts\Activate.ps1

# Cài đặt gói phụ thuộc:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Tải Dữ Liệu MVTec AD (Category: `bottle`)

```bash
python scripts/download_data.py
```

### 3. Huấn Luyện Mô Hình PatchCore

```bash
python -m src.train \
  --category bottle \
  --batch-size 8 \
  --calibration-fraction 0.2 \
  --threshold-quantile 0.99 \
  --coreset-fraction 0.05 \
  --smooth-sigma 1.0 \
  --seed 42
```

### 4. Đánh Giá Mô Hình (Evaluation)

```bash
python -m src.evaluate
```

> Lưu ý artifact `models/` đi kèm có thể thuộc schema v1. Sau khi chuyển sang random held-out calibration, hãy chạy lại `python -m src.train` trong môi trường đã cài PyTorch để tạo artifact schema v2 trước khi triển khai.

### 5. Chạy Kiểm Thử Tự Động (Automated Testing)

```bash
python -m pytest -v
```

### 6. Khởi Chạy REST API Server

```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

---

## 📊 Kết Quả Đánh Giá (Benchmark Category `bottle`)

Kết quả thu được khi đánh giá trên toàn bộ tập ảnh thử nghiệm MVTec AD (`bottle` category):

| Chỉ số Đánh Giá (Metric) | Giá Trị Thực Nghiệm | Ý Nghĩa Kỹ Thuật |
|---|---:|---|
| **Image-level AUROC** | **0.9960** | Phân loại chính xác 99.6% ảnh lỗi vs ảnh bình thường |
| **Image-level AP** | **0.9987** | Độ chính xác trung bình phân loại cấp ảnh cực kỳ cao |
| **Pixel-level AUROC** | **0.9475** | Phân biệt pixel lỗi trên toàn bộ bản đồ nhiệt |
| **Pixel-level AP** | **0.4179** | Độ chính xác định vị pixel lỗi (loại bỏ âm tính giả) |
| **AUPRO@0.3** | **0.7421** | Độ phủ trên từng vùng lỗi nhỏ/kích thước hạt mịn |
| **Accuracy tại Threshold** | **0.9759** | Độ chính xác đưa ra quyết định vận hành PASS/FAIL |

---

## 🔌 Tài Liệu REST API Endpoints

### 1. Health Check (`GET /health`)
Kiểm tra sức khỏe dịch vụ và tình trạng nạp mô hình.

```bash
curl -X GET http://127.0.0.1:8000/health
```

**Response mẫu (`200 OK`):**
```json
{
  "status": "ok",
  "model_ready": true,
  "model_version": "mvtec-resnet18-patchcore-v3"
}
```

### 2. Kiểm Định Lỗi Ảnh (`POST /inspect`)
Tải ảnh lên để phân tích điểm lỗi, nhận quyết định vận hành và chuỗi ảnh heatmap overlay Base64.

```bash
curl -X POST http://127.0.0.1:8000/inspect \
  -F "file=@sample_bottle.png" \
  -F "include_overlay=true"
```

**Response mẫu (`200 OK`):**
```json
{
  "anomaly_score": 0.4215,
  "threshold": 0.4850,
  "decision": "PASS",
  "heatmap_shape": [28, 28],
  "model_version": "mvtec-resnet18-patchcore-v3",
  "overlay_b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMAAAAD..."
}
```

*Quy tắc Quyết định Vận hành (Decision Logic):*
- `FAIL`: `anomaly_score >= threshold` (Sản phẩm bị lỗi ngoại quan)
- `REVIEW`: `0.8 <= ratio < 1.0` (Cần nhân viên QC kiểm tra lại thủ công)
- `PASS`: `ratio < 0.8` (Sản phẩm đạt chuẩn chất lượng)

---

## 💼 Mẫu Điểm Nổi Bật Đưa Vào CV (CV Bullet Points)

Dưới đây là các câu tóm tắt thành tựu kỹ thuật bạn có thể sử dụng trực tiếp trong CV của mình:

- **Industrial Anomaly Detection System (MVTec AD - PatchCore)**:
  - Thiết kế pipeline phát hiện lỗi ngoại quan theo hướng *One-Class Classification* dựa trên bài báo PatchCore (CVPR 2022), đạt **Image AUROC 0.9960** và **Pixel AUROC 0.9475** trên dữ liệu MVTec AD `bottle`.
  - Tối ưu hóa **Greedy K-Center Coreset (Johnson-Lindenstrauss 64D Projection)** giúp nén tập bộ nhớ đặc trưng, giảm **90% dung lượng RAM** và duy trì thời gian suy luận thời gian thực trên CPU.
  - Thiết lập quy trình **Held-out Normal Calibration** giữ riêng 20% dữ liệu chuẩn hóa để xác định ngưỡng $99^\text{th}$ percentile độc lập, ngăn ngừa overfitting.
  - Triển khai **FastAPI Server** với Pydantic Data Contracts, bộ lọc **Gaussian Blur Heatmap**, tự động xuất **Visual Overlay Base64** và tích hợp phòng chống rủi ro bảo mật (Decompression Bomb, max upload 10MB).

---

## ⚠️ Lưu Ý Triển Khai Production (Production Guidelines)

1. **Độ phân giải 224x224**: Ảnh được resize cố định về 224x224 để phù hợp với chuẩn ResNet18. Với các lỗi cực nhỏ (micro-scratches < 2px), cần xem xét chia patch độ phân giải cao hơn (448x448).
2. **Ngưỡng theo Danh Mục (Category-Specific Threshold)**: Giá trị `threshold` được căn chỉnh theo từng loại sản phẩm. Khi chuyển sang danh mục mới (ví dụ: `cable`, `transistor`), cần chạy lại `python -m src.train --category <name>`.
3. **Mở rộng GPU/FAISS**: Với dây chuyền công nghiệp tốc độ cao (>100 FPS), khuyến nghị thay thế `sklearn.neighbors.NearestNeighbors` bằng chỉ mục `faiss-gpu` (IndexFlatL2).
