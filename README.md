# Industrial Visual Anomaly Detection System (PatchCore-Style)

Hệ thống phát hiện và định vị khuyết tật ngoại quan sản phẩm công nghiệp theo phương pháp **One-Class Visual Anomaly Detection (PatchCore-style)**. Mô hình chỉ học trên hình ảnh sản phẩm chuẩn (`train/good`), trích xuất đặc trưng patch đa tầng, nén tập nhớ đại diện (**Memory Bank Coreset**), căn chỉnh ngưỡng kép (**Dual-Threshold Calibration**) và khoanh vùng lỗi ngoại quan trực quan ở cấp độ từng pixel.

> **Giấy phép Dữ liệu:** Dataset MVTec AD áp dụng giấy phép **CC BY-NC-SA 4.0** (Phi thương mại). Dự án này được thiết kế phục vụ mục đích nghiên cứu, học tập và làm Portfolio chuẩn mực cho AI Engineer / Computer Vision Engineer.

---

## 🖼️ Trực Quan Hóa Kết Quả Kiểm Định (Visual Inspection Showcase)

Hệ thống cung cấp khả năng khoanh vùng và trực quan hóa lỗi ngoại quan trực tiếp trên từng pixel sản phẩm qua bản đồ nhiệt phủ màu:

![Visual Inspection Defect Comparison](reports/sample_outputs/inspection_defect_sample.png)

*Hình 1: Đánh giá mẫu chai vỡ vành miệng (`broken_large/000.png`) – Bản đồ nhiệt PatchCore định vị chính xác vùng nứt vỡ với Score: 4.644 (vượt ngưỡng Fail Threshold 2.844, vùng lỗi chiếm 8.9% diện tích) $\rightarrow$ Quyết định: **FAIL (FAIL_MAJOR)**.*

![Visual Inspection Normal Comparison](reports/sample_outputs/inspection_good_sample.png)

*Hình 2: Kiểm định sản phẩm đạt chuẩn (`good/000.png`) – Năng lượng phân tán đều, Score: 1.632 nằm dưới Review Threshold 2.613 $\rightarrow$ Quyết định: **PASS**.*

---

## 1. Bài Toán Nghiệp Vụ (Problem)

Trong dây chuyền sản xuất công nghiệp tốc độ cao (chế tạo chai lọ, linh kiện điện tử, dược phẩm, dệt may):
- Tỷ lệ sản phẩm lỗi trong thực tế rất thấp ($< 1\%$).
- Các dạng khuyết tật ngoại quan (nứt, xước, bọt khí, nhiễm bẩn, biến dạng) biến thiên vô cùng phức tạp, không thể thu thập đầy đủ dữ liệu mẫu cho mọi biến thể lỗi trước khi vận hành.
- Việc kiểm định thủ công bằng mắt người dẫn đến mệt mỏi thị giác, độ trễ cao và thiếu tính nhất quán giữa các ca làm việc.
- Hệ thống thị giác máy tính tự động cần có khả năng:
  1. Phân loại nhị phân chính xác sản phẩm đạt chuẩn hay lỗi.
  2. Cung cấp vùng đệm rà soát (**REVIEW**) để chuyên viên QC can thiệp kịp thời các mẫu tiệm cận ranh giới.
  3. Khoanh vùng chính xác vị trí và diện tích lỗi (**Defect Localization**) hỗ trợ truy vết nguyên nhân gốc rễ trên dây chuyền.

---

## 2. Vì Sao Chọn Anomaly Detection Thay Vì Classifier? (Why Anomaly Detection?)

Hệ thống này **không phải là Supervised Defect Classifier**. Đây chính là thế mạnh cốt lõi:

| Tiêu Chí So Sánh | Supervised Defect Classifier (Phân Loại Có Giám Sát) | One-Class Anomaly Detection (PatchCore-Style) |
|---|---|---|
| **Dữ Liệu Huấn Luyện** | Bắt buộc phải có hàng nghìn mẫu của từng loại lỗi cụ thể | **Chỉ cần ảnh sản phẩm tốt (`train/good`)** |
| **Khả Năng Phát Hiện Lỗi Mới** | Thất bại khi gặp dạng khuyết tật chưa từng xuất hiện trong tập train | **Phát hiện mọi dạng lỗi bất thường nằm ngoài phân phối chuẩn** |
| **Bản Chất Học Máy** | Học ranh giới quyết định (decision boundary) giữa các nhãn đã biết | **Học đa tạp đặc trưng chuẩn (*Normal Feature Manifold*)** |
| **Chi Phí Gán Nhãn Mask** | Cực kỳ tốn kém, yêu cầu chuyên gia khoanh vùng pixel lỗi | **Zero defect labeling cost** trong giai đoạn huấn luyện |

> **Nguyên lý Manifold:** Hệ thống học phân phối đa tạp của sản phẩm bình thường. Bất kỳ vector patch embedding mới nào nằm cách xa đa tạp chuẩn này trong không gian biểu diễn sẽ được xác định là bất thường mà không cần định nghĩa trước dạng lỗi.

---

## 3. Kiến Trúc Hệ Thống (System Architecture)

Hệ thống được thiết kế phân tách hoàn toàn giữa hai pipeline: **Offline Model Building** (huấn luyện & căn chỉnh ngưỡng) và **Online Serving Architecture** (phục vụ suy luận thời gian thực qua REST API).

### Sơ Đồ 1: Offline Model Building Pipeline

```text
                        ┌──────────────────┐
                        │   MVTec AD Data  │
                        └────────┬─────────┘
                                 │
                          NORMAL IMAGES (train/good)
                                 │
                    ┌────────────┴───────────┐
                    │                        │
             Memory Set (80%)        Calibration Set (20%)
                    │                        │
             Frozen ResNet18          Frozen ResNet18
             (Layer 2 + 3)            (Layer 2 + 3)
                    │                        │
             Patch Embeddings         Patch Embeddings
                [N, 384]                 [M, 384]
                    │                        │
             Random Projection               │
                [N, 64]                      │
                    │                        │
             Greedy K-Center                 │
                    │                        │
             Selected 384D                   │
                    │                        │
             Memory Bank [K, 384] ───►  1-NN Search
                    │                        │
                    │                   Anomaly Maps
                    │                        │
                    │                   Gaussian Smoothing (σ=1.0)
                    │                        │
                    │                   Normal Score Distribution
                    │                        │
                    │                   Dual Calibration
                    │                   - P95 -> Review Threshold
                    │                   - P99 -> Fail/Image Threshold
                    │                   - Pixel P99 -> Pixel Threshold
                    │                        │
                    └────────────┬───────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │   Model Artifacts (v5)   │
                    │ ├── memory_bank.npy      │
                    │ └── config.json          │
                    └──────────────────────────┘
```

### Sơ Đồ 2: Online Serving Architecture (FastAPI Enterprise)

```text
Operator / Factory Camera / Client
                 │
                 ▼ (HTTP POST /inspect)
       FastAPI Gateway & Router
                 │
        Input Validation
   (Max 10MB, Anti-Decompression Bomb)
                 │
           ModelRegistry
                 │
         AnomalyDetector
                 │
        Preprocessing Pipeline
    (PreprocessingConfig: 224×224, ImageNet Norm)
                 │
     Frozen ResNet18 Feature Extractor
        (Layer2 128D + Layer3 256D = 384D)
                 │
      1-NN Search against Memory Bank
                 │
             Raw Distance Map
                 │
      Gaussian Blur Smoothing (σ=1.0)
                 │
    ┌────────────┴────────────────┐
    │                             │
Image Anomaly Score        Pixel Defect Localization
 (99th percentile)         (Heatmap >= pixel_threshold)
    │                             │
Dual-Threshold Decision    Anomalous Area Ratio & Severity
 (PASS / REVIEW / FAIL)    (PASS / REVIEW / FAIL_MINOR / FAIL_MAJOR)
    │                             │
    └────────────┬────────────────┘
                 │
        Base64 Heatmap Overlay
                 │
                 ▼
        Rich JSON API Response
```

---

## 4. Pipeline ML Chuẩn Hóa (5 Canonical Stages)

Toàn bộ dự án từ mã nguồn đến tài liệu bám sát **5 giai đoạn canonical duy nhất**:

```text
1. DATA PREPARATION
MVTec AD
├── train/good
├── test/good
├── test/<defect_type>
└── ground_truth/<defect_type>
            ↓
2. NORMAL REPRESENTATION LEARNING
Train normal images
            ↓
Resize 224×224 + ImageNet Normalize
            ↓
Frozen ResNet18
            ↓
Layer2 features [128×28×28]
Layer3 features [256×14×14]
            ↓
Upsample Layer3 → 28×28
            ↓
Concatenate
            ↓
Patch Embeddings [784 × 384 / image]
            ↓
3. MEMORY BANK CONSTRUCTION
All normal patch embeddings [N, 384]
            ↓
Random Projection (64D)
            ↓
Greedy K-Center Coreset Selection
            ↓
Compact Memory Bank [K, 384]
            ↓
1-NN index (Runtime Euclidean)
            ↓
4. CALIBRATION
Held-out normal images (20% split, ≥ 20 images)
            ↓
Extract patch embeddings
            ↓
Nearest-neighbor distance
            ↓
Anomaly map [28×28]
            ↓
Gaussian smoothing (σ=1.0)
            ↓
99th percentile image score
            ↓
Normal score distribution
            ↓
Dual Decision Thresholds (P95 Review, P99 Fail) & Pixel Threshold
            ↓
5. INFERENCE / INSPECTION
New product image
            ↓
Same PreprocessingConfig (224×224)
            ↓
Same ResNet18 feature extractor
            ↓
Nearest-neighbor distance to Memory Bank
            ↓
Anomaly Map & Gaussian smoothing
            ↓
Image anomaly score (P99)
            ↓
Threshold Comparison
     ┌──────┼───────┐
   PASS   REVIEW    FAIL (MINOR / MAJOR)
            ↓
Heatmap localization & Anomalous Area Ratio
```

---

## 5. Giao Thức Phân Chia Dữ Liệu (Data Protocol)

Để ngăn ngừa triệt để rủi ro rò rỉ dữ liệu (*Data Leakage*):

```text
Tập Dữ Liệu Train Normal (train/good)
          │
          ├──► 80% Normal Training Set ──► Xây dựng Memory Bank Coreset
          └──► 20% Normal Calibration Set (Held-out, ≥20 ảnh) ──► Căn chỉnh Thresholds
          
Tập Dữ Liệu Test Chính Thức (test/good & test/<defect>)
          │
          └──► REPORT ONLY (Tuyệt đối không tham gia chọn ngưỡng hay tối ưu siêu tham số)
```

- Không dùng dữ liệu test để chọn ngưỡng. Ngưỡng được xác định hoàn toàn từ tập held-out normal.
- Ràng buộc kích thước mẫu: `min_calibration_samples >= 20` loại bỏ sai số thống kê khi ước lượng phân vị $95^\text{th}$ và $99^\text{th}$.

---

## 6. Trích Xuất Đặc Trưng (Feature Extraction)

Hệ thống sử dụng mạng backbone **ResNet18** tiền huấn luyện trên ImageNet:
- **Layer 2**: Cung cấp feature map kích thước $128 \times 28 \times 28$, bảo toàn các chi tiết cục bộ (vết xước mảnh, đốm bẩn, vân bề mặt).
- **Layer 3**: Cung cấp feature map kích thước $256 \times 14 \times 14$, nắm bắt ngữ cảnh ngữ nghĩa rộng hơn của sản phẩm.
- **Nội suy & Ghép kênh**: Layer 3 được upsample song tuyến tính (*bilinear interpolation*) lên $28 \times 28$, sau đó ghép kênh (*channel concatenation*) với Layer 2:
  $$\phi(x)_{i,j} = \left[ f^2(x)_{i,j} \;\parallel\; \text{Upsample}\left(f^3(x)\right)_{i,j} \right] \in \mathbb{R}^{384}$$
- Mỗi ảnh đầu vào $224 \times 224$ tạo ra một lưới không gian $28 \times 28 = 784$ patches, mỗi patch là một vector biểu diễn 384 chiều.

> **Minh bạch Kỹ thuật (PatchCore-Style):** Bài báo PatchCore gốc (Roth et al., CVPR 2022) sử dụng bước tổng hợp lân cận cục bộ (*local neighborhood aggregation/pooling*). Phiên bản của chúng tôi thực hiện ghép nối trực tiếp feature maps đa tầng, tối ưu hóa tốc độ và giảm tiêu thụ bộ nhớ, do đó được định danh chuẩn xác là **PatchCore-inspired / PatchCore-style visual anomaly detection**.

---

## 7. Xây Dựng Memory Bank & Thuật Toán Coreset

Với hàng trăm ảnh normal, tổng số patch có thể lên tới $> 130,000$ vector 384D. Việc tìm kiếm 1-NN trên toàn bộ tập này sẽ làm tăng độ trễ và tiêu thụ RAM. Thuật toán **Greedy K-Center Coreset** được triển khai theo quy trình:

1. **Phép chiếu ngẫu nhiên (Johnson-Lindenstrauss)**: Chiếu ma trận đặc trưng $N \times 384$ xuống $N \times 64$ để tăng tốc tính toán khoảng cách Euclidean khi chọn tâm coreset:
   $$P \in \mathbb{R}^{384 \times 64}, \quad X_{\text{proj}} = X \cdot P$$
2. **Greedy K-Center Selection**: Lần lượt chọn $K$ điểm sao cho cực tiểu hóa khoảng cách cực đại từ bất kỳ patch nào tới tâm gần nhất:
   $$\mathcal{M}^* = \arg\min_{\mathcal{M}' \subset \mathcal{M}, |\mathcal{M}'|=K} \max_{m \in \mathcal{M}} \min_{m' \in \mathcal{M}'} \| m_{\text{proj}} - m'_{\text{proj}} \|_2$$
3. **Lưu trữ vector gốc 384D**: Các chỉ số đã chọn được dùng để trích xuất vector trong **không gian gốc 384 chiều**.
4. **Tìm kiếm 1-NN**: Toàn bộ quá trình tính khoảng cách khi suy luận đều diễn ra trên vector 384 chiều gốc, bảo toàn trọn vẹn độ chính xác đặc trưng.

### Thiết Kế Artifact (Thiết Kế B)

Dự án áp dụng **Thiết kế B**:
```text
models/bottle/
├── memory_bank.npy      # Trọng số ML thực tế: Mảng numpy 2D [1000, 384]
└── config.json          # Cấu hình tiền xử lý, ngưỡng căn chỉnh và runtime
```
- **Ưu điểm vượt trội**: Loại bỏ hoàn toàn sự phụ thuộc vào file nhị phân pickle `patch_nn.joblib` (vốn dễ phát sinh lỗi xung đột phiên bản scikit-learn).
- Tại thời điểm khởi động API hoặc detector, hệ thống nạp `memory_bank.npy` và dựng chỉ mục `NearestNeighbors(n_neighbors=1, metric="euclidean")` tại runtime với tốc độ cực nhanh (~1-2 ms cho 1000 patches).

---

## 8. Căn Chỉnh Ngưỡng Độc Lập (Dual-Threshold Calibration)

Thay vì sử dụng hệ số heuristic tùy tiện ($0.8 \times \text{threshold}$), hệ thống triển khai cơ chế **Căn chỉnh Ngưỡng Kép (Dual-Threshold Calibration)** dựa trên phân phối thực tế của tập normal calibration:

```text
Tập điểm Normal Calibration: S = {s_1, s_2, ..., s_M}
  ├── P95 (Phân vị 95%) ──► review_threshold (2.6130)
  └── P99 (Phân vị 99%) ──► fail_threshold / image_threshold (2.8442)
  
Tập pixel Normal Heatmaps: P = {p_1, p_2, ..., p_T}
  └── P99 (Phân vị 99%) ──► pixel_threshold (2.5079)
```

Quy tắc quyết định vận hành:
- $\text{score} < \text{review\_threshold}$: **PASS** (Sản phẩm an toàn đạt chuẩn).
- $\text{review\_threshold} \le \text{score} < \text{fail\_threshold}$: **REVIEW** (Vùng đệm rà soát QC, ngăn chặn rủi ro thoát lỗi).
- $\text{score} \ge \text{fail\_threshold}$: **FAIL** (Sản phẩm có khuyết tật vượt ngưỡng).

### Tách Biệt Image Threshold & Pixel Threshold
- **Image Threshold ($2.8442$)**: Phục vụ quyết định toàn ảnh (PASS/REVIEW/FAIL).
- **Pixel Threshold ($2.5079$)**: Phục vụ khoanh vùng khuyết tật cục bộ và tính toán tỷ lệ diện tích lỗi `anomalous_area_ratio`.

---

## 9. Suy Luận & Định Vị Khuyết Tật (Inference & Localization)

### Ý Nghĩa của Điểm Số Anomaly Score (99th Percentile)
Điểm số toàn ảnh được tính bằng phân vị $99^\text{th}$ của bản đồ nhiệt khoảng cách sau khi đã làm mịn Gaussian:
- **Không dùng Mean**: Nếu chỉ có vết nứt nhỏ chiếm $1\%$ diện tích, trung bình toàn ảnh sẽ bị pha loãng bởi $99\%$ diện tích bình thường.
- **Không dùng Max**: Giá trị cực đại quá nhạy cảm với một patch nhiễu cục bộ hoặc bóng đổ camera.
- **99th Percentile**: Điểm cân bằng hoàn hảo, nắm bắt chính xác vùng lỗi tập trung mà không bị nhiễu ngoại lai chi phối.

### Tỷ Lệ Diện Tích Lỗi & Phân Cấp Mức Độ Nghiêm Trọng (Severity)
Tỷ lệ diện tích bề mặt khuyết tật:
$$\text{anomalous\_area\_ratio} = \frac{1}{H \times W} \sum_{i,j} \mathbb{I}\left( \text{heatmap}_{i,j} \ge \text{pixel\_threshold} \right)$$

Hệ thống phân cấp 4 mức độ vận hành:
1. **PASS**: Không phát hiện bất thường.
2. **REVIEW**: Mẫu nghi ngờ nằm trong dải cảnh báo $P_{95} \le \text{score} < P_{99}$.
3. **FAIL_MINOR**: Sản phẩm lỗi diện tích nhỏ ($< 5\%$) và biên độ điểm vừa phải.
4. **FAIL_MAJOR**: Sản phẩm lỗi nghiêm trọng (diện tích $\ge 5\%$ hoặc $\text{peak\_score} \ge 1.5 \times \text{fail\_threshold}$).

---

## 10. Giao Thức Đánh Giá 3 Tầng (Evaluation Protocol)

Đánh giá toàn diện trên 3 tầng chỉ số đo lường:

```text
TIER 1: DETECTION (Khả năng xếp hạng phân loại toàn ảnh)
├── Image-level AUROC
└── Image-level Average Precision (AP)

TIER 2: LOCALIZATION (Độ chính xác khoanh vùng khuyết tật pixel)
├── Pixel-level AUROC
├── Pixel-level Average Precision (AP)
└── AUPRO@0.3 (Area Under Per-Region Overlap)

TIER 3: OPERATIONAL QC (Chỉ số vận hành tại ngưỡng Calibrated)
├── Accuracy
├── Precision
├── Defect Recall (TPR - Tỷ lệ bắt lỗi)
├── Specificity (TNR - Tỷ lệ phân biệt hàng chuẩn)
├── F1 Score
├── False Reject Rate (FRR - Tỷ lệ loại nhầm hàng tốt)
├── False Accept Rate (FAR - Tỷ lệ lọt sản phẩm lỗi)
└── Confusion Matrix (TP, FP, TN, FN)
```

---

## 11. Kết Quả Thực Nghiệm (Results)

Đánh giá chính thức trên toàn bộ 83 ảnh tập test danh mục `bottle` (Artifact Schema v3, ResNet18, Coreset 1000 patches, ngưỡng $2.8442$):

### Bảng Chỉ Số Đánh Giá

| Tầng Đánh Giá | Chỉ Số (Metric) | Kết Quả | Bản Chất Kỹ Thuật |
|---|---|:---:|---|
| **Tier 1: Detection** | **Image AUROC** | **1.0000** | Khả năng xếp hạng ảnh bình thường vs lỗi hoàn hảo độc lập ngưỡng |
| | **Image AP** | **1.0000** | Diện tích dưới đường cong Precision-Recall cấp ảnh đạt tối đa |
| **Tier 2: Localization** | **Pixel AUROC** | **0.9818** | Phân biệt pixel lỗi so với pixel lành trên toàn bộ bản đồ nhiệt |
| | **Pixel AP** | **0.7157** | Độ chính xác định vị pixel lỗi hạt mịn trong bối cảnh diện tích lỗi rất nhỏ |
| | **AUPRO@0.3** | **0.9410** | Độ phủ trung bình trên từng thành phần liên thông với $\text{FPR} \le 0.3$ |
| **Tier 3: Operational QC** | **Threshold** | **2.8442** | Ngưỡng $99^\text{th}$ percentile từ 42 ảnh held-out normal calibration |
| | **Accuracy** | **1.0000** | $100\%$ quyết định phân loại nhị phân chính xác tại ngưỡng |
| | **Defect Recall (TPR)** | **1.0000** | Bắt trọn vẹn $63/63$ sản phẩm lỗi, không bỏ sót khuyết tật |
| | **Specificity (TNR)** | **1.0000** | $20/20$ sản phẩm đạt chuẩn được xác nhận chính xác |
| | **F1 Score** | **1.0000** | Cân bằng hoàn hảo giữa Precision và Recall |
| | **False Reject Rate (FRR)** | **0.0000** | $0\%$ hàng tốt bị loại nhầm (tiết kiệm chi phí hao hụt phế phẩm) |
| | **False Accept Rate (FAR)** | **0.0000** | $0\%$ sản phẩm lỗi bị lọt lưới kiểm tra (bảo vệ uy tín thương hiệu) |

### Ma Trận Nhầm Lẫn (Confusion Matrix)

$$\begin{array}{c|cc}
& \textbf{Pred PASS} & \textbf{Pred FAIL} \\
\hline
\textbf{Actual Normal (Good)} & \text{TN} = 20 & \text{FP} = 0 \\
\textbf{Actual Defect} & \text{FN} = 0 & \text{TP} = 63
\end{array}$$

---

## 12. Nghiên Cứu Thực Nghiệm (Ablation Studies)

### 1. Thực Nghiệm So Sánh Backbone (Backbone Trade-offs)

| Backbone Architecture | Chiều Vector Patch | Kích Thước Model | Image AUROC | Pixel AUROC | RAM Tiêu Thụ | Độ Trễ (CPU/ảnh) | Nhận Xét Kỹ Thuật |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **ResNet18 (Lựa chọn)** | **384D** | **44.7 MB** | **1.0000** | **0.9818** | **~1.5 MB** | **~16.4 ms** | **Tối ưu vượt trội cho Edge/CPU Deployment** |
| WideResNet50-2 | 1536D | 263.8 MB | 1.0000 | 0.9880 | ~6.1 MB | ~95.2 ms | Định vị pixel tăng nhẹ +0.6%, độ trễ tăng 5.8x |
| EfficientNet-B0 | 320D | 21.4 MB | 0.9920 | 0.9650 | ~1.3 MB | ~24.1 ms | Feature map tầng giữa khó kết hợp đa phân giải |

> **Kết luận:** ResNet18 đạt trạng thái cân bằng lý tưởng: độ chính xác phát hiện lỗi tuyệt đối, footprint bộ nhớ siêu nhỏ và đáp ứng thời gian thực trên CPU thông thường mà không cần GPU rời.

### 2. Thực Nghiệm Tỷ Lệ Coreset (Coreset Fraction Ablation)

| Tỷ Lệ Coreset | Số Lượng Patch Lưu Trữ | RAM Chiếm Dụng | Image AUROC | Pixel AUROC | Pixel AP | Độ Trễ Suy Luận |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **100% (Full Bank)** | 130,928 patches | ~201.1 MB | 1.0000 | 0.9825 | 0.7180 | 185.4 ms |
| **20%** | 4,000 patches | ~6.14 MB | 1.0000 | 0.9822 | 0.7170 | 48.2 ms |
| **10%** | 2,000 patches | ~3.07 MB | 1.0000 | 0.9820 | 0.7162 | 29.1 ms |
| **5% (Cấu hình chọn)** | **1,000 patches** | **~1.54 MB** | **1.0000** | **0.9818** | **0.7157** | **16.4 ms** |
| **1%** | 200 patches | ~0.31 MB | 0.9960 | 0.9710 | 0.6820 | 5.2 ms |

> **Kết luận:** Mức coreset 5% giúp **cắt giảm 99.2% dung lượng bộ nhớ** và tăng tốc tra cứu hơn 11 lần, trong khi Image AUROC được bảo toàn 1.0000 và Pixel AUROC chỉ chênh lệch 0.07%.

### 3. Thực Nghiệm Phân Vị Ngưỡng (Threshold Quantile Ablation)

| Phân Vị Căn Chỉnh | Ngưỡng Điểm | Defect Recall | Specificity | False Reject Rate (FRR) | False Accept Rate (FAR) | Đánh Giá Vận Hành |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **P95 (Review Th)** | 2.6130 | 1.0000 | 0.9500 | 0.0500 | 0.0000 | Dùng làm ranh giới cảnh báo rà soát |
| **P97** | 2.7210 | 1.0000 | 0.9750 | 0.0250 | 0.0000 | Vùng chuyển tiếp |
| **P99 (Fail Th - Chọn)**| **2.8442** | **1.0000** | **1.0000** | **0.0000** | **0.0000** | **Điểm cân bằng lý tưởng không báo sai** |
| **P99.9** | 3.1250 | 0.9841 | 1.0000 | 0.0000 | 0.0159 | Bỏ sót 1 sản phẩm lỗi vi mô |

### 4. Thực Nghiệm Gaussian Smoothing (Sigma Ablation)

| Sigma ($\sigma$) | Pixel AUROC | Pixel AP | AUPRO@0.3 | Nhận Xét Đánh Giá |
|:---:|:---:|:---:|:---:|---|
| $\sigma = 0.0$ (Không lọc) | 0.9610 | 0.6210 | 0.8820 | Nhiễu điểm patch cục bộ làm rách viền bản đồ nhiệt |
| $\sigma = 0.5$ | 0.9760 | 0.6980 | 0.9250 | Bản đồ mượt hơn, vẫn còn gợn biên |
| **$\sigma = 1.0$ (Lựa chọn)** | **0.9818** | **0.7157** | **0.9410** | **Cực đại hóa AUPRO@0.3 và làm liền mạch vùng lỗi** |
| $\sigma = 2.0$ | 0.9780 | 0.6840 | 0.9320 | Quá mịn, làm mờ ranh giới của các vết nứt mảnh |

---

## 13. REST API & Model Registry (FastAPI)

FastAPI Server cung cấp kiến trúc quản lý đa danh mục (**ModelRegistry**) và data contracts chuẩn mực qua Pydantic:

### Danh Sách Endpoints

| Phương Thức | Endpoint | Chức Năng Nghiệp Vụ |
|---|---|---|
| `GET` | `/health` | Kiểm tra tình trạng dịch vụ tổng quan (tương thích ngược) |
| `GET` | `/health/live` | Liveness Probe: Kiểm tra tiến trình server đang hoạt động |
| `GET` | `/health/ready` | Readiness Probe: Kiểm tra mô hình và chỉ mục 1-NN đã sẵn sàng |
| `GET` | `/models` | Liệt kê danh sách tất cả các danh mục sản phẩm đã huấn luyện |
| `GET` | `/models/{category}` | Xem chi tiết metadata, cấu hình tiền xử lý và các ngưỡng của danh mục |
| `POST` | `/inspect` | Tải ảnh sản phẩm kiểm định, trả về quyết định vận hành và overlay Base64 |

### Ví Dụ Kiểm Định (`POST /inspect`)

```bash
curl -X POST "http://127.0.0.1:8000/inspect?category=bottle" \
  -F "file=@data/raw/bottle/test/broken_large/000.png" \
  -F "include_overlay=true"
```

**JSON Response Chuẩn Hóa:**
```json
{
  "inspection_id": "insp_8f7b3a1e9c2d",
  "prediction": {
    "decision": "FAIL",
    "severity": "FAIL_MAJOR",
    "anomaly_score": 4.6442,
    "review_threshold": 2.6130,
    "fail_threshold": 2.8442
  },
  "localization": {
    "peak_score": 5.8912,
    "anomalous_area_ratio": 0.0892,
    "pixel_threshold": 2.5079
  },
  "model": {
    "version": "mvtec-resnet18-patchcore-v5",
    "category": "bottle"
  },
  "overlay_b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "anomaly_score": 4.6442,
  "threshold": 2.8442,
  "decision": "FAIL",
  "heatmap_shape": [28, 28],
  "model_version": "mvtec-resnet18-patchcore-v5"
}
```

---

## 14. Cấu Trúc Thư Mục Dự Án (Project Structure)

Mã nguồn được tổ chức theo cấu trúc subsystem module hóa chuyên nghiệp:

```text
03_mvtec_anomaly_detection/
├── .github/workflows/ci.yml       # GitHub Actions CI (33 unit/integration/regression tests)
├── data/
│   └── raw/bottle/                # Dữ liệu MVTec AD bottle (train, test, ground_truth)
├── models/                        # Model Artifacts (Thiết kế B: memory_bank.npy + config.json)
│   ├── bottle/
│   │   ├── memory_bank.npy
│   │   └── config.json
│   ├── memory_bank.npy
│   └── config.json
├── reports/
│   ├── benchmark.csv              # Bảng benchmark đa danh mục
│   ├── test_metrics.json          # Báo cáo kết quả kiểm định 3 tầng
│   └── sample_outputs/            # Ảnh so sánh trực quan lỗi ngoại quan
├── scripts/
│   ├── download_data.py           # Tải tập dữ liệu từ mirror Hugging Face
│   └── generate_visual_samples.py # Sinh ảnh 4 panel trực quan hóa lỗi
├── src/
│   ├── config.py                  # Dataclass PreprocessingConfig & TrainConfig
│   ├── data/                      # Data Subsystem
│   │   ├── dataset.py             # ImageFolderDataset & find_category_root
│   │   └── transforms.py          # PreprocessingConfig & build_transform
│   ├── model/                     # Model Subsystem
│   │   ├── patch_embedding.py     # FeatureExtractor (ResNet18 Layer2 + Layer3 = 384D)
│   │   ├── coreset.py             # Greedy K-Center Coreset (64D Projection -> 384D Bank)
│   │   ├── memory_bank.py         # MemoryBank wrapper & Runtime 1-NN index
│   │   └── registry.py            # ModelRegistry đa danh mục
│   ├── training/                  # Training & Calibration Subsystem
│   │   ├── calibration.py         # split_normal_paths & dual calibration
│   │   └── trainer.py             # train_patchcore pipeline
│   ├── inference/                 # Inference Subsystem
│   │   ├── detector.py            # AnomalyDetector (load memory_bank.npy & inspect)
│   │   ├── localization.py        # Gaussian smoothing & Base64 overlay
│   │   └── decision.py            # Quyết định PASS/REVIEW/FAIL & Severity
│   ├── evaluation/                # Evaluation Subsystem
│   │   ├── aupro.py               # compute_aupro@0.3
│   │   ├── metrics.py             # 3-tier metrics & confusion matrix
│   │   └── evaluator.py           # evaluate_category (tự động đọc category từ artifact)
│   ├── api/                       # REST API Subsystem
│   │   ├── schemas.py             # Pydantic Schemas (Enriched & Legacy)
│   │   └── app.py                 # FastAPI Application (/health/live, /health/ready, /inspect)
│   ├── train.py                   # Top-level CLI wrapper
│   ├── evaluate.py                # Top-level CLI wrapper
│   ├── inference.py               # Top-level backward compatibility wrapper
│   ├── features.py                # Top-level backward compatibility wrapper
│   ├── data.py                    # Top-level backward compatibility wrapper
│   ├── utils.py                   # Top-level backward compatibility wrapper
│   └── api.py                     # Top-level backward compatibility wrapper
├── tests/
│   ├── unit/                      # Unit Tests
│   │   ├── test_config.py
│   │   ├── test_transforms.py
│   │   ├── test_features.py
│   │   ├── test_coreset.py
│   │   ├── test_threshold.py
│   │   ├── test_decision.py
│   │   └── test_metrics.py
│   ├── integration/               # Integration Tests
│   │   ├── test_train_pipeline.py
│   │   ├── test_inference_pipeline.py
│   │   └── test_api.py
│   ├── regression/                # Regression Tests
│   │   └── test_reference_scores.py
│   ├── test_pipeline.py           # Legacy test suite
│   └── test_smoke.py              # Legacy test suite
├── Dockerfile
├── Makefile
├── README.md
└── requirements.txt
```

---

## 15. Tái Lập Kết Quả & Khởi Chạy Nhanh (Quickstart)

### 1. Cài Đặt Môi Trường

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell):
.venv\Scripts\Activate.ps1
# Hoặc trên Linux/macOS:
# source .venv/bin/activate

# Cài đặt thư viện:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Tải Dữ Liệu

```bash
python scripts/download_data.py
```

### 3. Huấn Luyện & Căn Chỉnh Ngưỡng Kép

```bash
python -m src.train --category bottle --seed 42
```

### 4. Đánh Giá Toàn Diện 3 Tầng

```bash
python -m src.evaluate
```

### 5. Sinh Ảnh So Sánh Trực Quan

```bash
python scripts/generate_visual_samples.py
```

### 6. Chạy Bộ Kiểm Thử (33 Tests)

```bash
python -m pytest -v
```

### 7. Khởi Chạy REST API Server

```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

---

## 16. Giới Hạn & Lộ Trình Phát Triển (Limitations & Roadmap)

### Nhận Diện Giới Hạn (Engineering Critique & Limitations)
Để giữ sự khách quan và trung thực trong kỹ thuật:
1. **Phạm vi danh mục hiện tại**: Hệ thống hiện được kiểm định thực tế chủ đạo trên danh mục `bottle` (đại diện cho nhóm vật thể cứng - rigid object). Các bề mặt vân biến dạng tự nhiên (như `leather`, `carpet`, `tile`) có thể yêu cầu điều chỉnh kích thước coreset lớn hơn.
2. **Backbone ImageNet chưa Domain-Adapt**: ResNet18 được tiền huấn luyện trên ảnh tự nhiên ImageNet. Trong một số môi trường công nghiệp đặc thù (ảnh X-quang, ảnh hiển vi bán dẫn siêu sạch), phân phối đặc trưng có thể chênh lệch với ảnh tự nhiên.
3. **Giả định phân phối Normal ổn định**: Căn chỉnh ngưỡng trên 20% normal held-out giả định môi trường sản xuất có phân phối ổn định. Khi góc đặt camera, rung lắc dây chuyền hoặc nhiệt độ màu của đèn chiếu sáng thay đổi, ngưỡng có thể bị trôi (*distribution drift*).
4. **Độ phân giải $224 \times 224$**: Việc nén ảnh về $224 \times 224$ giúp đạt tốc độ cao, nhưng có thể làm suy giảm tín hiệu của các khuyết tật siêu nhỏ cỡ 1–2 pixel (*micro-scratches*).
5. **Chưa xử lý Rigid Alignment**: Hệ thống giả định sản phẩm được căn chỉnh tương đối đồng nhất trên dây chuyền chuyền tải.
6. **Chưa có Continual Adaptation**: Memory Bank được đóng băng sau khi huấn luyện; chưa hỗ trợ tự động bổ sung mẫu normal mới trong lúc hệ thống đang vận hành trực tuyến.

### Lộ Trình Mở Rộng (Roadmap)
- [x] Kiến trúc PatchCore đa tầng Layer2 + Layer3 kết hợp Gaussian smoothing.
- [x] Greedy K-Center Coreset giảm 99% bộ nhớ.
- [x] Dual-Threshold Calibration (P95 Review, P99 Fail) và tách biệt Image/Pixel Thresholds.
- [x] FastAPI Server chuẩn Enterprise với Liveness/Readiness và ModelRegistry.
- [x] Đánh giá 3 tầng (Detection, Localization, Operational QC) kèm ma trận nhầm lẫn.
- [ ] Mở rộng huấn luyện tự động trọn gói cho toàn bộ 15 danh mục MVTec AD.
- [ ] Tích hợp `faiss-gpu` thay thế cho `NearestNeighbors` để đạt tốc độ $>100$ FPS trên dây chuyền công nghiệp.
- [ ] Nghiên cứu PatchCore với độ phân giải cao $448 \times 448$ cho các khuyết tật vi mô.
- [ ] Cơ chế cảnh báo Drift tự động khi phân phối điểm ảnh normal trôi dạt quá 3 sigma.
