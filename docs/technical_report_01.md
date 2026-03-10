# 1. Tổng quan Hệ thống

## Mục đích
Phát hiện hành vi Fear of Missing Out (FOMO) trong lịch sử giao dịch của các nhà đầu tư. 

## Kiến trúc Tổng thể
```
Raw Data → Feature Engineering → XGBoost Model → Dashboard
   ↓              ↓                  ↓               ↓
3 files      5 features          FOMO Score       Interactive UI
           (5-day windows)     + SHAP values    + Explainability
```

### Quy trình Hoạt động
1. **Thu thập dữ liệu**: Transactions, customer info, market prices
2. **Tạo cửa sổ**: Chia thành các cửa sổ giao dịch 5 ngày liên tiếp (sliding windows)
3. **Tính đặc trưng**: 5 behavioral features cho mỗi cửa sổ
4. **Gán nhãn**: Rule-based labeling (FOMO = 1, Non-FOMO = 0)
5. **Train model**: XGBoost classifier với explainability (SHAP)
6. **Phân tích**: Dashboard tương tác với visualizations

## Phân loại FOMO
- **Low** (<0.5%): Hành vi giao dịch bình thường
- **Medium** (0.5-3%): Có một số dấu hiệu FOMO
- **High** (>3%): Hành vi FOMO rõ rệt

---

# 2. Pipeline Xử lý Dữ liệu

## Dữ liệu đầu vào
| File | Nội dung | Columns chính |
|------|----------|---------------|
| `transactions.csv` | Lịch sử giao dịch | customerID, ISIN, timestamp, transactionType, totalValue, units |
| `customer_information.csv` | Thông tin khách hàng | customerID, ... |
| `close_prices.csv` | Giá đóng cửa thị trường | ISIN, timestamp, close_price |

## Chọn cửa sổ giao dịch (sliding windows)

Với dataset được cung cấp, dữ liệu giao dịch thưa và theo ngày (không liên tục), window 5 ngày là lựa chọn tối ưu dựa trên các lý do sau:

1. **Đủ dài để có giao dịch**: Giao dịch thưa → có thể 2-3 ngày không có giao dịch. Window 5 ngày tăng xác suất có ít nhất vài giao dịch để tính features đáng tin cậy.

2. **Đủ ngắn để signal không bị pha loãng**: 5 ngày = 1 tuần giao dịch, đủ compact để bắt được "burst" của FOMO behavior mà không trộn lẫn nhiều hành vi khác nhau.

3. **Phù hợp chu kỳ tâm lý**: Retail investors thường review portfolio theo tuần. 5 ngày giao dịch = 1 tuần làm việc, align với decision cycle tự nhiên.

4. **Balance data quality**: Window càng nhỏ → nhiều windows nhưng có thể thiếu data. Window càng lớn → ít windows và signal bị pha loãng. 5 ngày là điển tối ưu cân bằng giữa số lượng và chất lượng samples.

## Feature Engineering (5 đặc trưng chính)

1. **`n_trades`**: Tổng số giao dịch trong cửa sổ
   - Đo lường mức độ hoạt động giao dịch

2. **`n_buys`**: Số lượng lệnh mua
   - Phân biệt với SELL, quan trọng cho các features khác

3. **`avg_return_before_buy`**: Return trung bình trước khi mua
   - **Chỉ số đuổi giá** (price chasing indicator)
   - Return dương cao → mua sau khi giá đã tăng → FOMO
   - `NaN` nếu không có lệnh mua trong cửa sổ

4. **`buy_after_spike_ratio`**: Tỷ lệ mua sau khi giá tăng đột biến
   - Tính % lệnh mua có return > 2% (threshold)
   - **Chỉ số momentum chasing**
   - Giá trị cao → xu hướng mua khi thấy giá tăng mạnh

5. **`avg_missed_return`**: Return trung bình bỏ lỡ
   - Return trung bình của thị trường trong những ngày không giao dịch
   - **Chỉ số regret** - tiềm năng sinh FOMO trong tương lai
   - Return bỏ lỡ cao → cảm giác tiếc nuối → dễ FOMO lần sau

## Gán nhãn (Labeling)

**Quy tắc:** Một cửa sổ được gán nhãn **FOMO = 1** nếu thỏa mãn:
```python
avg_return_before_buy > 0.02 (2%)        # Mua sau khi giá tăng
AND buy_after_spike_ratio > 0.3 (30%)    # Phần lớn mua sau spike
AND avg_missed_return > 0.01 (1%)        # Có return bỏ lỡ
AND n_buys >= 1                           # Có ít nhất 1 lệnh mua
```

**Output:**
- `fomo_feature_data.csv`: Features không có nhãn
- `fomo_feature_label_data.csv`: Features + labels
- `fomo_train_data.csv`: Training set (80%)
- `fomo_test_data.csv`: Test set (20%)

---

# 3. Mô hình Machine Learning

## Thuật toán: XGBoost Classifier

**Lý do chọn XGBoost:**
- Xử lý missing values tự động
- Performance cao với tabular data
- Native support cho SHAP explainability
- Tree-based → interpretable (khi cần thiết)

## Model Evaluation

**1. Accuracy (Độ chính xác)**
- Đo lường overall performance của model dựa trên tỷ lệ predictions đúng trên tổng số predictions

**2. ROC-AUC (Area Under ROC Curve)**
- Đo lường khả năng phân biệt giữa FOMO và Non-FOMO
- **Metric quan trọng hơn Accuracy** cho bài toán này vì dữ liệu imbalanced (có thể phải đánh giá lại và nâng cấp quá trình gán nhãn)

## Output của Model

**1. FOMO Score (0-1)**
- Xác suất investor thể hiện hành vi FOMO trong cửa sổ đó
- Score càng cao → càng nhiều dấu hiệu FOMO

**2. FOMO Level**
- Low: score < 0.005 (0.5%)
- Medium: 0.005 ≤ score < 0.03 (3%)
- High: score ≥ 0.03 (3%)

**3. Model Certainty**
- Gần 1.0 → model rất chắc chắn (hoặc chắc chắn FOMO hoặc chắc chắn KHÔNG FOMO)
- Gần 0.0 → model không chắc chắn, score gần decision boundary

**4. Key Behavioral Signals**
- Top 2 features có SHAP value tuyệt đối cao nhất
- Cho biết features nào đóng góp nhiều nhất vào prediction

## SHAP Values (Explainability)

**SHAP (SHapley Additive exPlanations):**
- Giải thích đóng góp của từng feature vào prediction
- SHAP > 0: Feature làm tăng FOMO score
- SHAP < 0: Feature làm giảm FOMO score
- Magnitude: Mức độ ảnh hưởng

**Ví dụ:**
```
Feature: avg_return_before_buy = 0.05 (5%)
SHAP Value: +0.12
→ Feature này làm tăng FOMO score lên 0.12 điểm
```

**Trường hợp đặc biệt:** Feature = NaN vẫn có SHAP value
- SHAP value cho NaN = đóng góp của việc "thiếu dữ liệu"
- Ví dụ: avg_return_before_buy = NaN (không có buy) → SHAP âm → giảm FOMO

---

# 4. Dashboard

Dashboard cung cấp giao diện trực quan để xem kết quả phát hiện FOMO. Người dùng có thể lựa chọn model, lọc nhà đầu tư theo cấp độ FOMO, và phân tích chi tiết hành vi giao dịch trong từng cửa sổ với giải thích SHAP và so sánh với trung bình thị trường.

**Tính năng chính:**
- Chọn model đã train từ danh sách có sẵn
- Lọc nhà đầu tư theo cấp độ FOMO (Low/Medium/High)
- Lựa chọn hiển thị nhà đầu tư bằng nút Previous/Next hoặc nhập trực tiếp ID
- Xem phân tích FOMO chi tiết cho từng cửa sổ giao dịch
- Xem SHAP explanation cho các kết quả của model
- So sánh hành vi cá nhân với các mẫu với giá trị trung bình

---

# 5. Bảng tham khảo

## Thresholds Quan trọng

| Mục đích | Threshold | Giá trị | Ý nghĩa |
|----------|-----------|---------|---------|
| **Labeling** | avg_return_before_buy | > 2% | Mua sau khi giá tăng |
| | buy_after_spike_ratio | > 30% | Đa số mua sau spike |
| | avg_missed_return | > 1% | Có return bỏ lỡ |
| | min_buys | ≥ 1 | Có ít nhất 1 mua |
| **Classification** | Low FOMO | < 0.5% | Hành vi bình thường |
| | Medium FOMO | 0.5-3% | Có dấu hiệu FOMO |
| | High FOMO | > 3% | FOMO rõ rệt |

## Feature Summary

| Feature | Type | Range | NaN possible? | Ý nghĩa FOMO |
|---------|------|-------|---------------|--------------|
| `n_trades` | int | ≥ 0 | No | Càng nhiều → càng active |
| `n_buys` | int | ≥ 0 | No | 0 → không có lệnh mua |
| `avg_return_before_buy` | float | -1 to ∞ | **Yes** | Cao → đuổi giá → FOMO |
| `buy_after_spike_ratio` | float | 0-1 | No | Cao → mua sau spike → FOMO |
| `avg_missed_return` | float | -1 to ∞ | Yes | Cao → regret → dễ FOMO |

---

# 6. Hướng phát triển

## 1. Data & Features

### Thêm Features Mới
- **Volume-based features**: 
  - `avg_volume_spike`: Phát hiện mua khối lượng bất thường
  - `volume_to_avg_ratio`: So sánh với volume trung bình của investor
  
- **Temporal features**:
  - `time_since_last_trade`: Khoảng cách giữa các giao dịch (gap dài + đột ngột mua nhiều → FOMO)
  - `day_of_week`: FOMO có thể cao hơn vào đầu tuần sau khi xem news cuối tuần
  
- **Portfolio context**:
  - `diversification_score`: FOMO thường tập trung vào vài ISIN hot
  - `position_size_change`: Mua đột ngột tăng position size → FOMO

- **Market sentiment**:
  - `market_volatility`: Market-wide volatility trong window
  - `sector_momentum`: Return của sector trong cùng kỳ

### Improve Existing Features
- **Dynamic thresholds**: Thay vì fixed 2% cho spike, dùng percentile động theo từng ISIN
- **Weighted averages**: Weight giao dịch gần đây cao hơn trong window

## 2. Model Improvements

### Ensemble Methods
```python
# Thay vì chỉ XGBoost, dùng voting/stacking ensemble
- XGBoost (current)
+ LightGBM (faster, sometimes better)
+ CatBoost (better with categorical features)
→ Voting/Stacking ensemble
```

### Hyperparameter Tuning
- Research và áp dụng các kỹ thuật tự động tune hyperparameters
- Cross-validation strategy tốt hơn (TimeSeriesSplit thay vì random split)

### Multi-task Learning
```python
# Train đồng thời:
Task 1: FOMO vs Non-FOMO (binary)
Task 2: FOMO level (Low/Medium/High) (multi-class)
Task 3: Regret prediction (regression: dự đoán avg_missed_return tiếp theo)
```

### Hidden Markov Model (HMM)
- Model FOMO như các hidden states (Normal → Building FOMO → Full FOMO) để bắt temporal dependencies và cung cấp early warning: detect "Building FOMO" trước khi peak thay vì chỉ identify FOMO đã xảy ra.
- Phù hợp khi có ≥20 consecutive windows/investor, data gaps ≤10 ngày, và business cần cảnh báo sớm.

## 3. Window Design

### Adaptive Windows
```python
# Thay vì fixed 5 ngày:
- Window size dựa trên trading frequency của investor
- Active traders: 3 days
- Infrequent traders: 7-10 days
```

### Multi-scale Windows
```python
# Tính features cho nhiều window sizes:
features_3d = compute_features(window=3)
features_5d = compute_features(window=5)  # current
features_10d = compute_features(window=10)
→ Concatenate tất cả làm input cho model
```

## 4. Labeling Strategy

### Semi-supervised Learning
```python
# Current: Rule-based labeling (có thể thiếu chính xác)
# Improvement:
1. Label một phần data bằng rules
2. Train model sơ bộ
3. Dùng model predict unlabeled data với high confidence
4. Re-train với expanded dataset
```

### Label Smoothing
```python
# Thay vì hard labels (0 hoặc 1):
FOMO score continuous:
- Mạnh thỏa rules → label = 0.9
- Gần threshold → label = 0.6
- Rõ ràng không FOMO → label = 0.1
```

### Active Learning
- Cho expert review những cases model không chắc chắn (certainty < 0.5)
- Update labels → retrain → improve boundary cases

## 5. Dashboard Enhancements

### Real-time Monitoring
```python
# Thay vì batch analysis:
- Stream new transactions
- Update FOMO scores real-time
- Alert khi detect FOMO spike
```

### Personalized Insights
- "Investor X có xu hướng FOMO với sector nào?"
- "Pattern FOMO thường xảy ra vào thời điểm nào?"

## 6. Các nâng cấp khác
- Outlier Detection
- Feature Scaling
- Handling Imbalanced Data
- Automated Retraining Pipeline
- API Service
