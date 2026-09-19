# Wasserstein cho Slimmable Network — Tổng hợp và định hướng

*Cập nhật: 19/09/2026*

---

## 1. Trạng thái dự án hiện tại

### 1.1. Kết quả E2E (development seed 3, epoch 100)

| Method | Dense | Interior | Worst | Full |
|---|---|---|---|---|
| **Uniform** | **66.121** | **66.299** | 62.300 | 67.460 |
| SW-continuity | 65.915 | 66.049 | **62.440** | **67.520** |
| Resource | 65.508 | 65.647 | 62.300 | 66.760 |
| SW-anneal | 65.064 | 65.210 | 61.560 | 66.520 |
| Resource-Geo | 64.890 | 65.017 | 61.660 | 66.340 |
| Pure-SW | 64.581 | 64.724 | 61.160 | 66.000 |

**Không phương pháp nào thắng Uniform về dense mean.** Sau 5 phiên bản RQ2.

### 1.2. Năm formulation đã thử và thất bại

| Phiên bản | Ý tưởng | Vì sao hỏng |
|---|---|---|
| PureGeo-v1 | hard anchors theo SW | fixed K → trade-off zero-sum |
| FinalGeo-v1 | cân bằng geometry/resource | không giải được bottleneck cấu trúc |
| Dynamic-v2 | nonuniform unary marginals | đổi marginals = đổi target objective |
| Geo-HT-v3 | HT correction | variance tại checkpoint ≠ accuracy cuối |
| Pure-SW-Pair | LP trên 91 cặp | thấp nhất trong four-way |

Điểm chung: **cả năm đều dùng SW để điều khiển sampling.**

### 1.3. Ba nghịch lý chưa giải thích được

**(a) Variance đảo ngược hoàn toàn với accuracy.** Tại common E10:

```
V_SW = 126.05 < V_RG = 141.09 < V_R = 150.43 < V_U = 175.35
A_SW  <  A_RG  <  A_R  <  A_U      (epoch 100)
```

Thứ tự đảo ngược chính xác cả bốn. Với n=4 và một seed, chưa được nâng thành quy luật, nhưng đủ để nói objective variance **misaligned** với accuracy.

**(b) Raw Resource variance cao hơn Uniform 23–82%** trên fresh seed 6–7, mà vẫn thắng Pure-SW ở E2E.

**(c) SW dự báo gradient geometry rất tốt** (ρ 0.965–0.995) nhưng mọi cách dùng nó đều thua bốc ngẫu nhiên.

### 1.4. Điểm yếu thực nghiệm cần biết

| Vấn đề | Mức |
|---|---|
| Một seed, chênh lệch 0.206 pp | **rất cao** — chưa có σ |
| Không có định lý nối variance ↔ accuracy | **rất cao** |
| Loss `CE + KD` khác chuẩn US-Net (KD thuần) | cao |
| Chia ¼ gradient thay vì cộng dồn → effective LR nhỏ 4× | cao |
| 14 mức nội → mỗi mức chỉ 1/7 ngân sách | cao |
| Cầu SW ↔ gradient thuần thực nghiệm (có lúc ρ = −0.289) | cao |
| Optimizer reset ở E51 làm mọi nhánh tệ đi | vừa (đã nhận diện) |

---

## 2. Bối cảnh literature

### 2.1. Dòng slimmable / elastic

| Bài | Năm | Đóng góp | Số mức |
|---|---|---|---|
| Slimmable Networks | ICLR 2019 | switchable BN | 4 |
| **US-Net** | ICCV 2019 | sandwich rule + inplace KD + BN post-stats | **liên tục** |
| AutoSlim | 2019 | greedy per-layer channel search | ~10 nhóm/lớp |
| MutualNet | ECCV 2020 | width × resolution | liên tục |
| OFA | ICLR 2020 | progressive shrinking, 10¹⁹ subnet | rời rạc 4 chiều |
| BigNAS | ECCV 2020 | single-stage, không post-process | rời rạc |
| CompOFA | ICLR 2021 | ghép depth-width, bỏ PS | rời rạc |
| AttentiveNAS | CVPR 2021 | Pareto-aware sampling | rời rạc |
| **AlphaNet** | ICML 2021 | **alpha-divergence thay KL** | rời rạc |
| PSS-Net | TPAMI 2023 | subnet pool ưu tiên theo batch loss | bội 8/lớp |
| US3L | CVPR 2023 | 3 nguyên tắc loss cho SSL + group reg | 9 |
| ElasticViT | ICCV 2023 | ràng buộc chênh FLOPs, memory bank | 8 |
| HydraViT | NeurIPS 2024 | nested theo attention head | tới 10 |
| MatFormer | 2023 | FFN lồng nhau (Gemma 3n) | ~4 |
| **Slimmable ConvNeXt** | CVPR-W 2026 | LayerNorm bỏ được BN/KD/sandwich | 3–5 |

**Rời rạc là thông lệ.** Chỉ US-Net và MutualNet dùng liên tục.

### 2.2. Ba chỗ trống còn lại trong US-Net

| Chỗ | Trạng thái |
|---|---|
| Bất đẳng thức chặn (Eq. 3) chỉ đúng cho **một lớp ẩn** — tác giả tự thừa nhận | **trống hoàn toàn** |
| Chọn cận dưới k₀ — họ đo ảnh hưởng nhưng không có nguyên tắc chọn | **trống hoàn toàn** |
| "Nested thắng independent" — quan sát ở 4 bài, chưa ai giải thích | **trống hoàn toàn** |
| Chọn 2 width ở giữa | phần **ghép cặp** còn trống |
| Bất đối xứng tần suất kênh | US3L có lời giải (group reg), chưa ai **đo** |

### 2.3. Mẫu hình quan trọng: mọi cải tiến sampling chỉ giúp ở vùng thấp

| Bài | Đầu nhỏ | Đầu lớn |
|---|---|---|
| PSS-Net vs random | +1.1 (300M) | +0.1 (500M) |
| ElasticViT adjacent sampling | +3.3 (100M) | +0.0 (800M) |
| Slimmable ConvNeXt AutoSlim | +1.0 (p=0.25) | −0.4 (p=1.0) |

Ba bài, ba cơ chế khác nhau, cùng một hình dạng.

### 2.4. Ba bài phải đọc

1. **PSS-Net** (arXiv:2109.05432) — nêu vấn đề "homogeneity", nhưng theo nghĩa **giữa các lớp**, không phải giữa các subnet co-sampled. Chỉ bốc **1** width ở giữa → không có cặp. Họ tự nhận chưa có phân tích lý thuyết cho việc vì sao sampling của họ hoạt động.

2. **ElasticViT** (ICCV 2023) — kết luận **ngược** bạn: ghép subnet **cùng FLOPs** để giảm gradient conflict. Nhưng dải của họ là 86× (37M–3191M), của bạn ~10×. DRESS cho thấy ở không gian thuần-width nested, gradient tương quan **dương** — nên hai bên có thể ở hai chế độ khác nhau.

3. **AlphaNet** (ICML 2021) — **đã chiếm ô "đổi divergence trong supernet KD"**. KL khiến student đánh giá sai độ bất định của teacher; alpha-divergence sửa được, ăn 35% FLOPs. Baseline của bạn không phải KL mà là alpha-divergence.

---

## 3. Vì sao Uniform khó đánh bại

Ba tính chất mà mọi policy cấu trúc đều mất ít nhất một:

| Tính chất | Uniform | Resource | Pure-SW |
|---|---|---|---|
| Phủ đầy đủ (91/91 cặp) | ✓ | 7 cặp | 7 cặp |
| Không lệch | ✓ | ✗ | ✗ |
| Ổn định tuyệt đối | ✓ | ✓ | L1 tới 2.0 |

Và thứ tự sáu policy khớp gần hoàn hảo với "dày + ổn định" — hai biến **không liên quan gì tới nội dung SW**.

Tiền lệ trong literature:
- **SPOS** chọn uniform vì nó tách tối ưu trọng số khỏi chọn kiến trúc
- **Shipard & Wiliem** (CVPRW 2022): các chiến lược giảm interference gần như không đổi accuracy; Random Subnet Sampling **thắng** progressive shrinking trên 4 dataset nhỏ-vừa
- **CompOFA**: bỏ PS mà accuracy không đổi, tiết kiệm 50% thời gian

---

## 4. Định hướng: đổi vai của Wasserstein

### 4.1. Ba vai

| Vai | Đã thử | Kết quả | Còn trống |
|---|---|---|---|
| **Điều khiển sampling** | 5 lần | thua cả 5 | — |
| **Vào loss (KD)** | chưa | — | **có** |
| **Mô tả / phân tích** | RQ1 | **dương** | có |

**Vai 1 nên bỏ.** Không phải vì ý tưởng dở, mà vì đã có 5 bằng chứng liên tiếp và một nghịch lý cấu trúc (variance ↛ accuracy).

### 4.2. Vì sao bỏ vai 1 lại gỡ được nhiều ràng buộc

| Vai của SW | Cần lưới rời rạc? |
|---|---|
| Điều khiển sampling (Định lý 1) | **Có** — q định nghĩa trên tập cặp hữu hạn |
| Vào loss (KD) | **Không** |
| Mô tả (RQ1) | **Không** — G(c) là đạo hàm, càng mịn càng tốt |

Bỏ vai 1 → có thể quay về **US-Net chuẩn** (width liên tục, BN post-statistics, KD thuần). Gỡ được ba vấn đề cùng lúc:
- không còn câu hỏi "14 mức quá nhiều"
- không còn khác biệt protocol với literature
- đo được G(c) mịn hơn, tìm được nhiều cặp matched hơn

Cái mất: Định lý 1. Nhưng nó là lý thuyết cho một đại lượng mà dữ liệu cho thấy tương quan **âm** với accuracy.

---

## 5. Vai 2 — Wasserstein trong KD

### 5.1. Ba tầng dùng được

| Tầng | So cái gì | Cần ma trận chi phí | Chỗ trống | Lập luận |
|---|---|---|---|---|
| **Logit** | xác suất 100 lớp | có, 100×100 từ FC cuối | hẹp (AlphaNet) | vừa |
| **Feature** | đám mây 128 chiều | không (Euclid) | vừa | mạnh |
| **Ngang** | hai student cùng batch | tùy tầng | **cao** | **mạnh nhất** |

### 5.2. Tầng logit

KL chỉ so xác suất **cùng một lớp**. Wasserstein mang metric giữa các lớp.

Ví dụ: student nhầm "chó → sói" và student nhầm "chó → xe tải" bị KL phạt gần như nhau. Wasserstein phạt cái sau nặng hơn nhiều.

**Lợi thế riêng của slimmable:** teacher và student **dùng chung lớp FC cuối**, nên ma trận chi phí giữa các lớp là một và giống hệt nhau cho cả hai bên — không phải áp đặt từ ngoài như KD thông thường.

Cài đặt: Sinkhorn (entropic regularization) để có gradient mượt và chạy được trên GPU. Wasserstein chính xác cần giải LP, không song song hóa được.

### 5.3. Tầng feature

WKD-F lập luận: KL **không xử lý được phân phối không chồng lấn** và **không nhận biết hình học** manifold bên dưới.

Đây đúng là thứ RQ1 đo. Bạn đã có bằng chứng hình học ở tầng này mang thông tin thật (R² 0.05 → 0.55).

### 5.4. Tầng ngang — đóng góp chính

Hiện tại: mọi subnet chỉ học từ 1.0×. Quan hệ **dọc**.

```
hiện tại:     1.0 → 0.40,  1.0 → 0.70
đề xuất:      thêm  0.40 ↔ 0.70
```

**Lập luận mạnh nhất:** hai student cùng cấp không có quan hệ thầy-trò. KL bất đối xứng — dùng nó ở đây là **sai về nguyên tắc**, không chỉ là lựa chọn kém. Wasserstein đối xứng: W(a,b) = W(b,a).

Đây là một câu, không cần ba mắt xích như Định lý 1. Về lý thuyết sạch hơn hẳn.

Và nó dùng lại được trực giác pairwise mà không cần khung variance.

### 5.5. Thiết kế thí nghiệm

| Nhánh | Loss cho width nhỏ | Vai trò |
|---|---|---|
| A | KL dọc | baseline US-Net |
| B | alpha-divergence dọc | **baseline thật** — AlphaNet |
| C | W dọc | WKD |
| **D** | **W dọc + W ngang** | **đóng góp** |

**Bỏ qua B là hỏng.** Thắng KL mà không so với alpha-divergence thì không ai tin.

---

## 6. Vai 3 — hoàn thiện RQ1

### 6.1. Đã có (không cần train thêm)

- Specialization gap tại 0.40 gấp **3.2 lần** tại 0.60, dù **coverage bằng nhau** (0.10)
- G(0.40)/G(0.60) ≈ **2.45**, giữ 3/3 seed
- LOSO: R² từ 0.049 → **0.548**, MAE giảm 29.4%
- Robustness qua learned projection / backbone / random projection, ρ ≥ 0.99

### 6.2. Còn thiếu

**1. Nhiều cặp matched hơn.** Hiện chỉ có (0.40, 0.60). Tìm thêm 3–4 cặp cùng coverage khác G. Nếu tất cả cùng kết luận → bằng chứng có kiểm soát, mạnh hơn bất kỳ bảng R² nào.

**2. Một backbone thứ hai.** AutoSlim đã cảnh báo bằng số: cấu hình tối ưu trên ImageNet đem sang CIFAR10 cho **9.9 lỗi so với baseline 8.6** — tệ hơn. Cấu hình width phụ thuộc dataset nặng.

**3. Lời giải thích**, không chỉ tương quan.

### 6.3. Lưu ý về thung lũng

Bảng 2 (RQ1): 62.88 → 59.76 → **58.24** → 58.65 → 61.17 → 66.11
Bảng 24 (E2E, cột Uniform): 62.30 → 63.28 → 63.72 → 65.12 → 65.50 → 65.82

**Thung lũng biến mất trong run E2E.** Chênh lệch ở 0.35 là **5.5 điểm**. Hai bảng từ hai run khác nhau, nhưng điều này cần giải thích trước khi xây lý thuyết lên thung lũng.

Ba giả thuyết cho thung lũng:
- **(a) Tần suất sample** — sandwich rule train dư 0.25× và 1.0×; gián đoạn tần suất 7:1 tại neo
- **(b) Bất đối xứng kênh** — nhưng N(j) giảm trơn tru từ 4.00 xuống 1.14, không có bất thường ở 0.35–0.40 → **bác bỏ được bằng tính toán**
- **(c) Mốc làm tròn kênh của ResNet-18** — kiểm tra được không cần train

---

## 7. Cách ghép hai vai thành một bài

> Hình học biểu diễn cho thấy một số width phân hóa xa khỏi họ (RQ1). Đó là nơi việc dùng chung trọng số thiệt hại nhất. KD dựa trên f-divergence không nhìn thấy sự phân hóa này vì nó chỉ so từng lớp. Wasserstein nhìn thấy, và kéo chúng lại (KD).

RQ1 thành **chẩn đoán**, KD thành **cách chữa**. Wasserstein ở cả hai vai, không lần nào phải điều khiển sampling.

**Bằng chứng nối cần đo:** width nào có G(c) lớn thì cũng là width mà W(teacher, student) lớn hơn KL một cách bất thường. Đo được trên checkpoint đã có.

Nếu quan hệ đó tồn tại → một bài mạch lạc. Nếu không → tách thành hai bài.

---

## 8. Kế hoạch thực nghiệm

### 8.1. Ba phép thử rẻ, không cần GPU nhiều

| # | Việc | Quyết định gì |
|---|---|---|
| 1 | Đo KL / alpha-div / W giữa teacher và từng student; xem cái nào tương quan tốt hơn với accuracy thật | Vai 2 sống không |
| 2 | Đo hai width ở giữa hiện lệch nhau bao nhiêu (logit và feature) | Tầng ngang có đất không |
| 3 | Tìm các cặp width cùng coverage khác G trong lưới 16 mức | RQ1 mở rộng được không |

Cả ba vài giờ. Chúng quyết định toàn bộ hướng đi.

### 8.2. Việc bắt buộc trước mọi kết luận

**Đo σ.** Chạy Uniform × 3 seed.

| σ ước lượng | Hệ quả |
|---|---|
| ≲ 0.1 pp | mọi so sánh hiện tại có ý nghĩa |
| ≈ 0.3 pp | chỉ phân biệt được Pure-SW (1.5 pp); continuity/Resource nằm trong nhiễu |
| ≳ 0.5 pp | cần thiết kế lại quanh hiệu ứng lớn hơn |

Specialization gap của chính dự án biến thiên 0.73 / 2.71 / 3.15 qua ba seed tại cùng width — nên σ có thể không nhỏ.

Không có con số này, mọi gate tiếp theo có thể tiêu 4–6 lần train mà không kết luận được gì.

### 8.3. Sửa baseline

| Sửa | Vì sao |
|---|---|
| KD thuần thay `CE + KD` | US-Net đã thử `CE + KD` và **bác bỏ**; và cần loss sạch để đo tác động của W |
| Cộng dồn gradient thay chia ¼ | effective LR đang nhỏ hơn chuẩn 4 lần |
| BN post-statistics | chuẩn US-Net, và cho phép width liên tục |

---

## 9. Bảng quyết định

| Nếu... | Thì... |
|---|---|
| Phép thử 1 cho W tương quan tốt hơn alpha-div | đi vai 2, thiết kế 4 nhánh A/B/C/D |
| Phép thử 2 cho hai width ở giữa lệch đáng kể | tầng ngang (D) là đóng góp chính |
| Phép thử 3 tìm được 3+ cặp matched | RQ1 đứng một mình được |
| σ ≳ 0.3 pp | mọi kết quả E2E hiện tại chưa kết luận được |
| Muốn giữ Định lý 1 | phải giữ lưới rời rạc; nhưng cân nhắc 6 mức thay vì 14 |

---

## 10. Tham chiếu

**Slimmable / elastic**
- Slimmable Networks — arXiv:1812.08928 · github.com/JiahuiYu/slimmable_networks
- US-Net — arXiv:1903.05134
- AutoSlim — arXiv:1903.11728
- MutualNet — arXiv:1909.12978 · github.com/taoyang1122/MutualNet
- OFA — arXiv:1908.09791 · github.com/mit-han-lab/once-for-all
- BigNAS — arXiv:2003.11142
- CompOFA — arXiv:2104.12642
- AttentiveNAS — arXiv:2011.09011
- **AlphaNet** — arXiv:2102.07954 · github.com/facebookresearch/AlphaNet
- PSS-Net — arXiv:2109.05432 · github.com/chenbong/PSS-Net
- US3L — arXiv:2303.06870 · github.com/megvii-research/US3L-CVPR2023
- ElasticViT — arXiv:2303.09730
- HydraViT — OpenReview kk0Eaunc58 · github.com/ds-kiel/HydraViT
- MatFormer — arXiv:2310.07707
- Slimmable ConvNeXt — arXiv:2605.22677 · github.com/ds-kiel/Slimmable-ConvNeXt
- Shipard & Wiliem — arXiv:2204.09210

**Wasserstein KD**
- WCoRD (CVPR 2021) — arXiv:2012.08674 — bài đầu tiên đưa OT vào KD
- WKD (NeurIPS 2024) — Wasserstein cho cả logit và feature
- KD²M — khung thống nhất KD như khớp phân phối

**Gradient conflict trong supernet**
- NASViT (ICLR 2022) — nguồn gốc khái niệm
- Mixture-of-Supernets — arXiv:2306.04845
- GM-NAS — arXiv:2203.15207 — dùng cosine similarity để quyết định chia sẻ trọng số
- DYNAS — arXiv:2503.10740
- DRESS — arXiv:2207.00670 — gradient **tương quan dương** ở nested width/sparsity
- PA&DA (CVPR 2023) — arXiv:2302.14772

---

## Phụ lục: chi tiết code Slimmable ConvNeXt

Đọc từ `github.com/ds-kiel/Slimmable-ConvNeXt` (2048 dòng, phần slimming ~30 dòng).

**Chọn width là tuần hoàn, không ngẫu nhiên** — khác với bài báo:
```python
p_list_1 = p_list[data_iter_step % len(p_list)]
```

**Cơ chế slim** (`Block.forward`):
```python
p_out = round(self.dim * p_percent)
if p_out > x.shape[1]:
    x = torch.cat([x, zeros], dim=1)   # zero-pad
else:
    x = x[:, :p_out]                    # cắt NGAY TRÊN ĐẦU VÀO
```
Cắt trên đầu vào nghĩa là block hẹp **vứt bỏ vĩnh viễn** kênh của block trước. Đây là nút thắt khiến nonuniform chỉ được +1.0 pp.

**Downsample layer không bị slim** — luôn chạy full width.

**Loss**: chỉ `criterion(output, targets)`, không KD.

**AutoSlim base** (2 dòng): `[randint(10, 100)/100 for _ in range(36)]` — mỗi block một width độc lập.

**Greedy search**: bắt đầu từ toàn 1.0, giảm từng block 0.10, đánh giá trên **validation set**, giữ cái giảm ít nhất.
