# Group4 Lab 3 - Tái lập và phân tích Deep Plug-and-Play Cluster

## 1. Mục tiêu và phạm vi

Repo này được xây dựng để tái lập bài báo **Deep Plug-and-Play Cluster** trong khuôn khổ môn Data Mining, đồng thời đánh giá mức độ phù hợp giữa:

- pipeline **SCAN** của mã nguồn gốc,
- module **PnP split/merge** do nhóm tự cài đặt lại,
- và các kết quả thực nghiệm thu được trên môi trường local.

Phần backbone và quy trình huấn luyện SCAN được tái sử dụng từ repo gốc `Unsupervised-Classification` theo đúng định hướng của giảng viên. Phần Plug-and-Play clustering là thành phần nhóm tự thiết kế, tích hợp, chạy thực nghiệm, điều chỉnh tham số và tổng hợp kết quả trong repo này.

Quy ước tên trong các bảng:

- `SCAN`: kết quả baseline của paper hoặc local SCAN.
- `PnP (paper)`: kết quả "Ours" do bài báo công bố.
- `Ours (local)`: kết quả chạy trong repo này với SCAN + module PnP do nhóm cài đặt.

## 2. Toàn bộ công việc đã thực hiện trong repo

Nhóm đã hoàn thành các đầu việc chính sau:

1. **Tích hợp repo upstream**
- Bootstrap `third_party/Unsupervised-Classification`.
- Cố định upstream ở commit `952ec31eee2c38e7233d8ad3ac0de39bb031877a` để đảm bảo tính ổn định và khả năng lặp lại.

2. **Chuẩn bị pretrained weights và dữ liệu**
- Tự động tải các checkpoint pretext cho `cifar-10`, `cifar-20`, `stl-10`.
- Tự động tải checkpoint `moco_v2_800ep_pretrain` cho các thực nghiệm kiểu ImageNet.
- Hỗ trợ cache feature tại `data/scan_cache` để tránh trích xuất lại nhiều lần.

3. **Xây dựng pipeline dữ liệu thống nhất**
- Viết `src/scan_datasets.py` để đóng gói và chuẩn hóa các dataset:
  - `cifar-10`
  - `cifar-20` (CIFAR-100 gồm 20 superclass)
  - `stl-10`
  - `imagenet-10`
  - `beans` (mở rộng ngoài phạm vi paper)
- Bổ sung wrapper cho dataset Hugging Face và quy trình chia train/eval cho `imagenet-10`.

4. **Xây dựng pipeline trích xuất đặc trưng**
- Viết `src/data_pipeline.py` để:
  - nạp pretrained encoder phù hợp từng dataset,
  - trích xuất frozen features,
  - cache tensor đặc trưng,
  - khai phá nearest neighbors phục vụ SCAN loss.

5. **Cài đặt module PnP cục bộ**
- Viết `src/pnp_training.py` gồm:
  - `DynamicClusterHead`,
  - phép `split_cluster` và `merge_clusters`,
  - khởi tạo bằng `KMeans`,
  - quy trình train head với SCAN loss,
  - cơ chế cooldown sau split/merge,
  - heuristic split dựa trên variance nội cụm + 2-means bootstrap,
  - merge dựa trên Jensen-Shannon divergence.

6. **Xây dựng khung thực nghiệm và sinh bảng kết quả**
- Viết `src/experiment_pipeline.py` để chạy từng thiết lập và đối chiếu với kết quả paper.
- Viết `scripts/generate_reproduction_tables.py` để sinh:
  - Table 1: so sánh tổng thể
  - Table 2: inferred K
  - Table 3: tóm tắt ImageNet-50
  - Table 5: độ ổn định theo `K0`
  - Table 6: ablation theo `lambda`
  - Table 7: ablation split/merge
- Viết `scripts/generate_extension_tables.py` để sinh:
  - `table8_init_ablation.csv`
  - `additional_datasets/beans_comparison.csv`

7. **Tự động hóa quy trình chạy trên máy mới**
- `scripts/bootstrap_third_party.py`: clone/fetch repo gốc.
- `scripts/bootstrap_checkpoints.py`: tải checkpoint nếu thiếu.
- `scripts/prepare_and_run_report.py`: chuẩn bị môi trường, dữ liệu, cache feature trước khi sinh bảng.

8. **Notebook và tài liệu báo cáo**
- `notebooks/01_main_experiments.ipynb`: bảng chính và đối chiếu paper.
- `notebooks/02_ablation_study.ipynb`: các bảng ablation.
- `notebooks/03_additional_datasets.ipynb`: thực nghiệm mở rộng.
- `docs/reproduction_section_notes.md`: ghi chú cách trình bày reproduction section.
- `docs/Report.docx`: bản thảo báo cáo.

## 3. Cấu trúc repo

- `src/scan_datasets.py`: đóng gói dataset và transform đánh giá.
- `src/data_pipeline.py`: trích xuất feature và mine neighbor.
- `src/pnp_training.py`: train SCAN head + logic PnP.
- `src/divergence.py`: tính JS divergence và threshold merge.
- `src/metrics.py`: ACC, NMI, ARI.
- `src/experiment_pipeline.py`: tổ chức thực nghiệm, xuất bảng CSV.
- `scripts/`: các script chuẩn bị và sinh kết quả.
- `data/scan_results/`: toàn bộ kết quả tổng hợp phục vụ viết báo cáo.
- `third_party/Unsupervised-Classification/`: mã upstream được tái sử dụng.

## 4. Quy trình thực nghiệm hiện tại

### 4.1 Chuẩn bị môi trường và cache

```bash
python scripts/prepare_and_run_report.py
```

Script này sẽ:

- đảm bảo có `third_party/Unsupervised-Classification`,
- tải các checkpoint cần thiết nếu đang thiếu,
- trích xuất hoặc tái sử dụng feature cache,
- sẵn sàng hóa dữ liệu cho các bảng báo cáo.

### 4.2 Sinh bảng phục vụ reproduction

```bash
python scripts/generate_reproduction_tables.py --tables table1 table2 table3 table5 table6 table7
python scripts/generate_extension_tables.py
```

Kết quả được lưu tại:

- `data/scan_results/table1_main_comparison.csv`
- `data/scan_results/table1_paper_vs_local.csv`
- `data/scan_results/table2_inferred_k.csv`
- `data/scan_results/table3_imagenet50_summary.csv`
- `data/scan_results/table5_k0_stability.csv`
- `data/scan_results/table6_lambda_ablation.csv`
- `data/scan_results/table7_ablation_components.csv`
- `data/scan_results/table8_init_ablation.csv`
- `data/scan_results/additional_datasets/beans_comparison.csv`

## 5. Tóm tắt kết quả đã đạt được

### 5.1 Những điểm đạt được

- **Baseline SCAN local tái lập khá tốt** trên `cifar-20` và `stl-10`, và rất tốt trên `imagenet-10`.
- **PnP local với `K0` gần target** cho kết quả hợp lý hơn rõ rệt so với `K0` quá nhỏ.
- Pipeline đã đủ khả năng:
  - đối chiếu kết quả paper/local,
  - theo dõi độ lệch tương đối,
  - sinh bảng phục vụ viết báo cáo,
  - phân tích tác động của `K0`, `lambda`, split/merge và init strategy.

### 5.2 Số liệu nổi bật

Từ `table1_main_comparison.csv`:

- `imagenet-10` baseline SCAN local đạt `ACC = 96.77`, cao hơn mốc paper `92.0`.
- `cifar-20` baseline SCAN local đạt `ACC = 41.41`, sát paper `42.2`.
- `stl-10` baseline SCAN local đạt `ACC = 73.50`, sát paper `75.5`.
- `cifar-10` baseline SCAN local đạt `ACC = 74.94`, thấp hơn paper `81.8`.

Với PnP local:

- `imagenet-10, K0=20`: `ACC = 88.81`, khá gần paper `91.8`.
- `cifar-20, K0=30`: `ACC = 41.04`, gần paper `43.1`.
- `cifar-10, K0=20`: `ACC = 71.7`, thấp hơn paper `81.6`.
- `stl-10, K0=20`: `ACC = 57.74`, thấp hơn paper `74.7`.

## 6. Giải thích ý nghĩa số liệu trên các bảng

Phần này giải thích trực tiếp ý nghĩa của các bảng kết quả, để khi đưa vào báo cáo người đọc không chỉ thấy số mà còn hiểu số liệu đang chứng minh điều gì.

### 6.1 Bảng so sánh chính (Table 1)

Bảng `table1_main_comparison.csv` được dùng để trả lời hai câu hỏi:

- baseline SCAN local có tái lập được repo/paper hay không,
- và module PnP local có cải thiện hoặc ít nhất giữ được chất lượng như paper hay không.

Ý nghĩa của số liệu:

- Khi **SCAN local gần SCAN paper**, điều đó cho thấy pipeline dữ liệu, feature extractor và cách tính metric của nhóm là hợp lý.
- Khi **PnP local thấp hơn rõ so với PnP paper**, độ lệch chủ yếu nằm ở phần Plug-and-Play mà nhóm tự cài đặt, không nằm hoàn toàn ở SCAN.

Đọc bảng này có thể rút ra:

- `cifar-20` và `stl-10`: SCAN local khá sát paper, nên pipeline có thể xem là ổn.
- `imagenet-10`: SCAN local cao hơn paper, cho thấy feature extractor và chia dữ liệu local đang có lợi thế trong setting này.
- `cifar-10`: SCAN local thấp hơn paper một mức đáng kể, nên ngay từ baseline đã có khoảng cách cần lưu ý.
- PnP local chỉ giữ được mức khá gần paper khi `K0` khởi tạo đã gần với số cụm mục tiêu, ví dụ `imagenet-10, K0=20` hoặc `cifar-20, K0=30`.
- PnP local sụt mạnh khi `K0=3`, nghĩa là local implementation chưa tái lập được khả năng tự động điều chỉnh số cụm mạnh như paper công bố.

Nói ngắn gọn, Table 1 cho thấy:

- phần **SCAN đã tái lập ở mức chấp nhận được**,
- nhưng phần **PnP mới là nơi gây ra sai lệch chính** giữa local và paper.

### 6.2 Bảng inferred K (Table 2)

Bảng `table2_inferred_k.csv` cho biết sau quá trình split/merge, model local có hội tụ về số cụm đúng hay không.

Ý nghĩa của cột `K0` và `K*`:

- `K0`: số cụm khởi tạo ban đầu.
- `K*`: số cụm model suy ra ở cuối quá trình.

Nếu `K*` gần số cụm thật của dataset, có thể hiểu là cơ chế PnP đang hoạt động đúng hướng. Ngược lại, nếu `K*` lệch xa, thì split/merge chưa dẫn model về cấu trúc cụm mong muốn.

Đọc bảng này có thể thấy:

- `cifar-10, K0=20`: paper suy ra `10`, local suy ra `11` -> khá gần, nghĩa là merge có hoạt động một phần.
- `cifar-20, K0=30`: paper `19.8`, local `21` -> cũng khá gần.
- `stl-10, K0=20` và `imagenet-10, K0=20`: local `11`, gần target `10`.
- Nhưng với `K0=3`, local sai lệch lớn:
  - `stl-10`: `3 -> 3`, gần như không split được.
  - `imagenet-10`: `3 -> 3`, cũng cho thấy split thất bại.
  - `cifar-20`: `3 -> 8`, vẫn còn rất xa mức `~20`.

Bảng này thể hiện rất rõ:

- local PnP **làm tốt hơn trong bài toán giảm cụm** (`K0` lớn hơn target),
- nhưng **làm kém trong bài toán tăng cụm** (`K0` nhỏ hơn target).

### 6.3 Bảng độ ổn định theo `K0` (Table 5)

Bảng `table5_k0_stability.csv` được dùng để đánh giá tính robust của phương pháp khi số cụm khởi tạo thay đổi mạnh.

Nếu một phương pháp tốt, ACC không nên sập mạnh chỉ vì `K0` thay đổi. Đây là ý tưởng mà paper muốn chứng minh: PnP cần tự điều chỉnh được để kết quả ổn định hơn SCAN thường.

Số liệu đang nói lên:

- SCAN paper và SCAN local đều giảm khi `K0` lệch xa target, điều này là hợp lý vì SCAN không có cơ chế split/merge mạnh.
- PnP paper vẫn giữ được ACC cao trên nhiều giá trị `K0`, tức là paper chứng minh được tính robust với khởi tạo.
- Ở local implementation, PnP có cải thiện so với SCAN ở nhiều mốc, nhưng vẫn giảm rất mạnh khi `K0` quá lớn:
  - `K0=30`: local `51.13`, paper `82.0`
  - `K0=40`: local `36.48`, paper `77.5`
  - `K0=50`: local `26.63`, paper `70.9`
  - `K0=100`: local `12.13`, paper `72.4`

Ý nghĩa:

- Local PnP **chưa đạt được tính ổn định theo `K0`** như bài báo.
- Khi khởi tạo quá nhiều cụm, merge của local implementation không đủ sức gom các cụm lại về số cụm hợp lý.
- Nhưng ở các mốc trung gian như `K0=10`, `15`, `20`, local vẫn cho thấy PnP hữu ích hơn SCAN, nghĩa là hướng tiếp cận vẫn có giá trị.

### 6.4 Bảng ablation theo `lambda` (Table 6)

Bảng `table6_lambda_ablation.csv` được dùng để xem tham số `lambda` có thực sự điều khiển hành vi PnP hay không.

Trong paper:

- `lambda` thay đổi thì ACC thay đổi, nghĩa là threshold merge/split nhạy cảm với tham số này.

Trong local:

- `cifar-20`: tất cả giá trị `lambda` đều cho `ACC = 41.04`.
- `stl-10`: tất cả giá trị `lambda` đều cho `ACC = 57.7375`.

Ý nghĩa:

- Local implementation hiện tại **không phân biệt được tác động của `lambda`** trên kết quả cuối.
- Điều này gợi ý rằng `lambda` trong code local đang chưa tham gia đủ sâu vào động học PnP, hoặc số lần split/merge quá ít nên đổi `lambda` vẫn không đổi hành vi.

Nói cách khác, Table 6 không chứng minh rằng `lambda` vô nghĩa. Ngược lại, nó cho thấy implementation local chưa tái lập được cơ chế mà paper phân tích.

### 6.5 Bảng ablation thành phần split/merge (Table 7)

Bảng `table7_ablation_components.csv` trả lời câu hỏi: từng thành phần của PnP có đóng góp rõ ràng hay không.

Về mặt kỳ vọng:

- Nếu split và merge thật sự quan trọng, khi tắt từng thành phần thì metric phải thay đổi rõ.

Số liệu local cho thấy:

- `K0=3`: `No merge`, `No split loss (proxy)` và `Full method` cho cùng một kết quả.
- `K0=10`: tất cả các dòng ablation đều trùng nhau.
- `K0=20`: `No split` và `Full method` trùng nhau.

Ý nghĩa:

- Trong local implementation, nhiều ablation **chưa tạo ra sự khác biệt đủ rõ** về kết quả.
- Điều này thường xảy ra khi:
  - split/merge ít khi được kích hoạt,
  - hoặc logic hiện tại chưa đủ nhạy cảm để mỗi thành phần thể hiện tác động riêng.

Do đó, Table 7 cần được diễn giải theo hướng:

- nó có giá trị **chẩn đoán implementation**,
- nhưng chưa đủ sức làm bằng chứng thực nghiệm mạnh như Table 7 trong paper.

### 6.6 Bảng so sánh paper - local (`table1_paper_vs_local.csv`)

Bảng `table1_paper_vs_local.csv` không chỉ cho thấy metric, mà còn cho thấy **độ lệch tương đối** giữa local và paper.

Ý nghĩa của cột `Relative deviation(%)`:

- giá trị càng nhỏ -> local càng sát paper,
- giá trị càng lớn -> local càng lệch khỏi mức paper.

Có thể đọc nhanh:

- SCAN local trên `cifar-20` và `stl-10` có độ lệch nhỏ, cho thấy phần tái lập baseline là khá ổn.
- PnP local với `K0=20` hoặc `30` có độ lệch vừa phải trên một số dataset, nghĩa là implementation đã hoạt động một phần.
- PnP local với `K0=3` có độ lệch rất lớn, nhiều mức trên `30%` đến `67%`, thể hiện đây là nhóm thực nghiệm thất bại rõ nhất.

Bảng này rất hợp để đưa vào báo cáo vì nó biến nhận xét định tính thành một chỉ số rõ ràng: local đang cách paper bao xa.

## 7. Các thực nghiệm không đạt mức của bài báo và giải thích

Đây là phần quan trọng để viết báo cáo B: nhóm không chỉ liệt kê kết quả, mà cần phân tích rõ vì sao một số thực nghiệm không đạt mức bài báo.

### 7.1 Nhóm thực nghiệm PnP với `K0` rất nhỏ (`K0=3`) thất bại trên hầu hết dataset

Quan sát:

- `cifar-10, K0=3`: `ACC 55.86` so với paper `82.4`, `K* = 8` thay vì xấp xỉ `10`.
- `cifar-20, K0=3`: `ACC 27.49` so với paper `43.8`, `K* = 8` thay vì `19.7`.
- `stl-10, K0=3`: `ACC 28.75` so với paper `74.5`, `K* = 3` thay vì `9.7`.
- `imagenet-10, K0=3`: `ACC 29.81` so với paper `91.2`, `K* = 3` thay vì `10`.

Giải thích:

- Local PnP **không mở rộng cụm đủ nhanh** từ một `K0` quá nhỏ lên số cụm thực.
- Logic split hiện tại dựa trên **variance heuristic** thay vì tái lập đầy đủ cơ chế split trong paper.
- Nếu split không kích hoạt sớm, model bị khóa trong trạng thái cluster coarse, dẫn tới pseudo-label yếu và head học sai hướng.
- Sau mỗi lần split, cooldown dài làm quá trình điều chỉnh tiếp theo chậm, trong khi số epoch hữu dụng không còn nhiều.

Kết luận:

- Bài toán từ `K0=3` lên `K*=10` hoặc `20` là điểm yếu lớn nhất của local implementation.
- Đây là bằng chứng rõ ràng cho thấy module PnP của nhóm **chưa tái lập được cơ chế tăng cụm mạnh như paper**.

### 7.2 Độ ổn định theo `K0` chưa đạt được như paper

Quan sát từ `table5_k0_stability.csv` trên `cifar-10`:

- Paper giữ `ACC` khoảng `70-82%` trên nhiều giá trị `K0`.
- Local implementation giảm mạnh khi `K0` lớn:
  - `K0=30`: `ACC = 51.13`, `K*=17`
  - `K0=40`: `ACC = 36.48`, `K*=27`
  - `K0=50`: `ACC = 26.63`, `K*=37`
  - `K0=100`: `ACC = 12.13`, `K*=87`

Giải thích:

- Cơ chế merge hiện tại **không đủ mạnh để đưa số cụm về gần target** khi khởi tạo quá nhiều cụm.
- Khi `K0` lớn, phân bố xác suất cụm ban đầu manh mún, JS divergence giữa các cụm không đủ nhỏ để merge xảy ra liên tục.
- Sau đó head tiếp tục học trên cấu trúc cụm phân mảnh, làm suy giảm ACC/ARI rõ rệt.

Kết luận:

- Local implementation nhạy cảm cao với `K0`.
- Paper thể hiện tính robust với `K0`, còn bản cài đặt hiện tại thì chưa.

### 7.3 Ablation theo `lambda` gần như không thay đổi kết quả local

Quan sát từ `table6_lambda_ablation.csv`:

- `cifar-20`: mọi giá trị `lambda` đều ra `ACC = 41.04`.
- `stl-10`: mọi giá trị `lambda` đều ra `ACC = 57.7375`.

Giải thích:

- Trong code hiện tại, `lambda` chỉ ảnh hưởng trực tiếp đến **merge threshold**.
- Tuy nhiên, nếu quá trình train của local run không rơi vào vùng mà merge threshold thay đổi hành vi, thì kết quả cuối sẽ gần như giống nhau.
- Điều này cho thấy local implementation hiện tại **chưa khai thác đủ độ nhạy cảm của `lambda`** mà paper đã báo cáo.

Kết luận:

- Đây là dấu hiệu cho thấy biến `lambda` trong repo này có tác động thực tế thấp hơn kỳ vọng lý thuyết.
- Nhóm nên nêu rõ trong báo cáo rằng Table 6 local phản ánh một hạn chế của implementation, không phải bằng chứng rằng `lambda` không quan trọng.

### 7.4 Ablation split/merge chưa tách bạch rõ như paper

Quan sát từ `table7_ablation_components.csv`:

- Với `K0=3`, các dòng `No merge`, `No split loss (proxy)` và `Full method` cho cùng một kết quả local.
- Với `K0=10`, tất cả ablation đều ra cùng một kết quả local.
- Với `K0=20`, `No split` và `Full method` trùng nhau.

Giải thích:

- Các toggle ablation trong local code chưa tạo ra **hành vi động học đủ khác biệt** như trong paper.
- Mục `No split loss (proxy)` thực chất là một **phép gần đúng**, không phải tái lập đúng từng thành phần loss của bài báo.
- Nếu split/merge không xảy ra hoặc xảy ra rất ít, các ablation khác nhau trên danh nghĩa nhưng không khác nhau về kết quả.

Kết luận:

- Bảng ablation local có giá trị chẩn đoán implementation, nhưng chưa đủ mạnh để tái lập lập luận khoa học của paper một cách trọn vẹn.

### 7.5 ImageNet-50 chưa được reproduction đầy đủ

Quan sát:

- `table3_imagenet50_summary.csv` mới tổng hợp kết quả paper.
- Local status đang là `not_run_missing_imagenet50_dataset`.

Nguyên nhân:

- Không có sẵn bộ dữ liệu `imagenet-50` trong workspace hiện tại.
- Chi phí dữ liệu và tài nguyên tính toán cao hơn các tập con còn lại.

Kết luận:

- Đây là giới hạn dữ liệu/thiết bị, không phải lỗi thuật toán đơn thuần.

## 8. Lý do có thể khiến performance của PnP giảm xuyên suốt quá trình cài đặt

### 8.1 Độ lệch giữa implementation hiện tại và paper

Đây là nguyên nhân lớn nhất. Local code đang sử dụng một số lựa chọn thực dụng để chạy được trên repo:

- split dựa trên variance nội cụm,
- subcluster được tạo bằng `KMeans(n_clusters=2)`,
- KMeans init cho head,
- cooldown bất đối xứng sau split/merge,
- một số ablation được mô phỏng bằng proxy.

Những lựa chọn này hợp lý về kỹ thuật, nhưng có thể **không trùng khớp với cơ chế học và định nghĩa loss đầy đủ trong paper**. Vì vậy, kết quả PnP local giảm là điều dễ hiểu.

### 8.2 PnP phụ thuộc mạnh vào chất lượng feature frozen

- Toàn bộ split/merge được thực hiện trên feature đã trích xuất từ backbone có sẵn.
- Nếu feature chưa tách cụm tốt trên dataset cụ thể, PnP sẽ ra quyết định trên một không gian biểu diễn chưa thuận lợi.
- PnP không thể "cứu" được feature representation nếu backbone đã có giới hạn.

### 8.3 Cân bằng giữa split và merge rất nhạy cảm

- Split quá ít: model bị under-clustering.
- Split quá nhiều: model bị over-fragmentation.
- Merge quá ít: không quay về số cụm mục tiêu.
- Merge quá nhiều: mất cấu trúc cụm đúng.

Chỉ cần cooldown, threshold, variance gate, hoặc init sai lệch nhỏ là động học đã có thể thay đổi lớn.

### 8.4 Nhạy cảm với `K0`

Kết quả local cho thấy:

- `K0` quá nhỏ thì không tăng cụm kịp.
- `K0` quá lớn thì merge không gom lại kịp.

Điều này làm cho PnP local hiện tại chưa đạt tính robust với khởi tạo mà paper báo cáo.

### 8.5 Giới hạn tài nguyên và thời gian chạy

- Để tránh thời gian train quá lớn, local run phải cân đối giữa số epoch và phạm vi thực nghiệm.
- Mỗi thay đổi trong split/merge đều cần chạy lại nhiều lần mới xác định được xu hướng.
- Không có nhiều lần lặp với seed khác nhau như một reproduction quy mô lớn.

### 8.6 Khác biệt nguồn dữ liệu và cách chia dữ liệu

- `imagenet-10` trong repo được nạp từ Hugging Face và chia bằng `StratifiedShuffleSplit`.
- Điều này tiện lợi cho việc chạy lại, nhưng vẫn có thể khác thiết lập dữ liệu nguyên bản của paper.
- Khác biệt nhỏ ở distribution cũng có thể dẫn đến chênh lệch metric.

## 9. Các khó khăn nhóm gặp phải

- Khó xác định đầy đủ chi tiết implementation của PnP nếu paper không công bố toàn bộ mã nguồn trung gian.
- Việc ghép PnP vào SCAN cần tránh phá vỡ pipeline gốc trong khi vẫn phải giữ khả năng lặp lại.
- Cần tự động hóa việc tải checkpoint, cache feature, sinh bảng để tránh thao tác thủ công.
- Quá trình tuning rất tốn thời gian do cần cân đối giữa `K0`, `lambda`, init strategy, cooldown, số epoch và batch size.
- Một số bảng trong paper để tái lập cần dataset lớn hoặc setup khó, đặc biệt là `ImageNet-50`.

## 10. Đánh giá tổng kết cho báo cáo phần B

Nếu viết thành nhận xét tổng kết, có thể phát biểu ngắn gọn như sau:

> Nhóm đã tái sử dụng thành công SCAN pipeline, xây dựng được module PnP cục bộ, tự động hóa quy trình reproduction và sinh đầy đủ các bảng kết quả chính phục vụ báo cáo. Tuy nhiên, local implementation của PnP mới đạt kết quả tốt khi `K0` nằm gần số cụm mục tiêu, trong khi các thiết lập khó hơn như `K0=3` hoặc `K0` rất lớn vẫn chênh lệch rõ so với bài báo. Nguyên nhân chính đến từ độ lệch implementation so với paper, độ nhạy cảm cao của split/merge và giới hạn tài nguyên khi tuning.

## 11. Tệp kết quả nên được ưu tiên trích vào báo cáo

- `data/scan_results/table1_main_comparison.csv`
- `data/scan_results/table1_paper_vs_local.csv`
- `data/scan_results/table2_inferred_k.csv`
- `data/scan_results/table5_k0_stability.csv`
- `data/scan_results/table6_lambda_ablation.csv`
- `data/scan_results/table7_ablation_components.csv`
- `data/scan_results/table8_init_ablation.csv`
- `data/scan_results/additional_datasets/beans_comparison.csv`

## 12. Ghi chú trung thực học thuật

Repo này phân biệt rõ:

- thành phần upstream được tái sử dụng,
- thành phần nhóm tự cài đặt,
- kết quả paper,
- và kết quả local.

Điều này quan trọng để báo cáo đúng mức độ đóng góp và tránh đồng nhất giữa "tái lập bài báo" với "lặp lại y nguyên kết quả paper".
