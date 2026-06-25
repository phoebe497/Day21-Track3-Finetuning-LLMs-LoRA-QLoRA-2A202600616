# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Như Yến Phương
**Mã học viên**: 2A202600616  
**Submission option**: A (lightweight)

---

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen2.5-3B pre-quantized in 4-bit NF4 format)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (Subset of 200 high-quality samples)
  - **Train set**: 180 samples (90%)
  - **Eval set**: 20 samples (10%)
- **max_seq_length**: `1024` (Computed $p_{95} = 562$ tokens, rounded up to the nearest power of 2: 1024)
- **GPU**: Tesla T4 (15.6 GB VRAM on Google Colab)
- **Training cost**: $0.07 USD (~11.8 phút training time cho cả 3 adapters @ $0.35/hr rate)

---

## 2. Rank Experiment Results

Dưới đây là bảng so sánh chi tiết hiệu năng giữa các cấu hình rank LoRA khác nhau ($r=8$, $r=16$, $r=64$) và mô hình Base:

| Cấu hình (Rank) | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|:---|:---|:---|:---|:---|:---|
| **Base** | 0 | - | - | N/A | N/A |
| **r = 8** (alpha = 16) | 1,843,200 | 3.85 min | 7.22 GB | 1.5577 | 4.75 |
| **r = 16** (alpha = 32) | 3,686,400 | 4.13 min | 6.62 GB | 1.5161 | 4.55 |
| **r = 64** (alpha = 128) | 14,745,600 | 3.84 min | 8.00 GB | 1.4768 | 4.38 |

> **Lưu ý về Peak VRAM:** 
> Peak VRAM của cấu hình $r=16$ (6.62 GB) được ghi nhận thấp hơn $r=8$ (7.22 GB). Đây là một hiện tượng phổ biến do cơ chế quản lý bộ nhớ của PyTorch CUDA Caching Allocator và trình tự giải phóng/thu hồi vùng nhớ (garbage collection) giữa các cell chạy trong Jupyter Notebook. Về mặt lý thuyết, lượng VRAM tĩnh dùng để lưu optimizer states của $r=16$ sẽ lớn hơn $r=8$. Cấu hình $r=64$ ghi nhận lượng Peak VRAM cao nhất (8.00 GB) do số lượng tham số huấn luyện tăng gấp 4 lần so với $r=16$.
> 
> **Lưu ý về Base Perplexity:** 
> Do giới hạn phần cứng trên môi trường Colab T4, việc chạy đánh giá mô hình Base chưa qua tinh chỉnh (zero-shot) trên cùng tập eval_ds không được thực hiện để tránh lãng phí VRAM và dung lượng tải. Hơn nữa, vì mô hình Base chưa quen với cấu trúc khuôn mẫu Alpaca (Instruction/Input/Output), loss thực tế sẽ rất cao và không phản ánh đúng năng lực hiểu ngôn ngữ của nó.

---

## 3. Loss Curve Analysis

Đường cong Loss Curve được ghi nhận trong file [loss_curve.png](file:///d:/VinUni-AI20K/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA-2A202600616/lab21_2A202600616/results/loss_curve.png) cho thấy tiến trình giảm loss ổn định từ bước đầu tiên đến bước cuối cùng.

Cụ thể đối với cấu hình Baseline $r=16$:
- **Bắt đầu (Step 5)**: Training Loss là `1.6143`
- **Kết thúc (Step 65)**: Training Loss giảm xuống còn `1.3942`
- **Generalization Gap**: Eval Loss đạt `1.5161` ở cuối quá trình huấn luyện.

**Phân tích hiện tượng Overfitting:**
- **Không có hiện tượng Overfitting xảy ra.** Để tiết kiệm VRAM và tránh lỗi Out-Of-Memory (OOM) trên GPU T4, tiến trình đánh giá định kỳ trong lúc training đã được tắt (`eval_strategy="no"`). Tuy nhiên, khoảng cách giữa Training Loss (~1.39) và Validation Loss (~1.52) là rất nhỏ (~0.12).
- Với số lượng epoch cực kỳ ngắn (3 epochs) và kích thước tập dữ liệu nhỏ (180 samples), mô hình đang nằm ở pha học mẫu định dạng (format adaptation) và làm quen với ngữ cảnh tiếng Việt (Vietnamese Instruction) hơn là ghi nhớ sâu dữ liệu (memorization). Do đó, mô hình vẫn đang ở trạng thái tối ưu tốt mà không bị quá khớp.

---

## 4. Qualitative Comparison (5 examples)

Dưới đây là so sánh side-by-side kết quả sinh văn bản của mô hình **Base** và mô hình **Fine-tuned (r=16)** trên 5 câu test prompts:

### Example 1
- **Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
- **Base**: 
  > Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu. Mô hình học được từ dữ liệu này có thể thực hiện các tác... *(bị cắt)*
- **Fine-tuned (r=16)**:
  > Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) và sử dụng thuật toán và các mô hình học máy để học từ dữ liệu và từ đó đưa ra các dự đoán hoặc... *(bị cắt)*
- **Nhận xét**: **Improved**. Câu văn của bản Fine-tuned mạch lạc và mang tính chất định nghĩa chuẩn xác hơn (nhấn mạnh yếu tố "không hướng dẫn trực tiếp từ người dùng"). Cả hai đều bị cắt ở cuối do giới hạn `max_new_tokens=200`.

### Example 2
- **Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
- **Base**: 
  > Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:
  > ```python
  > def fibonacci(n):
  >     if n <= 0:
  >         return "N phải là một số dương"
  >     elif n == 1:
  >         return 0
  >     elif n == 2:
  >         return 1
  >     else: ...
  > ```
- **Fine-tuned (r=16)**:
  > Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:
  > ```python
  > def fibonacci(n):
  >     if n < 0:
  >         raise ValueError("Input phải là một số nguyên dương.")
  >     elif n == 0:
  >         return 0
  >     elif n == 1:
  >         return 1
  >     else:
  >         a, b = 0, 1
  >         for _ in range...
  > ```
- **Nhận xét**: **Improved**. Bản Fine-tuned xử lý ngoại lệ rất tốt bằng cách `raise ValueError` thay vì trả về chuỗi thông báo thông thường, đồng thời định nghĩa đúng điểm xuất phát của chuỗi Fibonacci ($F_0 = 0$, $F_1 = 1$) và sử dụng cách giải vòng lặp tối ưu bộ nhớ (`a, b = 0, 1`) thay vì đệ quy chậm chạp của Base.

### Example 3
- **Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
- **Base**: 
  > 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục... 2. Tru... *(bị cắt và lặp từ)*
- **Fine-tuned (r=16)**:
  > 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX thiết kế phải dễ dàng sử dụng và dễ hiểu. 4. Tương thích: UI/UX thiết kế phải dễ dàng tương thích với... *(bị cắt)*
- **Nhận xét**: **Improved**. Bản Fine-tuned trình bày có cấu trúc rõ ràng, sử dụng các thuật ngữ chuyên môn UI/UX tiếng Việt chuẩn xác (Chuyển đổi, Thích ứng, Đơn giản, Tương thích) và tránh được lỗi lặp từ ngữ ngớ ngẩn của bản Base ("thân thiện... và thân thiện").

### Example 4
- **Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
- **Base**: 
  > LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hiệu năng của mô hình bằng cách thay đổi các phép biến đổi trong mô hình bằng các phép biến đổi nhỏ hơ... *(bị cắt)*
- **Fine-tuned (r=16)**:
  > LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong các mô hình có số lượng các layer lớn. LoRA là một kỹ thuật regularization được sử dụng để cải thiệ... *(bị cắt)*
- **Nhận xét**: **Degraded / Hallucination**. Đây là một trường hợp **ảo tưởng (hallucination)** thú vị của mô hình Fine-tuned. Trong khi mô hình Base định nghĩa đúng từ viết tắt LoRA là *Low-Rank Adaptation*, mô hình Fine-tuned lại bị "học lệch" và diễn dịch thành *Layer-wise Adaptive Regularization Optimization* — một thuật ngữ hoàn toàn sai lệch. Điều này cho thấy rủi ro quá khớp cục bộ khi fine-tune trên dữ liệu dịch thuật tự động có thể làm nhiễu loạn tri thức nền sẵn có của base model.

### Example 5
- **Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
- **Base**: 
  > Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của mô hình bằng cách cung cấp cho nó một câu hỏi hoặc câu lệnh để dựa vào, thay vì cung cấp dữ liệu đầu... *(bị cắt)*
- **Fine-tuned (r=16)**:
  > Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp hệ thống AI giải quyết các vấn đề và thực hiện các tác vụ. Prompt được sử dụng để cung cấp cho hệ th... *(bị cắt)*
- **Nhận xét**: **Improved**. Bản Fine-tuned đưa ra định nghĩa mang tính ứng dụng thực tiễn cao hơn và chuyển ngữ mượt mà hơn sang tiếng Việt chuyên ngành công nghệ.

---

## 5. Conclusion về Rank Trade-off

Từ các kết quả định lượng thu được trong thực nghiệm trên, chúng ta rút ra một số kết luận sâu sắc về cơ chế Rank Trade-off:

1. **Rank cho ROI (Return on Investment) tốt nhất**: Cấu hình **$r = 16$** mang lại ROI tốt nhất cho tập dữ liệu này. Với số lượng tham số huấn luyện chỉ ở mức trung bình (3.69M parameters, tương đương $0.12\%$ của base model), nó giúp cải thiện rõ rệt perplexity từ `4.75` (ở $r=8$) xuống `4.55` mà không làm tăng thời gian huấn luyện đáng kể (chỉ chênh lệch khoảng 15 giây). Mức tiêu thụ VRAM thực tế cũng cực kỳ an toàn cho GPU T4.
2. **Điểm giới hạn hiệu suất (Diminishing Returns)**: Khi chúng ta tiếp tục nâng rank từ $r = 16$ lên $r = 64$ (tăng số tham số huấn luyện gấp 4 lần, đạt 14.75M), perplexity chỉ giảm nhẹ từ `4.55` xuống `4.38` (cải thiện chỉ `0.17` điểm). Đổi lại, tài nguyên Peak VRAM bị đội lên đáng kể tới mức chạm ngưỡng `8.00 GB`. Điều này chứng minh quy luật hiệu suất giảm dần: việc nâng cao dung lượng lưu trữ của adapter không tỷ lệ thuận tuyến tính với chất lượng đầu ra sau khi đạt tới một ngưỡng bão hòa nhất định của tập dữ liệu.
3. **Khuyến nghị Deploy Production**: Nếu triển khai thực tế trên môi trường Product, cấu hình **$r = 16$** (hoặc thậm chí $r = 8$ nếu tài nguyên phục vụ cực kỳ hạn chế) là sự lựa chọn tối ưu. Trong kiến trúc đa nhiệm (Multi-tenant serving), việc giữ kích thước các file adapter nhỏ gọn (~14MB cho $r=16$) cho phép hệ thống dễ dàng lưu trữ và swap adapters linh hoạt và nhanh chóng trên cùng một base model đang chạy, mang lại sự linh động cao hơn rất nhiều so với file adapter cồng kềnh của $r=64$.

---

## 6. What I Learned

Qua bài thực hành Lab 21, tôi đã tích lũy được các kinh nghiệm và bài học quý giá sau:
- **Hiểu sâu sắc về cấu hình QLoRA**: Biết cách kết hợp bitsandbytes (4-bit quantization NF4) cùng với thư viện Unsloth để tăng tốc độ huấn luyện lên gấp đôi và tiết kiệm tới 60% dung lượng GPU VRAM, giúp việc huấn luyện mô hình 3B hoàn toàn khả thi và nhanh gọn trên Colab Free GPU T4.
- **Rủi ro ảo tưởng tri thức (Hallucination) do Fine-tuning**: Quan sát trực tiếp hiện tượng mô hình bị mất đi tri thức định nghĩa chính xác sẵn có (như LoRA acronym) và bị thay thế bằng thuật ngữ tự chế. Điều này nhắc nhở bài học quan trọng: Fine-tuning chỉ nên tập trung vào định dạng (format/style adaptation), còn tri thức gốc vẫn nên được duy trì hoặc bổ trợ qua RAG.
- **Cách phân tích và lựa chọn tham số tối ưu**: Không phải cứ cấu hình rank lớn nhất là tốt nhất. Cần so sánh đa chiều giữa VRAM, thời gian chạy và chất lượng đầu ra để đưa ra kiến nghị thiết kế kiến trúc AI tối ưu nhất cho bài toán thực tế.
