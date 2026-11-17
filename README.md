# ImageRetrieval

## 1. Cài đặt

Cần Python 3.11–3.12 và Node.js 22+. Chạy PowerShell tại thư mục repo:

```powershell
git clone https://github.com/UIT-Entropy/ImageRetrieval.git
cd ImageRetrieval
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r source/image_retrieval/requirements-api.txt
Copy-Item .env.example .env
npm --prefix frontend ci
npm --prefix frontend run build
python source/image_retrieval/download_data.py
```

Lệnh tải data lấy bộ Castle5000 đã xử lý (5.000 ảnh, metadata, object/color và embedding) vào thư mục local. Không cần chạy lại indexing nếu dùng artifact public hiện có.

## 2. Chạy model trên GPU laptop

Local dùng multi-index đã dựng ở mục 3.2; không nạp Qwen8B. Cài PyTorch CUDA phù hợp driver/GPU, sau đó:

```powershell
py -3.11 -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
py -3.11 -m pip install -r source/image_retrieval/requirements-gpu.txt
```

Trong `.env`, đặt `IMAGE_MULTI_INDEX` tới index local và giữ `IMAGE_LOCAL_MODEL_CACHE=2`. Qwen2B dùng NF4 4-bit; SigLIP2, EVA-CLIP và DFN-CLIP chạy FP16. Local chỉ cung cấp Qwen2B và các model nhỏ; Qwen8B chỉ có trong Kaggle profile.

```powershell
python source/image_retrieval/serve.py --profile local --smoke
python source/image_retrieval/serve.py --profile local
```

Mở **http://127.0.0.1:8100/images**. Backend giữ tối đa hai model trong VRAM và tự giải phóng model cũ nếu thiếu bộ nhớ; đổi model lần đầu có thể cần thời gian nạp. Nếu gặp CUDA OOM, đặt `IMAGE_LOCAL_MODEL_CACHE=1` rồi khởi động lại.

## 3. Chạy API local, dùng GPU Kaggle

API, SQLite/Qdrant và UI chạy local; model chạy trên Kaggle GPU. Cần Kaggle API token và quyền GPU. Chỉ dùng token của chính bạn: điền `KAGGLE_USERNAME` và `KAGGLE_API_TOKEN` trong `.env`. Không commit `.env`.

### 3.1. Chọn cấu hình 2 hoặc 3 API

- **2 API:** slot 1 chạy Qwen3-VL 8B embedding + reranker; slot 2 chạy Qwen3-VL 2B embedding + reranker và SigLIP2, EVA-CLIP, DFN-CLIP.
- **3 API:** slot 1 chạy Qwen3-VL 8B; slot 2 chạy Qwen3-VL 2B embedding + reranker; slot 3 chạy SigLIP2, EVA-CLIP, DFN-CLIP.

Mỗi slot dùng một tài khoản Kaggle. Điền thông tin theo suffix: slot 1 dùng `KAGGLE_USERNAME`/`KAGGLE_API_TOKEN`, slot 2 dùng `_2`, slot 3 dùng `_3`. Notebook private bật Internet và GPU T4; lệnh push sẽ tạo notebook mới mang tên ngẫu nhiên trong tài khoản tương ứng.

### 3.2. Tạo index cho các embedding model

Chỉ cần làm một lần, hoặc làm lại khi đổi bộ ảnh/model. Index được tạo từ dataset Castle5000, không cần tải ảnh lên từng notebook:

```powershell
python source/image_retrieval/manage_workers.py push --index --slot 1
```

Đợi notebook Kaggle hoàn tất. Lấy `owner/notebook-slug` từ link được in ra rồi tải output và dựng index local:

```powershell
kaggle kernels output owner/notebook-slug -p source/image_retrieval/output/multi-embeddings
python source/image_retrieval/build_index.py --artifacts source/image_retrieval/output/image_retrieval_output --embeddings source/image_retrieval/output/multi-embeddings/embeddings --output source/image_retrieval/output/multi-index
```

Nếu notebook embedding đã chạy trước đó, bỏ qua bước này. Nếu cần tiếp tục từ output checkpoint, truyền `--resume owner/notebook-slug` vào lệnh `push --index`.

Trong `.env`, trỏ `IMAGE_MULTI_INDEX` đến index vừa tạo, ví dụ:

```dotenv
IMAGE_MULTI_INDEX=source/image_retrieval/output/multi-index
```

### 3.3. Push và kết nối worker

Chọn **một** topology, đặt `IMAGE_GPU_WORKERS` trong `.env` bằng 2 hoặc 3, rồi chạy đúng số slot:

```powershell
$workerCount = 2 # đổi thành 3 nếu dùng topology 3 API
1..$workerCount | ForEach-Object {
    python source/image_retrieval/manage_workers.py push --workers $workerCount --slot $_
}
1..$workerCount | ForEach-Object {
    python source/image_retrieval/manage_workers.py connect --workers $workerCount --slot $_
}
```

`connect` đợi worker khởi động, kiểm tra model rồi lưu kết nối cục bộ. Khi notebook/tunnel hết hạn, chạy lại `push` và `connect` cho slot đó.

### 3.4. Mở web và chọn model

```powershell
npm --prefix frontend run build
python -m source.image_retrieval --profile multi
```

Mở **http://127.0.0.1:8100/images**. Sau khi mở web hoặc reconnect worker, chạy thử mỗi model định dùng khoảng 2–3 lượt để làm ấm kết nối. Lượt đầu có thể chậm do worker/tunnel vừa khởi động; các lượt lặp cùng model và query sẽ nhanh hơn nhờ cache embedding local. Ctrl+S mở tùy chọn model tìm kiếm, ensemble, số kết quả Top K và reranker K. Khi đổi từ 2 sang 3 API (hoặc ngược lại), cập nhật `IMAGE_GPU_WORKERS`, kết nối đủ slot rồi khởi động lại backend.

Đo p50, p95 và Recall@20 trên bộ query gán nhãn bằng:

```powershell
python source/image_retrieval/benchmark_search.py queries.json
```

Mỗi mục trong file JSON có `query` và danh sách `image_ids` đúng nếu đã gán nhãn. Lưu ý: với Kaggle, thời gian encode một query warm thường dưới một giây; độ trễ còn dao động theo tunnel. Ensemble gửi yêu cầu đồng thời, nhưng các model đặt chung một GPU sẽ lần lượt dùng GPU đó.

## 4. Gắn Codex cho chat

```powershell
npm install -g @openai/codex
codex login
codex login status
```

Đăng nhập ChatGPT có quyền dùng `gpt-6-luna`. Backend tự dùng Codex CLI khi gửi chat; không cần OpenAI API key. Nếu CLI không nằm trên PATH, điền `CODEX_QUERY_EXECUTABLE` trong `.env` và khởi động lại API.


## 5. Tìm ảnh với nhiều model trên Kaggle

Dùng Python 3.11 hoặc 3.12. Cài lại requirements API sau khi pull để có Qdrant.
Trong `.env`, đặt `IMAGE_GPU_WORKERS=2` hoặc `3`, điền Kaggle username và token theo slot.
Slot 1 dùng `KAGGLE_USERNAME` và `KAGGLE_API_TOKEN`; slot 2 và 3 dùng hậu tố `_2` và `_3`.

Chạy job nhúng lại ảnh trước khi chạy các worker phục vụ tìm kiếm.

```powershell
python source/image_retrieval/manage_workers.py push --index --slot 1
```

Notebook dùng bộ Castle5000 và xuất thư mục `embeddings` sau khi nhúng đủ năm model.
Xem slug trong đầu ra lệnh push, rồi tải output notebook vào một thư mục local.
Nếu đã có output checkpoint, thêm `--resume owner/notebook-slug` khi gửi job mới.

```powershell
kaggle kernels output owner/notebook-slug -p source/image_retrieval/output/multi-embeddings
python source/image_retrieval/build_index.py --artifacts source/image_retrieval/output/image_retrieval_output --embeddings source/image_retrieval/output/multi-embeddings/embeddings --output source/image_retrieval/output/multi-index
```

Đặt `IMAGE_MULTI_INDEX=source/image_retrieval/output/multi-index` trong `.env`.
SQLite và Qdrant chạy local qua API, với index được chọn lúc backend khởi động.
Khởi động lại backend sau khi tạo bản index mới. Muốn quay lại luồng cũ, bỏ biến này và dùng profile cũ.

```powershell
$workerCount = 2
foreach ($slot in 1..$workerCount) {
    python source/image_retrieval/manage_workers.py push --workers $workerCount --slot $slot
}
foreach ($slot in 1..$workerCount) {
    python source/image_retrieval/manage_workers.py connect --workers $workerCount --slot $slot
}
npm --prefix frontend run build
python -m source.image_retrieval --profile multi
```

Nếu dùng ba API, đặt cả `$workerCount` và `IMAGE_GPU_WORKERS` thành `3` trước khi chạy. Khởi động lại backend khi đổi số API. Trong app, nhấn Ctrl+S để chọn Model tìm kiếm, Ensemble và Reranker.
Khi tunnel hết hạn, chạy lại push và connect cho slot đó. Backend đọc kết nối mới ở query tiếp theo.

Đo p50, p95 và Recall@20 bằng `python source/image_retrieval/benchmark_search.py queries.json`.
Dùng cùng 20 query cho hai cấu hình. Mỗi mục có `query` chứa điều kiện tìm ảnh và
`image_ids` chứa danh sách ảnh đúng nếu đã gán nhãn. Log khởi động của mỗi worker ghi thời gian warm-up và peak VRAM theo GPU.

## Cú pháp tìm nhanh cho team

Nhấn **Ctrl+A** để mở trợ lý. Ngăn các điều kiện bằng dấu phẩy, chấm phẩy hoặc xuống dòng.

| Cú pháp | Bộ lọc |
| --- | --- |
| `2 người` hoặc `đúng 2 người` | Đúng hai người |
| `ít nhất 2 người`, `tối đa 3 người` | Từ hai người hoặc tối đa ba người |
| `hơn 2 người`, `dưới 3 người` | So sánh nghiêm ngặt, lần lượt từ ba người và tối đa hai người |
| `thêm áo đỏ`, `2 chai xanh dương` | Màu gắn với chính object được đếm |
| `không có chai` | Số chai phát hiện bằng 0 |
| `ngày 1`, `day1` | Lọc ngày nếu ngày đó có trong dữ liệu |
| `lúc 9h`, `từ 8h đến 10h`, `8h-10h` | Lọc theo giờ, gồm cả hai đầu mốc |
| `bỏ màu`, `bỏ giờ`, `bỏ ngày` | Xóa nhóm điều kiện tương ứng |

Ví dụ có thể copy: **đúng 2 người, thêm áo đỏ, ngày 1, từ 8h đến 10h**.
Bộ lọc nhanh dùng số đếm detection và giữ các điều kiện hiện tại ngoài phần vừa sửa.
Có thể dùng tên nhãn tiếng Anh trong bộ lọc hoặc alias tiếng Việt. Màu xanh
nên ghi rõ **xanh lá** hoặc **xanh dương**. Các câu mô tả hành động, câu có điều kiện
ngoài cú pháp hoặc cần số đếm metadata gốc được chuyển nguyên câu sang Luna.
