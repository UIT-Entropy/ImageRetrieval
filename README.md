# ImageRetrieval

Tìm ảnh bằng Qwen3-VL-Embedding-8B, object/màu/thời gian và Qwen3-VL-Reranker-2B. Hỗ trợ feedback, chat và DRES.

## 1. Cài đặt

Cần Python 3.11–3.12 và Node.js 22+. Các lệnh dưới dùng PowerShell, chạy tại thư mục repo.

```powershell
git clone https://github.com/UIT-Entropy/ImageRetrieval.git
cd ImageRetrieval
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r source/image_retrieval/requirements-api.txt
Copy-Item .env.example .env
npm --prefix frontend ci
npm --prefix frontend run build
```

Cấu trúc backend được gom trong `source/image_retrieval/` và có entry point module:

```powershell
python -m source.image_retrieval --profile local --smoke
```

Điều này giúp chạy API theo kiểu package mà vẫn giữ nguyên cách chạy file script cũ như `python source/image_retrieval/serve.py`.

Nếu chọn GPU Kaggle, điền `KAGGLE_API_TOKEN` và `KAGGLE_USERNAME` của bạn trong `.env` (Kaggle → Settings → API). `IMAGE_ENCODER_URL`/`IMAGE_ENCODER_KEY` được tạo tự động. Tải data không cần token; không commit `.env`.

## 2. Tải data đã xử lý

[ZIP trên GitHub Releases](https://github.com/UIT-Entropy/ImageRetrieval/releases/tag/data-v1) · [Dataset public trên Kaggle](https://www.kaggle.com/datasets/quands/image-retrieval-kitchen-5k-cc3d917a) · khoảng 1,5 GB.

```powershell
python source/image_retrieval/download_data.py
```

Lệnh tải và kiểm tra checksum, đặt ảnh vào `data/`, artifacts vào `source/image_retrieval/output/image_retrieval_output/`. Bộ tải sẵn có 5.000 ảnh, metadata, embedding 4.096 chiều, 111.085 object/mask và màu. Không cần chạy lại preprocessing.

## 3A. Dùng GPU laptop

```powershell
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r source/image_retrieval/requirements-gpu.txt
python source/image_retrieval/serve.py --profile local --smoke
python source/image_retrieval/serve.py --profile local
```

Mặc định 4-bit, nạp embedding/reranker tuần tự. Điều chỉnh `QWEN_GPU_MEMORY`, `QWEN_CPU_MEMORY` trong `.env` theo máy; chạy `--smoke` để kiểm tra trước.

## 3B. Dùng GPU Kaggle

API/UI chạy local; embedding và reranker chạy trên notebook Kaggle private, Internet, **2 × T4**. Tài khoản cần bật quyền dùng GPU.

```powershell
python source/image_retrieval/manage_encoder.py push
python source/image_retrieval/manage_encoder.py connect
python source/image_retrieval/serve.py --profile kaggle --smoke
python source/image_retrieval/serve.py --profile kaggle
```

Mỗi người dùng notebook trong tài khoản Kaggle của mình. Phiên GPU/tunnel tối đa 2 giờ; hết phiên thì push/connect lại.

Mở **http://127.0.0.1:8100/images**. Chưa có DRES thì chọn **Luyện tập**.

## 4. Gắn Codex cho chat

```powershell
npm install -g @openai/codex
codex login
codex login status
```

Đăng nhập ChatGPT có quyền dùng `gpt-6-luna`. Backend tự dùng Codex CLI khi gửi chat; không cần OpenAI API key. Nếu CLI không nằm trên PATH, điền `CODEX_QUERY_EXECUTABLE` trong `.env` và khởi động lại API.

## Phím tắt và tìm ảnh

- `Ctrl+S`: tìm thủ công. `Ctrl+A`: bật/tắt chat.
- **Qwen reranker**: rerank top 20, cần text hoặc ảnh query.
- Rê ảnh: like/dislike/submit. Inspector: `←`/`→`, **Tất cả frame**.
- DRES: server URL → user/password hoặc Session ID → Evaluation ID. Submit nhanh dùng imageID làm media item.

## Kiểm tra

```powershell
python -m unittest discover -s source/image_retrieval -p "test_*.py" -q
npm --prefix frontend test
```
