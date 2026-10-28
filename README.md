# ImageRetrieval

Tìm ảnh bằng Qwen3-VL-Embedding-8B, object/màu/thời gian, feedback và Qwen3-VL-Reranker-2B. Có chat chỉnh query và nộp bài DRES.

## Chuẩn bị

Python 3.11–3.12, Node.js 22+, ảnh gốc trong `data/`, output tiền xử lý trong `source/image_retrieval/output/image_retrieval_output/`. Dữ liệu, model weights và credentials không nằm trong Git.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r source/image_retrieval/requirements-api.txt
Copy-Item .env.example .env
npm --prefix frontend ci
npm --prefix frontend run build
```

## Cách 1 — GPU laptop

```powershell
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r source/image_retrieval/requirements-gpu.txt
python source/image_retrieval/serve.py --profile local --smoke
python source/image_retrieval/serve.py --profile local
```

Mặc định NF4 4-bit, ngân sách GPU 4 GiB; embedding/reranker nạp tuần tự. Đổi `QWEN_GPU_MEMORY`, `QWEN_CPU_MEMORY`, `QWEN_QUANTIZATION` trong `.env` nếu cần. Bản 4-bit giữ model/revision/4096 chiều, có sai số so với FP16. **RTX 3050 6 GB chưa được xác nhận bằng smoke test model thật.** Lần đầu cần tải model; chỉ dùng cấu hình này khi `--smoke` đạt.

## Cách 2 — GPU Kaggle

API/UI vẫn ở local; embedding + reranker chạy trên notebook **private, Internet, 2 × T4**. Điền `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME` trong `.env`.

```powershell
python source/image_retrieval/manage_encoder.py push
python source/image_retrieval/manage_encoder.py connect
python source/image_retrieval/serve.py --profile kaggle --smoke
python source/image_retrieval/serve.py --profile kaggle
```

`connect` lưu URL/key vào `.env`. Phiên GPU/tunnel kéo dài tối đa 2 giờ; hết phiên thì push/connect lại. Reranker gửi các ảnh ứng viên lên notebook private để chấm điểm.

Mở **http://127.0.0.1:8100/images**. Chưa có DRES thì chọn **Luyện tập**.

## Gắn Codex cho chat

Cài [Codex CLI](https://developers.openai.com/codex/cli/) trên máy chạy API:

```powershell
npm install -g @openai/codex
codex login
codex login status
```

Đăng nhập bằng tài khoản ChatGPT có quyền dùng `gpt-6-luna`. Nếu máy đã đăng nhập Codex CLI thì chỉ cần kiểm tra `codex login status`. Backend tự gọi CLI khi gửi chat (`Ctrl+A`), dùng được với cả hai profile, không cần thêm OpenAI API key vào `.env`.

Nếu API không tìm thấy CLI, đặt `CODEX_QUERY_EXECUTABLE` thành đường dẫn đầy đủ tới executable Codex trong `.env`, rồi khởi động lại API. Chưa cài/đăng nhập Codex vẫn dùng được tìm thủ công, embedding và reranker.

## Tiền xử lý trên Kaggle

```powershell
python source/image_retrieval/manage.py prepare --data data
python source/image_retrieval/manage.py upload-dataset
python source/image_retrieval/manage.py upload-notebook
python source/image_retrieval/manage.py status
python source/image_retrieval/manage.py download
```

## Dùng nhanh

- `Ctrl+S`: tìm thủ công. `Ctrl+A`: chat; cần Codex CLI đã đăng nhập để gọi Luna.
- Bật **Qwen reranker** trong form: rerank top 20 sau khi lọc, rồi chia trang. Cần text/ảnh query; like/dislike vẫn điều chỉnh bước lấy ứng viên.
- Rê ảnh: like/dislike/submit. Trong inspector: `←` / `→`, **Tất cả frame**.
- DRES: server URL → user/password hoặc Session ID → chọn Evaluation ID. Submit nhanh dùng imageID làm media item; có thể chỉnh tên trong inspector.

## Kiểm tra

```powershell
python -m unittest discover -s source/image_retrieval -p "test_*.py" -q
npm --prefix frontend test
```

Test tích hợp ảnh tự bỏ qua nếu chưa cài artifacts. Adapter/ranking tests không thay thế GPU smoke test. Model: [Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B), [Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B).
