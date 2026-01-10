## Video Understanding (Gemini) — Home Appliances Support with RAG

This repo provides a **FastAPI** service that accepts a **video upload** or a **video URL**, analyzes it with **Gemini**, retrieves relevant documentation from **Chroma DB** (using BGE-M3 embeddings), and returns a grounded technical answer with citations.

Features:
- Video analysis to identify appliance type, brand, and issues
- Question generation (extracted from transcript + clarifying questions)
- RAG retrieval from La Marzocco catalogs (Chroma DB + BGE-M3)
- Grounded answer generation using retrieved documentation

Reference architecture: [Voice AI Support System - Technical Proposal](file:///d%3A/SpaceMinders/rag/Voice%20AI%20Support%20System%20-%20Technical%20Proposal.pdf)

### Requirements

- Python 3.11+

## خطوات التشغيل / Setup Steps

### الخطوة 1: تثبيت المكتبات / Step 1: Install Dependencies

```bash
python -m pip install --user -r requirements.txt
```

**ملاحظة:** أول مرة هتحمل BGE-M3 model (~2GB) تلقائياً عند أول استخدام.

#### دعم GPU (اختياري لكن موصى به) / GPU Support (Optional but Recommended)

إذا كان عندك GPU (مثل RTX 5050)، النظام هيكتشفه تلقائياً ويستخدمه لتسريع إنشاء embeddings بشكل كبير (10-20x أسرع).

**للتحقق من دعم CUDA:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**لو CUDA مش متاح:**
- تأكد إنك مثبت PyTorch مع دعم CUDA:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
  (استبدل `cu121` بـ CUDA version بتاعك)

**مع GPU:** إنشاء embeddings للـ 111 chunks: ~5-10 ثواني  
**بدون GPU:** إنشاء embeddings للـ 111 chunks: ~2-3 دقائق

### الخطوة 2: إعداد API Keys / Step 2: Configure API Keys

**Option A (PowerShell, current session only):**

```bash
$env:GEMINI_API_KEY="PUT_YOUR_KEY_HERE"
```

**Option B (persistent on Windows):**

```bash
setx GEMINI_API_KEY "PUT_YOUR_KEY_HERE"
```

**Option C (recommended):** create a local `.env` file based on `env.example`:

```bash
# Copy env.example to .env
copy env.example .env

# Edit .env and add your API key
GEMINI_API_KEY=your_key_here
```

### الخطوة 3: عمل Embeddings للملفات PDF / Step 3: Generate Embeddings for PDFs

**مهم جداً:** لازم تعمل embeddings للملفات PDF قبل ما تستخدم الـ API.

```bash
python scripts/embed_pdfs.py
```

هذا السكريبت سيقوم بـ:
- استخراج النصوص من الملفين:
  - `2024-SCA_SalesBrochure_digital_USA-324-1-min.pdf`
  - `Productcatalog2025.pdf`
- تقسيمها إلى chunks
- إنشاء embeddings باستخدام BGE-M3
- حفظها في Chroma DB في مجلد `./chroma_db`

**الوقت المتوقع:** 5-15 دقيقة (حسب حجم الملفات وسرعة الإنترنت لتحميل BGE-M3)

**التحقق من النجاح:** هتشوف رسالة في النهاية تقول عدد الـ chunks اللي تم إضافتها.

### الخطوة 4: تشغيل السيرفر / Step 4: Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

السيرفر هيفتح على: `http://localhost:8000`

**للتحقق:** افتح `http://localhost:8000/health` في المتصفح (يفترض يرجع `{"status":"ok"}`)

### الخطوة 5: استخدام الـ API / Step 5: Use the API

#### Endpoints المتاحة:

1. **`GET /health`** - للتحقق من حالة السيرفر

2. **`POST /video/analyze`** - تحليل الفيديو فقط (بدون RAG)
   - نفس الـ endpoint القديم

3. **`POST /assist/video`** - **الـ endpoint الجديد** (مع RAG)
   - يحلل الفيديو
   - يستخرج الأسئلة
   - يسترجع من Chroma DB
   - يولد إجابة مُستندة للمستندات

#### استخدام `/assist/video`:

**مثال 1: رفع ملف فيديو (Upload)**

```bash
curl -X POST "http://localhost:8000/assist/video" ^
  -F "video_file=@sample.mp4" ^
  -F "language=ar"
```

**مثال 2: استخدام رابط فيديو (URL)**

```bash
curl -X POST "http://localhost:8000/assist/video" ^
  -F "video_url=https://example.com/video.mp4" ^
  -F "language=ar"
```

**مثال 3: مع user_hint (معلومات إضافية)**

```bash
curl -X POST "http://localhost:8000/assist/video" ^
  -F "video_file=@sample.mp4" ^
  -F "language=ar" ^
  -F "user_hint=العميل يقول إن المكنة مش بتشتغل من يومين"
```

#### Response Structure:

```json
{
  "ok": true,
  "request_id": "...",
  "model_video": "gemini-2.5-flash",
  "model_answer": "gemini-2.5-pro",
  "analysis": {
    "appliance_type": "coffee machine",
    "brand_or_model": "La Marzocco Linea Mini",
    "transcript": "...",
    "issue_summary": "...",
    "likely_root_causes": [...],
    "recommended_fix_steps": [...],
    "safety_warnings": [...]
  },
  "transcript": "النص الكامل من الفيديو",
  "user_questions": ["الأسئلة المستخرجة من الفيديو"],
  "clarifying_questions": ["أسئلة توضيحية مقترحة"],
  "retrieval": {
    "citations": [
      {
        "text": "مقتطف من الكتالوج",
        "source_file": "Productcatalog2025.pdf",
        "page_number": 15,
        "distance": 0.23
      }
    ]
  },
  "answer": {
    "text": "الإجابة المُستندة للمستندات مع الحل",
    "follow_up_questions": ["أسئلة متابعة"]
  }
}
```

### ملاحظات مهمة / Important Notes

- **لازم تعمل embeddings أولاً:** لو مش عملت embeddings للملفات PDF، الـ API مش هيعرف يسترجع معلومات من Chroma DB
- **الوقت المتوقع:** أول طلب ممكن ياخد 10-30 ثانية (تحميل BGE-M3 model)
- **حجم الفيديو:** الحد الأقصى 25MB (يمكن تعديله في `.env` بـ `MAX_VIDEO_BYTES`)
- **اللغة:** الـ default هو `ar` (عربي)، يمكن تغييره بـ `language=en`

---

## Using Streamlit Web Interface

This repository includes a **Streamlit web interface** for easy video analysis without using curl or API calls.

### Running the Streamlit App

After completing steps 1-3 above (install dependencies, set API key, generate embeddings), run:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Streamlit Features

- **Video Upload**: Upload video files directly from your computer (MP4, WebM, MOV, AVI)
- **Video URL**: Provide a direct link to a video file
- **Language Selection**: Choose English or Arabic for responses
- **Additional Context**: Add optional hints or context about the issue
- **Comprehensive Results**:
  - Device identification (appliance type, brand/model)
  - Problem analysis (issue summary, root causes)
  - Step-by-step repair instructions
  - Required tools and parts
  - Source citations with page numbers
  - Safety warnings (if applicable)
  - Follow-up questions for better diagnosis

### Streamlit UI Components

The interface displays:
1. **Device Information**: Appliance type and brand/model metrics
2. **Problem Analysis**: Issue summary and likely root causes
3. **Safety Warnings**: Important safety information (if any)
4. **Repair Instructions**: Detailed guide with steps and RAG-generated answer
5. **Sources & References**: Citations from documentation with excerpts
6. **Follow-up Questions**: Suggested questions for better diagnosis

### Requirements for Streamlit

All dependencies are included in `requirements.txt`. Make sure you have:
- Completed PDF embeddings generation (`python scripts/embed_pdfs.py`)
- Set `GEMINI_API_KEY` in `.env` or environment variables

The Streamlit app uses the same RAG pipeline as the FastAPI endpoint but provides a user-friendly web interface.

---

## ملخص سريع / Quick Summary

**الترتيب الصحيح للتشغيل:**

1. ✅ `pip install -r requirements.txt` - تثبيت المكتبات
2. ✅ إعداد `GEMINI_API_KEY` في `.env` أو environment variable
3. ✅ `python scripts/embed_pdfs.py` - **مهم جداً:** عمل embeddings للملفات PDF
4. ✅ Choose one:
   - **Streamlit UI**: `streamlit run streamlit_app.py` (user-friendly web interface)
   - **FastAPI Server**: `uvicorn app.main:app --reload` (then use `POST /assist/video`)

**لو نسيت خطوة:** راجع الخطوات أعلاه بالتفصيل.


