# دليل التشغيل الكامل - ElevenLabs Knowledge Base Voice Support System

## 📋 الخطوات بالترتيب

### الخطوة 1: تثبيت المكتبات المطلوبة

```bash
# تأكد إنك في المجلد الرئيسي للمشروع
cd D:\SpaceMinders\rag

# ثبت المكتبات
python -m pip install --user -r requirements.txt

# ثبت مكتبة ElevenLabs مع دعم الصوت (مهم جداً)
python -m pip install --user "elevenlabs[pyaudio]"
```

**ملاحظة:** لو عندك مشاكل في تثبيت `pyaudio` على Windows:
```bash
# جرب تثبيت PyAudio من binary wheel
python -m pip install --user pipwin
pipwin install pyaudio
```

---

### الخطوة 2: إعداد ملف `.env` مع API Keys

```bash
# انسخ ملف env.example إلى .env
copy env.example .env
```

افتح ملف `.env` وعدّل القيم التالية:

```env
# Gemini API Key (مطلوب)
GEMINI_API_KEY=your_gemini_key_here

# ElevenLabs API Key (مطلوب)
ELEVENLABS_API_KEY=your_elevenlabs_key_here

# ElevenLabs Agent ID (سنتعرف عليه في الخطوة التالية)
ELEVENLABS_AGENT_ID=your_agent_id_here

# ElevenLabs Knowledge Base ID (سيتم إنشاؤه تلقائياً)
ELEVENLABS_KNOWLEDGE_BASE_ID=your_kb_id_here

# إعدادات اختيارية
GEMINI_MODEL=gemini-2.5-flash
DEFAULT_FOLLOWUP_MODE=troubleshooting
```

---

### الخطوة 3: إنشاء ElevenLabs Agent و Knowledge Base

**ملاحظة مهمة:** إنشاء Knowledge Base من API قد لا يكون متاحاً في بعض الإصدارات. 
**الحل:** إنشاء Knowledge Base و Agent يدوياً من Dashboard (أسهل وأضمن).

#### 3.1: إنشاء Knowledge Base من Dashboard (موصى به)

1. **اذهب إلى:**
   ```
   https://elevenlabs.io/app/knowledge-base
   ```

2. **اضغط "Create Knowledge Base" أو "New Knowledge Base"**

3. **املأ البيانات:**
   - **Name:** `Technical Support Knowledge Base`
   - **Description:** (اختياري)

4. **احفظ وانسخ Knowledge Base ID**

5. **أضف في `.env`:**
   ```env
   ELEVENLABS_KNOWLEDGE_BASE_ID=kb_xxxxxxxxxxxxx
   ```

**أو جرب من API (إذا كان متاح):**
```bash
python setup_elevenlabs.py
```

#### 3.2: إنشاء Agent في ElevenLabs Dashboard

1. **اذهب إلى:**
   ```
   https://elevenlabs.io/app/agents
   ```

2. **اضغط "Create Agent" أو "New Agent"**

3. **املأ البيانات:**
   - **Name:** `Technical Support Agent`
   - **Voice:** اختر voice مناسب (يفضل multilingual)
   - **Knowledge Base:** اختر Knowledge Base اللي أنشأته
   - **Enable RAG:** ✅ فعّل RAG (مهم جداً!)

4. **System Instructions (موصى به):**
   - اذهب إلى قسم "System Instructions" أو "Prompt"
   - يمكنك استخدام محتوى من `app/prompts/agent_system_instructions.md`

5. **احفظ Agent**

6. **انسخ Agent ID** وضيفه في `.env`:
   ```env
   ELEVENLABS_AGENT_ID=agent_xxxxxxxxxxxxx
   ```

**للمزيد من التفاصيل:** راجع ملف `CREATE_KB_MANUALLY.md`

---

### الخطوة 4: رفع المستندات إلى Knowledge Base

#### الطريقة الأولى: من Streamlit UI (أسهل)

1. شغّل Streamlit (انظر الخطوة 6)
2. اذهب إلى tab **"📚 Knowledge Base"**
3. اضغط **"📤 Upload New Document"**
4. اختر ملف PDF أو نص
5. أدخل **Part Number** (مثلاً: `CHS199100RECiN`)
6. اضغط **"Upload Document"**

#### الطريقة الثانية: من API مباشرة

```bash
# مثال باستخدام curl
curl -X POST "http://localhost:8000/knowledge-base/documents" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/manual.pdf" \
  -F "part_number=CHS199100RECiN" \
  -F "name=Installation_Manual"
```

#### الطريقة الثالثة: من Python

```python
from app.elevenlabs_knowledge_base import upload_document_with_part_number
from app.settings import get_settings

settings = get_settings()

# اقرأ الملف
with open("path/to/manual.pdf", "rb") as f:
    file_bytes = f.read()

# ارفع المستند
result = upload_document_with_part_number(
    file_bytes=file_bytes,
    file_name="manual.pdf",
    part_number="CHS199100RECiN",
    api_key=settings.elevenlabs_api_key,
    knowledge_base_id=settings.elevenlabs_knowledge_base_id,
)

print(f"✅ Document uploaded: {result['document_id']}")
```

---

### الخطوة 5: تشغيل FastAPI Server

افتح terminal جديد واتبع الخطوات:

```bash
# تأكد إنك في المجلد الرئيسي
cd D:\SpaceMinders\rag

# شغّل السيرفر
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**التحقق:** افتح المتصفح على `http://localhost:8000/docs` - يجب أن ترى Swagger UI.

**ملاحظة:** اترك هذا الـ terminal مفتوح - السيرفر لازم يشتغل طول الوقت.

---

### الخطوة 6: تشغيل Streamlit UI

افتح terminal جديد (الـ terminal الأول لازم يفضل مفتوح للسيرفر):

```bash
# تأكد إنك في المجلد الرئيسي
cd D:\SpaceMinders\rag

# شغّل Streamlit
streamlit run streamlit_app.py
```

**التحقق:** Streamlit هيفتح تلقائياً في المتصفح على `http://localhost:8501`

---

### الخطوة 7: استخدام النظام

#### 7.1: تحليل فيديو

1. في Streamlit، اذهب إلى tab **"📤 Upload Video"**
2. ارفع فيديو أو صورة
3. اضغط **"🚀 Analyze Video"**
4. انتظر التحليل (قد يستغرق 10-30 ثانية)
5. شوف النتائج: نوع الجهاز، المشكلة، Part Number، إلخ

#### 7.2: بدء محادثة صوتية

1. بعد تحليل الفيديو، اذهب إلى tab **"🎤 Voice Conversation"**
2. تأكد إن Knowledge Base ID موجود
3. اضغط **"🎤 Start Voice Chat"**
4. الـ Agent هيفهم context من الفيديو ويهبدأ المحادثة

**ملاحظة:** المحادثة الصوتية الفعلية تحتاج استخدام ElevenLabs SDK أو WebSocket. الـ UI الحالي يعرض الـ configuration والـ messages.

#### 7.3: إدارة Knowledge Base

1. اذهب إلى tab **"📚 Knowledge Base"**
2. شوف قائمة المستندات الموجودة
3. ارفع مستندات جديدة مع Part Number
4. احذف مستندات إذا احتجت

---

## 🔍 التحقق من أن كل شيء يعمل

### 1. تحقق من API Keys

```python
from app.settings import get_settings

settings = get_settings()
print(f"Gemini API Key: {'✅ موجود' if settings.gemini_api_key else '❌ مفقود'}")
print(f"ElevenLabs API Key: {'✅ موجود' if settings.elevenlabs_api_key else '❌ مفقود'}")
print(f"Agent ID: {'✅ موجود' if settings.elevenlabs_agent_id else '❌ مفقود'}")
print(f"KB ID: {'✅ موجود' if settings.elevenlabs_knowledge_base_id else '❌ مفقود'}")
```

### 2. تحقق من FastAPI Server

افتح `http://localhost:8000/health` - يجب أن ترى:
```json
{"status": "ok"}
```

### 3. تحقق من Knowledge Base

في Streamlit، tab **"📚 Knowledge Base"**:
- اضغط **"🔄 Refresh Documents List"**
- يجب أن ترى المستندات اللي رفعتها

---

## ⚠️ حل المشاكل الشائعة

### مشكلة: "ELEVENLABS_API_KEY is required"
**الحل:** تأكد إنك ضفت الـ API key في ملف `.env` و restart الـ server.

### مشكلة: "Knowledge Base ID not set"
**الحل:** 
1. أنشئ Knowledge Base (الخطوة 3.1)
2. ضيف الـ ID في `.env`
3. restart الـ server

### مشكلة: "Agent ID not set"
**الحل:**
1. أنشئ Agent في ElevenLabs Dashboard
2. ضيف الـ Agent ID في `.env`
3. restart الـ server

### مشكلة: PyAudio installation failed
**الحل:**
```bash
# على Windows
pip install pipwin
pipwin install pyaudio

# أو استخدم conda
conda install pyaudio
```

### مشكلة: FastAPI server مش بيشتغل
**الحل:**
- تأكد إن port 8000 مش مستخدم
- جرب port تاني: `--port 8001`
- تأكد إن كل المكتبات مثبتة

---

## 📝 ملاحظات مهمة

1. **FastAPI Server لازم يشتغل طول الوقت** - لو قفلته، Streamlit مش هيعرف يتصل بالـ API.

2. **Knowledge Base لازم يكون فيه مستندات** - لو Knowledge Base فاضي، الـ Agent مش هيعرف يجاوب.

3. **Part Number مهم** - كل مستند لازم يكون معاه Part Number عشان الـ Agent يقدر يبحث بشكل صحيح.

4. **RAG لازم يكون مفعّل** - في ElevenLabs Dashboard، تأكد إن RAG enabled للـ Agent.

5. **المحادثة الصوتية** - الـ UI الحالي يعرض الـ configuration. للمحادثة الصوتية الفعلية، استخدم ElevenLabs SDK أو WebSocket API.

---

## 🎯 الخطوات السريعة (Quick Start)

```bash
# 1. ثبت المكتبات
pip install -r requirements.txt
pip install "elevenlabs[pyaudio]"

# 2. ضبط .env
copy env.example .env
# عدّل .env وضيف API keys

# 3. أنشئ Knowledge Base (من Python)
python setup_elevenlabs.py

# 4. شغّل FastAPI (terminal 1)
uvicorn app.main:app --reload --port 8000

# 5. شغّل Streamlit (terminal 2)
streamlit run streamlit_app.py

# 6. افتح Streamlit في المتصفح
# http://localhost:8501
```

---

## ✅ Checklist

- [ ] المكتبات مثبتة
- [ ] ملف `.env` موجود ومضبوط
- [ ] ElevenLabs API Key موجود
- [ ] Gemini API Key موجود
- [ ] Knowledge Base تم إنشاؤه و ID موجود في `.env`
- [ ] Agent تم إنشاؤه و ID موجود في `.env`
- [ ] FastAPI server شغال على port 8000
- [ ] Streamlit شغال على port 8501
- [ ] Knowledge Base فيه مستندات على الأقل واحد
- [ ] جربت تحليل فيديو واشتغل
- [ ] جربت رفع مستند جديد واشتغل

---

**لو عندك أي مشاكل، راجع قسم "حل المشاكل الشائعة" أو شوف الـ logs في الـ terminals.**

