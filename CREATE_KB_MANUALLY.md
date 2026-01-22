# كيفية إنشاء Knowledge Base و Agent من ElevenLabs Dashboard

بما أن إنشاء Knowledge Base من API قد لا يكون متاحاً في بعض الإصدارات، إليك الخطوات اليدوية:

## 📚 الخطوة 1: إنشاء Knowledge Base

1. **اذهب إلى Dashboard:**
   ```
   https://elevenlabs.io/app/knowledge-base
   ```

2. **اضغط على "Create Knowledge Base" أو "New Knowledge Base"**

3. **املأ البيانات:**
   - **Name:** `Technical Support Knowledge Base`
   - **Description:** (اختياري) `Knowledge base for technical support with part numbers`

4. **احفظ Knowledge Base**

5. **انسخ Knowledge Base ID:**
   - من صفحة Knowledge Base
   - أو من URL (عادة يكون في نهاية الرابط)
   - مثال: `kb_xxxxxxxxxxxxx`

6. **أضف ID في ملف `.env`:**
   ```env
   ELEVENLABS_KNOWLEDGE_BASE_ID=kb_xxxxxxxxxxxxx
   ```

---

## 🤖 الخطوة 2: إنشاء Agent

1. **اذهب إلى Dashboard:**
   ```
   https://elevenlabs.io/app/agents
   ```

2. **اضغط على "Create Agent" أو "New Agent"**

3. **املأ البيانات الأساسية:**
   - **Name:** `Technical Support Agent`
   - **Description:** (اختياري) `Voice agent for technical support with knowledge base`

4. **اختر Voice:**
   - اختر voice مناسب (يفضل multilingual يدعم العربية والإنجليزية)
   - مثال: `Rachel`, `Adam`, أو أي voice multilingual

5. **ربط Knowledge Base:**
   - في قسم **"Knowledge Base"** أو **"RAG"**
   - اختر Knowledge Base اللي أنشأته: `Technical Support Knowledge Base`
   - **فعّل RAG:** ✅ Enable RAG

6. **System Instructions (اختياري لكن موصى به):**
   - اذهب إلى قسم **"System Instructions"** أو **"Prompt"**
   - الصق محتوى من `app/prompts/agent_system_instructions.md`
   - أو اكتب instructions مخصصة

7. **إعدادات أخرى (اختياري):**
   - **Language:** اختر اللغة (أو اتركه multilingual)
   - **Model:** اختر LLM model (عادة `gpt-4` أو `claude`)
   - **Temperature:** (اختياري) 0.7 للتوازن

8. **احفظ Agent**

9. **انسخ Agent ID:**
   - من صفحة Agent
   - أو من URL
   - مثال: `agent_xxxxxxxxxxxxx`

10. **أضف ID في ملف `.env`:**
    ```env
    ELEVENLABS_AGENT_ID=agent_xxxxxxxxxxxxx
    ```

---

## 📄 الخطوة 3: رفع المستندات إلى Knowledge Base

### من Dashboard:

1. **اذهب إلى Knowledge Base:**
   ```
   https://elevenlabs.io/app/knowledge-base
   ```

2. **اختر Knowledge Base اللي أنشأته**

3. **اضغط "Upload Document" أو "Add Document"**

4. **اختر الملف:**
   - PDF, TXT, DOCX, MD, إلخ

5. **أضف Name مع Part Number:**
   - مثال: `Part_CHS199100RECiN_Installation_Manual.pdf`
   - **مهم:** ضمّن Part Number في الاسم

6. **احفظ**

### من Streamlit UI (بعد تشغيل النظام):

1. شغّل Streamlit
2. اذهب إلى tab **"📚 Knowledge Base"**
3. ارفع مستند مع Part Number
4. النظام هيرفعه تلقائياً للـ Knowledge Base

---

## ✅ التحقق من الإعداد

بعد إضافة IDs في `.env`:

```bash
# تحقق من الإعدادات
python -c "from app.settings import get_settings; s = get_settings(); print('KB ID:', s.elevenlabs_knowledge_base_id or '❌ Missing'); print('Agent ID:', s.elevenlabs_agent_id or '❌ Missing')"
```

---

## 🔄 بعد التحديثات

**مهم:** بعد أي تغيير في `.env`:
1. **أعد تشغيل FastAPI server**
2. **أعد تشغيل Streamlit** (إذا كان شغال)

---

## 📝 ملاحظات مهمة

1. **RAG لازم يكون مفعّل** - تأكد من تفعيل RAG في Agent settings
2. **Part Number في الاسم** - لأن ElevenLabs لا يدعم metadata مخصصة، نضع Part Number في اسم المستند
3. **System Instructions** - مهمة جداً لتوجيه Agent كيف يستخدم Knowledge Base
4. **Multilingual Voice** - اختر voice يدعم العربية إذا كنت ستستخدم العربية

---

## 🆘 إذا واجهت مشاكل

1. **Agent مش بيستخدم Knowledge Base:**
   - تأكد إن RAG مفعّل
   - تأكد إن Knowledge Base مربوط بالـ Agent
   - تأكد إن فيه مستندات في Knowledge Base

2. **Citations مش بتظهر:**
   - تأكد إن System Instructions فيها تعليمات لذكر المصادر
   - تأكد إن RAG مفعّل

3. **API errors:**
   - تأكد إن API key صحيح
   - تأكد إن عندك صلاحيات Agents Platform
   - جرب تحديث ElevenLabs SDK: `pip install --upgrade elevenlabs`

