# حلول تثبيت PyAudio على Windows

## الحل 1: تثبيت من Wheel File مباشرة (الأسهل)

```powershell
# تحميل PyAudio wheel لـ Python 3.11 على Windows
# اذهب إلى: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

# أو استخدم هذا الأمر مباشرة (لـ Python 3.11, 64-bit):
pip install https://download.lfd.uci.edu/pythonlibs/archived/PyAudio-0.2.14-cp311-cp311-win_amd64.whl
```

**ملاحظة:** إذا كان Python 32-bit، استخدم:
```powershell
pip install https://download.lfd.uci.edu/pythonlibs/archived/PyAudio-0.2.14-cp311-cp311-win32.whl
```

---

## الحل 2: استخدام conda (إذا كان متاح)

```powershell
conda install pyaudio
```

---

## الحل 3: تثبيت Visual C++ Build Tools ثم تثبيت من source

```powershell
# 1. ثبت Visual C++ Build Tools من:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 2. ثم ثبت PyAudio من source
pip install pyaudio
```

---

## الحل 4: تخطي PyAudio (إذا لم تكن تحتاج محادثة صوتية مباشرة)

**مهم:** PyAudio مطلوب فقط للمحادثة الصوتية المباشرة باستخدام `DefaultAudioInterface`.

إذا كنت ستستخدم:
- **WebSocket API** من ElevenLabs (مباشر من المتصفح)
- **ElevenLabs SDK** بدون audio interface مخصص

فيمكنك **تخطي PyAudio** والاستمرار بدون مشاكل.

---

## التحقق من التثبيت

بعد التثبيت، جرب:

```python
python -c "import pyaudio; print('✅ PyAudio installed successfully')"
```

---

## إذا استمرت المشكلة

يمكنك تعديل الكود ليعمل بدون PyAudio:

1. في `app/elevenlabs_agent.py`، يمكنك جعل `DefaultAudioInterface` اختياري
2. أو استخدام WebSocket مباشرة من المتصفح

