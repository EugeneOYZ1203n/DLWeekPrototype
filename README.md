# 🎙 AI Speaking Coach

An AI-powered discussion-based speaking practice app that helps learners improve fluency through open-ended questions, voice recording, transcription, and structured feedback.

[Video demo on Youtube (https://youtu.be/naK1A5bbJqA)](https://youtu.be/naK1A5bbJqA)
Made for DLWeek 2026, hosted by NTU

## ✨ Features

* AI-generated discussion questions (Ollama)
* Voice recording via browser
* Speech-to-text transcription (Whisper)
* Speaking metrics (silence, duration, fillers)
* AI-powered scoring & improvement suggestions
* Progress tracking (WIP)

---

## 🛠 Requirements

* Python 3.9+
* Ollama installed

---

## 🚀 Setup & Usage

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2️⃣ Install Ollama & Pull Model

Install Ollama from: [https://ollama.com](https://ollama.com)

Then run:

```bash
ollama pull translategemma:4b
```

---

### 3️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Mac/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5️⃣ Run the App

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in your terminal.

---

## 📌 How It Works

1. Generate a discussion question
2. Record your spoken response
3. Audio is transcribed
4. System analyzes fluency & structure
5. Receive scores + actionable suggestions

---

## 🔮 Future Improvements

* Real-time speaking feedback
* Pronunciation scoring
* Visual analytics dashboard
* Adaptive difficulty

---

Built to transform AI from a passive tutor into an active speaking coach.
