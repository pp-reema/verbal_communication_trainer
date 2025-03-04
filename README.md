# 🎤 AI Verbal Communication Trainer  
## A Speech Evaluation and Training Tool Powered by Open-Source LLMs  

### 📌 Overview  
This project is an **AI-powered verbal communication trainer** that helps users enhance their **public speaking, storytelling, and conflict resolution skills** through structured AI-driven feedback.  

It uses **Gradio** for the web interface, **Whisper** for speech-to-text conversion, and the **Falcon-7B-Instruct** model with **4-bit quantization** for speech analysis.  

---

## 🚀 Features  

✅ **Chat & Voice Interaction** – Engage with AI via text or voice with real-time feedback.  
✅ **Speech Training Modules** – Improve **impromptu speaking, storytelling, and conflict resolution** skills.  
✅ **Presentation Assessment** – AI evaluates **clarity, structure, pacing, and confidence**.  
✅ **Speech Tone Analysis** – Detects **confidence, vocal variation, and filler words**.  
✅ **Progress Tracking** – Stores user scores in JSON to track improvements over time.  

---

## 🛠️ Technologies Used  

- **Falcon-7B-Instruct (4-bit Quantization)** – Powers the speech evaluation system.  
- **Whisper (OpenAI)** – Converts **voice input to text** for analysis.  
- **Gradio** – Provides a **user-friendly web interface**.  
- **JSON Storage** – Tracks **user progress** over multiple sessions.  
- **Python (Transformers, PyTorch, NumPy, OpenAI Whisper)** – For **NLP, speech processing, and AI inference**.  

---

## 📂 Project Structure  

📦 **AI Verbal Communication Trainer**  
- 📂 **output/** – Contains screenshots & demo results.  
- 📜 **ai_speech.ipynb** – Jupyter Notebook with AI logic.  
- 📜 **training_prompts.json** – Predefined training prompts for speech tasks.  
- 📜 **README.md** – Project Documentation.

---

## 📌 Model Selection  

I chose **Falcon-7B-Instruct** with **4-bit quantization** for this project due to its efficiency and strong instruction-following capabilities.  

### 🔹 **Why Falcon-7B-Instruct?**  
- **Optimized for structured feedback** – Ideal for speech evaluation and coaching.  
- **Balanced performance & compute efficiency** – Outperforms many models in instruction-tuned tasks while being lighter than LLaMA-13B.  
- **Open-source & community-backed** – Ensures flexibility and ongoing improvements.  

### 🔹 **Why 4-bit Quantization?**  
- **Lower memory usage** – Runs on consumer-grade GPUs.  
- **Faster inference** – Reduces latency without significant accuracy loss.  

This choice provides **high-quality speech analysis** without requiring heavy computational resources. 🚀

---

## 🚀 Happy Speaking! 🎤  



