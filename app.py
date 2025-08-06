
import os
import json
import random
import torch
import gradio as gr
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import whisper

# Global variables
chat_history = []
PROGRESS_FILE = "progress.json"

# Training data
training_data = {
    "storytelling": [
        "Tell a story about a challenge you overcame.",
        "Describe a time when you helped someone in need.",
        "Imagine a futuristic world and narrate a short story.",
        "Tell a story about a childhood memory that shaped you.",
        "Share an experience where you had to make a difficult choice.",
        "Narrate a funny or embarrassing moment and what you learned from it.",
        "Describe a time when a stranger impacted your life in an unexpected way.",
        "Tell a story about a travel experience that changed your perspective.",
        "Recount a time when you faced failure and how it helped you grow.",
        "Share a moment when you discovered something new about yourself."
    ],
    "impromptu_speaking": [
        "If you could invent a new technology, what would it be?",
        "Describe a moment that changed your life.",
        "What advice would you give to your younger self?",
        "If you could live in any historical period, which one and why?",
        "If you could have dinner with any historical figure, who would it be and why?",
        "What is one thing you would change about the world and why?",
        "If you won a million dollars today, what would you do with it?",
        "What is the most important skill for success in the modern world?",
        "Describe a book or movie that had a significant impact on you.",
        "If you had to teach a class on any subject, what would it be and why?"
    ],
    "conflict_resolution": [
        "A coworker takes credit for your work. How do you handle it?",
        "You have a disagreement with your manager. How do you resolve it?",
        "A team member is not contributing to a group project. What do you do?",
        "You need to deliver bad news to a client. How do you approach it?",
        "A customer is unhappy with your service. How do you handle the situation?",
        "You have a conflict with a close friend over a misunderstanding. How do you fix it?",
        "Two team members are in a heated argument during a meeting. How do you mediate?",
        "You are negotiating a deal, but the other party is being unreasonable. What is your approach?",
        "A family member disagrees with your life choices. How do you communicate effectively?",
        "A colleague keeps interrupting you during meetings. How do you address it professionally?"
    ]
}

def initialize_models():
    """Initialize AI models with error handling"""
    try:
        # Load Falcon-7B-Instruct with 4-bit Quantization
        model_name = "tiiuae/falcon-7b-instruct"
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config if device == "cuda" else None,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Load Whisper model
        whisper_model = whisper.load_model("small")
        
        print(f"✅ Models loaded successfully on: {device}")
        return model, tokenizer, whisper_model, device
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, None, "cpu"

# Initialize models
model, tokenizer, whisper_model, device = initialize_models()

def transcribe_audio(audio_file):
    """Convert speech to text"""
    if audio_file is None or whisper_model is None:
        return "❌ No audio file provided or Whisper model not loaded."
    
    try:
        result = whisper_model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        return f"❌ Error transcribing audio: {e}"

def generate_response(user_input):
    """Generate AI response with error handling"""
    if model is None or tokenizer is None:
        return "❌ AI model not loaded. Please try again later."
    
    try:
        formatted_input = f"User: {user_input}\nAssistant:"
        inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.9,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
        
    except Exception as e:
        return f"❌ Error generating response: {e}"

def chat_with_ai(input_type, user_input, audio_input):
    """Handle chat with AI"""
    if input_type == "Voice":
        if audio_input is None:
            return "❌ No audio file provided. Please record your voice."
        user_input = transcribe_audio(audio_input)
    
    if not user_input or user_input.strip() == "":
        return "❌ Please provide some input."
    
    chat_history.append(f"User: {user_input}")
    
    if len(chat_history) > 10:
        chat_history.pop(0)
    
    response = generate_response(user_input)
    chat_history.append(f"Assistant: {response}")
    
    return response

def storytelling_prompt():
    """Pick a random storytelling prompt"""
    return random.choice(training_data["storytelling"])

def impromptu_speaking():
    """Pick a random topic for impromptu speaking"""
    return random.choice(training_data["impromptu_speaking"])

def conflict_resolution_scenario():
    """Pick a random conflict resolution scenario"""
    return random.choice(training_data["conflict_resolution"])

def evaluate_story(user_text):
    """Evaluate storytelling quality"""
    if not user_text or len(user_text.split()) < 5:
        return "❌ This input is too short to be a story. Please provide a complete story."
    
    prompt = f"""
You are a professional storytelling coach. Evaluate this story on:
1. Story Structure (1-10)
2. Emotional Engagement (1-10) 
3. Creativity & Originality (1-10)
4. Clarity & Flow (1-10)

Story: {user_text}

Provide scores and specific feedback for improvement.
"""
    
    return generate_response(prompt)

def evaluate_speech(user_text):
    """Evaluate impromptu speech"""
    if not user_text or len(user_text.split()) < 5:
        return "❌ This response is too short. Please provide a complete response."
    
    prompt = f"""
You are a professional speech coach. Evaluate this speech on:
1. Relevance to Topic (1-10)
2. Clarity & Structure (1-10)
3. Engagement & Persuasion (1-10)
4. Pacing & Confidence (1-10)

Speech: {user_text}

Provide scores and specific improvement tips.
"""
    
    return generate_response(prompt)

def assess_conflict_response(user_text):
    """Assess conflict resolution response"""
    if not user_text or len(user_text.split()) < 5:
        return "❌ This input is too short to assess. Please provide a full response."
    
    prompt = f"""
You are a conflict resolution expert. Evaluate this response on:
1. Empathy & Understanding (1-10)
2. Diplomacy & Professionalism (1-10)
3. Effectiveness of Resolution (1-10)

Response: {user_text}

Provide scores and actionable improvement tips.
"""
    
    return generate_response(prompt)

def save_progress_data(user_text, feedback, topic="General"):
    """Save user progress"""
    try:
        data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topic": topic,
            "speech": user_text[:200] + "..." if len(user_text) > 200 else user_text,
            "feedback": feedback[:300] + "..." if len(feedback) > 300 else feedback
        }
        
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r") as f:
                progress_data = json.load(f)
        else:
            progress_data = []
        
        progress_data.append(data)
        
        # Keep only last 20 entries
        if len(progress_data) > 20:
            progress_data = progress_data[-20:]
        
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress_data, f, indent=2)
        
        return "✅ Progress saved!"
        
    except Exception as e:
        return f"❌ Error saving progress: {e}"

def load_progress():
    """Load user progress"""
    try:
        if not os.path.exists(PROGRESS_FILE):
            return "⚠️ No progress data found."
        
        with open(PROGRESS_FILE, "r") as f:
            progress_data = json.load(f)
        
        if not progress_data:
            return "⚠️ No past results available."
        
        summary = "📊 **Your Recent Performance:**\n\n"
        for entry in progress_data[-5:]:
            summary += f"📅 **Date:** {entry['date']}\n"
            summary += f"📝 **Topic:** {entry['topic']}\n"
            summary += f"💬 **Speech:** {entry['speech']}\n"
            summary += f"💡 **Feedback:** {entry['feedback']}\n\n"
        
        return summary
        
    except Exception as e:
        return f"❌ Error loading progress: {e}"

# Custom CSS
custom_css = """
body { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1) !important;
}

.gr-button-primary {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important;
}
"""

# Build Gradio Interface
with gr.Blocks(css=custom_css, title="AI Communication Trainer") as interface:
    gr.Markdown("# 🎤 AI Verbal Communication Trainer")
    gr.Markdown("### Improve your speaking skills with real-time AI feedback! 🚀")
    
    with gr.Tabs():
        with gr.Tab("🎯 Training Activities"):
            with gr.Accordion("🎲 Impromptu Speaking", open=True):
                impromptu_topic_btn = gr.Button("🎲 Generate Topic")
                impromptu_topic = gr.Textbox(label="Your Topic", interactive=False)
                
                impromptu_input_type = gr.Radio(["Text", "Voice"], label="Input Method", value="Text")
                impromptu_text = gr.Textbox(label="Your Speech", placeholder="Type your response here...")
                impromptu_audio = gr.Audio(label="🎙️ Record Speech", type="filepath")
                
                impromptu_submit = gr.Button("📊 Evaluate Speech", variant="primary")
                impromptu_feedback = gr.Textbox(label="AI Feedback", interactive=False, lines=8)
                
                impromptu_topic_btn.click(impromptu_speaking, outputs=impromptu_topic)
                impromptu_submit.click(
                    lambda input_type, text, audio: evaluate_speech(
                        transcribe_audio(audio) if input_type == "Voice" and audio else text
                    ),
                    inputs=[impromptu_input_type, impromptu_text, impromptu_audio],
                    outputs=impromptu_feedback
                )
            
            with gr.Accordion("📖 Storytelling"):
                story_topic_btn = gr.Button("📖 Generate Story Prompt")
                story_topic = gr.Textbox(label="Your Story Prompt", interactive=False)
                
                story_input_type = gr.Radio(["Text", "Voice"], label="Input Method", value="Text")
                story_text = gr.Textbox(label="Your Story", placeholder="Tell your story here...")
                story_audio = gr.Audio(label="🎙️ Record Story", type="filepath")
                
                story_submit = gr.Button("📊 Evaluate Story", variant="primary")
                story_feedback = gr.Textbox(label="AI Feedback", interactive=False, lines=8)
                
                story_topic_btn.click(storytelling_prompt, outputs=story_topic)
                story_submit.click(
                    lambda input_type, text, audio: evaluate_story(
                        transcribe_audio(audio) if input_type == "Voice" and audio else text
                    ),
                    inputs=[story_input_type, story_text, story_audio],
                    outputs=story_feedback
                )
            
            with gr.Accordion("🤝 Conflict Resolution"):
                conflict_topic_btn = gr.Button("🔥 Generate Scenario")
                conflict_topic = gr.Textbox(label="Your Scenario", interactive=False)
                
                conflict_input_type = gr.Radio(["Text", "Voice"], label="Input Method", value="Text")
                conflict_text = gr.Textbox(label="Your Response", placeholder="How would you handle this?")
                conflict_audio = gr.Audio(label="🎙️ Record Response", type="filepath")
                
                conflict_submit = gr.Button("📊 Evaluate Response", variant="primary")
                conflict_feedback = gr.Textbox(label="AI Feedback", interactive=False, lines=8)
                
                conflict_topic_btn.click(conflict_resolution_scenario, outputs=conflict_topic)
                conflict_submit.click(
                    lambda input_type, text, audio: assess_conflict_response(
                        transcribe_audio(audio) if input_type == "Voice" and audio else text
                    ),
                    inputs=[conflict_input_type, conflict_text, conflict_audio],
                    outputs=conflict_feedback
                )
        
        with gr.Tab("💬 Chat with AI"):
            chat_input_type = gr.Radio(["Text", "Voice"], label="Input Method", value="Text")
            chat_text = gr.Textbox(label="Message", placeholder="Type your message...")
            chat_audio = gr.Audio(label="🎙️ Record Message", type="filepath")
            
            chat_submit = gr.Button("💬 Send", variant="primary")
            chat_output = gr.Textbox(label="AI Response", interactive=False, lines=6)
            
            chat_submit.click(
                chat_with_ai,
                inputs=[chat_input_type, chat_text, chat_audio],
                outputs=chat_output
            )
        
        with gr.Tab("📈 Progress"):
            gr.Markdown("## 📊 Track Your Improvement")
            
            progress_btn = gr.Button("📂 Load Progress")
            progress_output = gr.Textbox(label="Your Progress", interactive=False, lines=10)
            
            progress_btn.click(load_progress, outputs=progress_output)

if __name__ == "__main__":
    interface.launch(share=False)
