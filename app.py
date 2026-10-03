import os
import re
import sys
import site
import time
import requests
import numpy as np
import sounddevice as sd
from pynput import keyboard

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
JOBS_FILE = os.path.join(script_dir, "job_inventory.md")

# Fix Windows DLL path for PyTorch CUDA binaries if present
try:
    os.add_dll_directory(os.path.join(site.getsitepackages()[0], "torch", "lib"))
except Exception:
    pass

from faster_whisper import WhisperModel
from kokoro import KPipeline

# ==========================================
# CONFIGURATION
# ==========================================
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# MODEL_NAME = "qwen3.5:9b"  # Upgraded model for precise spec adherence
MODEL_NAME = "qwen3.8:27b"  # Upgraded model for precise spec adherence
SAMPLE_RATE = 16000
# JOBS_FILE = "./job_inventory.md"

# ==========================================
# DYNAMIC JOB INVENTORY PARSER
# ==========================================

def load_job_inventory(filepath):
    """Parses job_inventory.md into a dict of {company_role: job_description}."""
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found. Please create the file.")
        return {}
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract headers (## Company - Role) and the text following them
    matches = re.findall(r"^##\s+(.+?)\n(.*?)(?=\n##|\Z)", content, re.MULTILINE | re.DOTALL)
    jobs = {title.strip(): body.strip() for title, body in matches}
    return jobs

def select_target_job(jobs):
    """Prompts user to select which company/role to interview for."""
    print("\n==================================================")
    print("      SELECT TARGET COMPANY & JOB POSITION        ")
    print("==================================================")
    
    job_titles = list(jobs.keys())
    if not job_titles:
        raise ValueError("No jobs found in job_inventory.md!")

    for idx, title in enumerate(job_titles, 1):
        print(f" [{idx}] {title}")
        
    while True:
        try:
            choice = int(input("\nEnter choice number: "))
            if 1 <= choice <= len(job_titles):
                selected_title = job_titles[choice - 1]
                print(f"\n✓ Selected Position: {selected_title}\n")
                return selected_title, jobs[selected_title]
        except ValueError:
            pass
        print("Invalid selection. Please enter a valid number.")

def build_system_prompt(company_role, job_description):
    """Constructs a strict system prompt tailored to the target job requirements."""
    return (
        f"You are the Lead Technical Hiring Manager interviewing a candidate for the role of '{company_role}'.\n\n"
        f"TARGET JOB REQUIREMENTS & TECH STACK:\n{job_description}\n\n"
        "INTERVIEWER RULES:\n"
        "1. GROUND YOUR QUESTIONS IN THE JOB REQUIREMENTS ABOVE. Ask specifically about the technologies, tools, and responsibilities mentioned.\n"
        "2. Do NOT ask generic data engineering questions if they are not relevant to this specific role.\n"
        "3. Evaluate the candidate's answer against the required tech stack.\n"
        "4. Keep your responses concise (2-3 sentences max) and ask one focused follow-up question at a time.\n"
    )

# ==========================================
# INITIALIZE MODELS & STATE
# ==========================================

print("Loading Whisper STT Model...")
stt_model = WhisperModel("small.en", device="cpu", compute_type="int8")

print("Loading Kokoro TTS Engine...")
tts_pipeline = KPipeline(lang_code="a")

recording = False
audio_buffer = []

# ==========================================
# AUDIO & PIPELINE FUNCTIONS
# ==========================================

def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_buffer.append(indata.copy())

def record_audio_dynamic():
    global recording, audio_buffer
    audio_buffer = []
    recording = True

    print("\n🎤 [RECORDING STARTED] Speak now... Press [SPACE] when finished.")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)
    
    with stream:
        while recording:
            time.sleep(0.05)

    print("✓ [RECORDING STOPPED]")
    if not audio_buffer:
        return np.array([], dtype='float32')
    return np.concatenate(audio_buffer, axis=0).flatten()

def transcribe_audio(audio_array):
    if len(audio_array) == 0:
        return ""
    print("⚡ Transcribing audio...")
    segments, _ = stt_model.transcribe(audio_array, beam_size=5)
    return " ".join([segment.text for segment in segments]).strip()

def query_ollama(prompt, history):
    history += f"\nCandidate: {prompt}\nInterviewer:"
    payload = {
        "model": MODEL_NAME,
        "prompt": history,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload).json()
    ai_response = response.get("response", "").strip()
    history += f" {ai_response}"
    return ai_response, history

def speak_text(text):
    print(f"\n🤖 Interviewer: {text}\n")
    print("🔊 Speaking...")
    generator = tts_pipeline(text, voice="am_puck", speed=1.0)
    for _, _, audio_array in generator:
        sd.play(audio_array, samplerate=24000)
        sd.wait()

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    global recording
    
    # 1. Load job inventory and prompt for selection
    jobs = load_job_inventory(JOBS_FILE)
    company_role, job_description = select_target_job(jobs)
    
    # 2. Build tailored system prompt
    system_prompt = build_system_prompt(company_role, job_description)
    conversation_history = system_prompt + "\n"
    
    # 3. Dynamic Initial Greeting based on target company
    initial_greeting = (
        f"Welcome to your interview for the {company_role} position. "
        f"Looking at our team's tech stack and requirements, can you give me an overview of your experience "
        f"and how it aligns with this role?"
    )
    speak_text(initial_greeting)
    
    # 4. Main Conversation Loop
    while True:
        print("\n--------------------------------------------------")
        print("Press [ENTER] to start speaking...")
        
        while True:
            if input() == "":
                break
        
        def on_press(key):
            global recording
            if key == keyboard.Key.space:
                recording = False
                return False

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        
        audio_data = record_audio_dynamic()
        listener.join()
        
        user_text = transcribe_audio(audio_data)
        print(f" You said: \"{user_text}\"")
        
        if not user_text:
            print("⚠️ No speech detected. Please try again.")
            continue
            
        print("🧠 Thinking (qwen3.5:9b)...")
        ai_reply, conversation_history = query_ollama(user_text, conversation_history)
        speak_text(ai_reply)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession ended. Good luck with your preparation!")