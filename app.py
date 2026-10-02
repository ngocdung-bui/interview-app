import os
import site
import time
import requests
import numpy as np
import sounddevice as sd
from pynput import keyboard

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
MODEL_NAME = "qwen2.5-coder:7b"
SAMPLE_RATE = 16000  # 16kHz required by Whisper

print("Loading Whisper STT Model...")
# Use device="cuda" if configured, or "cpu"
stt_model = WhisperModel("small.en", device="cpu", compute_type="int8")

print("Loading Kokoro TTS Engine...")
tts_pipeline = KPipeline(lang_code="a")

SYSTEM_PROMPT = (
    "You are an expert technical interviewer conducting a mock interview. "
    "Keep your responses concise, professional, and limited to 2-3 sentences. "
    "Ask relevant follow-up questions based on the candidate's answer."
)

conversation_history = SYSTEM_PROMPT + "\n"

# Recording State Variables
recording = False
audio_buffer = []

# ==========================================
# KEYBOARD-CONTROLLED AUDIO RECORDING
# ==========================================

def audio_callback(indata, frames, time_info, status):
    """Callback function called continuously by sounddevice during recording."""
    if recording:
        audio_buffer.append(indata.copy())

def record_audio_dynamic():
    """Records audio dynamically until SPACE is pressed."""
    global recording, audio_buffer
    audio_buffer = []
    recording = True

    print("\n🎤 [RECORDING STARTED] Speak now... Press [SPACE] when you finish speaking.")

    # Start audio input stream
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)
    
    with stream:
        # Wait until the user sets recording = False via spacebar
        while recording:
            time.sleep(0.05)

    print("✓ [RECORDING STOPPED]")
    
    if not audio_buffer:
        return np.array([], dtype='float32')
        
    # Concatenate recorded audio chunks into a single 1D numpy array
    return np.concatenate(audio_buffer, axis=0).flatten()

# ==========================================
# PIPELINE FUNCTIONS
# ==========================================

def transcribe_audio(audio_array):
    """Transcribes audio array directly using faster-whisper."""
    if len(audio_array) == 0:
        return ""
    print("⚡ Transcribing audio response...")
    start_time = time.time()
    
    segments, _ = stt_model.transcribe(audio_array, beam_size=5)
    transcript = " ".join([segment.text for segment in segments]).strip()
    
    elapsed = time.time() - start_time
    print(f"⏱️ Transcribed in {elapsed:.2f} seconds.")
    return transcript

def query_ollama(prompt):
    global conversation_history
    conversation_history += f"\nCandidate: {prompt}\nInterviewer:"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": conversation_history,
        "stream": False
    }
    
    response = requests.post(OLLAMA_URL, json=payload).json()
    ai_response = response.get("response", "").strip()
    conversation_history += f" {ai_response}"
    return ai_response

def speak_text(text):
    print(f"\n🤖 Interviewer: {text}\n")
    print("🔊 Speaking...")
    generator = tts_pipeline(text, voice="am_puck", speed=1.0)
    for _, _, audio_array in generator:
        sd.play(audio_array, samplerate=24000)
        sd.wait()

# ==========================================
# KEYBOARD LISTENER LOOP
# ==========================================

def main():
    global recording
    print("==================================================")
    print("   LOCAL AI VOICE INTERVIEW PRACTICE APP READY    ")
    print("==================================================")
    
    initial_greeting = "Welcome! Let's begin the interview. Can you briefly introduce yourself and your technical background?"
    speak_text(initial_greeting)
    
    while True:
        print("\n--------------------------------------------------")
        print("Press [ENTER] to start speaking...")
        
        # Wait for ENTER to start
        while True:
            key = input()
            if key == "":  # User pressed ENTER
                break
        
        # Define key listener for stopping (SPACEBAR)
        def on_press(key):
            global recording
            if key == keyboard.Key.space:
                recording = False
                return False  # Stop key listener

        # Start listening for the spacebar in the background
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        
        # Record audio dynamically until spacebar sets recording = False
        audio_data = record_audio_dynamic()
        listener.join()  # Clean up listener thread
        
        # Process transcript
        user_text = transcribe_audio(audio_data)
        print(f" You said: \"{user_text}\"")
        
        if not user_text:
            print("⚠️ No speech detected or recording was too short. Try again.")
            continue
            
        print("🧠 Thinking...")
        ai_reply = query_ollama(user_text)
        speak_text(ai_reply)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession ended. Good luck with your practice!")