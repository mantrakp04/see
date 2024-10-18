import os
import gc
import logging

import spaces
import torch
import gradio as gr
import whisperx
import numpy as np
import edge_tts


# Global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
compute_type = 'float16'

# Load WhisperX model
def load_whisperx():
    return whisperx.load_model('deepdml/faster-whisper-large-v3-turbo-ct2', device, compute_type=compute_type)

# Load TTS Engine
def tts_stream(text_stream):
    # Collect text stream till a sentence is formed
    text = ''
    for t in text_stream:
        text += t
        if t in ['.', '?', '!']:
            break
    
    # Stream TTS
    audio_stream = edge_tts.Communicate(text, voice='en-GB-SoniaNeural')
    for chunk in audio_stream.stream_sync():
        if chunk['type'] == 'audio':
            yield chunk['data']

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")