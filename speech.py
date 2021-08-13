from gtts import gTTS
import os
from io import BytesIO






def speak(msg):
    
    filename = "audio.mp3"
    audio_folder = "./static/audio/"
    file_path = os.path.join(audio_folder, filename)
    speech = gTTS(text=msg, lang='en', tld="ca")
    speech.save(f"{file_path}")
    os.system(f"ffplay {file_path} -autoexit -nodisp")
    os.remove(file_path)



