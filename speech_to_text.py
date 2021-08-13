
from sys import float_repr_style
import speech_recognition as sr

from speech import speak


class STT:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        # self.engine = pyttsx3.init()



    def recognize_speech_from_mic(self,recognizer, microphone):
        if not isinstance(recognizer, sr.Recognizer):
            raise TypeError("`recognizer` must be Recognizer instance")
        
        if not isinstance(microphone, sr.Microphone):
            raise TypeError("`microphone` must be Microphone instance")
        
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            
            speak("Listening. What is your question")
            
            

            audio = recognizer.listen(source, timeout=0, phrase_time_limit=0)

        response = {
            "success": True,
            "error": None,
            "transcription": None
        }

        try:
            response["transcription"] = recognizer.recognize_google(audio)
        except sr.RequestError:
            response["success"] = False
            response["error"] = "API Unavailable"
        except sr.UnknownValueError:
            response["error"] = "Unable to recognize speech"
            response["success"] = False
        return response

    


    def speech_to_text(self):
        # recognizer = sr.Recognizer()
        # microphone = sr.Microphone()
        # engine = pyttsx3.init()
        
        transcription = self.recognize_speech_from_mic(self.recognizer, self.microphone)
        transcript = transcription["transcription"]
        text = f"You said. {transcript}. Getting Answer"

        # self.va.start()
        speak(text)
        
        
        # print(text)
        # self.engine.say(text)
        # self.engine.runAndWait()
        
        return transcript, text




