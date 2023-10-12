import speech_recognition as sr
import pyttsx3
from google.oauth2 import service_account
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

class SpeechToText:
    def __init__(self):
        # Define voice IDs
        self.en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
        self.ru_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_RU-RU_IRINA_11.0"

        # Initialize recognizer and microphone
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('voice', self.en_voice_id)

        # Initialize Google Cloud credentials
        self.credentials = service_account.Credentials.from_service_account_file('api-key.json')

    def recognize_speech_from_mic(self):
        print("Listening...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

        response = {
            "success": True,
            "error": None,
            "transcription": None
        }

        try:
            response["transcription"] = self.recognizer.recognize_google(audio)
            print("Recognized:", response["transcription"])
        except sr.RequestError:
            # API was unreachable or unresponsive
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            # Speech was unintelligible or not recognized
            response["error"] = "Speech not detected"

        return response["transcription"]

    def clean_text(self, text):
        lem = WordNetLemmatizer()
        stem = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        new_words = ["hey", "hi", "hello", "what's up", "i", "please", "help", "using", "show", "result", "large",
                     "also", "iv", "one", "two", "new", "previously", "shown"]
        stop_words = stop_words.union(new_words) - {"whom", "who"}

        text = text.lower()
        text = text.split()
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if word not in stop_words]
        text = " ".join(text)

        return text

    def text_to_speech(self, cleaned_text):
        self.engine.say(cleaned_text)
        self.engine.runAndWait()
