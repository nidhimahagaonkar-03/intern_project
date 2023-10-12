import functions
import yolopy
import speech
import cv2
import os
import detect
import datetime
import random

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dfkey.json"

labelsPath = "yolo/coco.names"
weightsPath = "yolo/yolov3.weights"
configPath = "yolo/yolov3.cfg"
args = {"threshold": 0.3, "confidence": 0.5}
project_id = "blindbot-4f356"
engine = speech.speech_to_text()

model = yolopy.yolo(labelsPath, weightsPath, configPath)
listening = False
intent = None

TIMEOUT = 5

while True:
    cam = cv2.VideoCapture(1)

    # Handle the case where the user is not listening
    if not listening:
        resp = engine.recognize_speech_from_mic(timeout=TIMEOUT)
        print(resp)
        if resp is not None:
            intent, text = detect.detect_intent_texts(project_id, 0, [resp], "en")
        if intent == "Jyoti" and resp is not None:
            listening = True

    # Handle the case where the user is listening
    else:
        engine.text_speech("What can I help you with?")
        intent = ""
        engine.text_speech("Listening")
        resp = engine.recognize_speech_from_mic(timeout=TIMEOUT)
        engine.text_speech("Processing")
        if resp is not None:
            print(resp)
            intent, text = detect.detect_intent_texts(project_id, 0, [resp], "en")

        # Handle the case where the user does not say anything or the intent detection fails
        if resp is None or intent is None:
            print("Please try again.")

        # Handle the case where the user wants to describe the scene
        elif intent == "Describe":
            detect.describeScene(cam, model, engine)

        # Handle the case where the user wants to end the conversation
        elif intent == "endconvo":
            print(text)
            listening = False
            engine.text_speech(text)

        # Handle the case where the user wants to know the brightness
        elif intent == "Brightness":
            engine.text_speech("It is {} outside".format((functions.getBrightness(cam))[0]))

        # Handle the case where the user wants to fill out a form
        elif intent == "FillForm":
            detect.detect_form(cam, engine)

        # Handle the case where the user wants to read text
        elif intent == "Read":
            print("read")
            detect.detect_text(cam, engine)

        # Handle the case where the user wants to know the time
        elif intent == "Time"
            currentDT = datetime.datetime.now()
            engine.text_speech("The time is {} hours and {} minutes".format(currentDT.hour, currentDT.minute))

        # Handle the case where the user wants to control a smart home device
        elif intent == "ControlSmartHomeDevice":
            # Implement the code to control the smart home device
            pass

        # Handle the case where the user gives an unknown intent
        else:
            print("I don't understand your request. Please try again.")

    cam.release()
