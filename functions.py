import cv2
import numpy as np
import wave
import pyaudio
import os  # For file existence check

def getBrightness(cam):
    ret, frame = cam.read()
    if not ret:
        return "Error: Unable to capture a frame", None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg = np.sum(frame) / (frame.shape[0] * frame.shape[1])
    avg /= 255

    brightness_description = ""
    if avg > 0.6:
        brightness_description = "Very bright"
    elif avg > 0.4:
        brightness_description = "Bright"
    elif avg > 0.2:
        brightness_description = "Dim"
    else:
        brightness_description = "Dark"

    return brightness_description, avg

def play_file(fname):
    if not os.path.isfile(fname):
        return "Error: Audio file not found"

    # create an audio object
    wf = wave.open(fname, 'rb')
    p = pyaudio.PyAudio()
    chunk = 1024

    # open stream based on the wave object which has been input.
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data (based on the chunk size)
    data = wf.readframes(chunk)

    # play stream (looping from beginning of file to the end)
    while data != '':
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(chunk)

    # cleanup stuff.
    stream.close()
    p.terminate()
