import os
from ocha import record_audio,ezuth,play_audio, analyze_voice  
import pyttsx3
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from speechbrain.inference.speaker import EncoderClassifier


load_dotenv()

llm_model = os.getenv("LLM_MODEL")
hf_token = os.getenv("HF_TOKEN")


client = InferenceClient(
    model = llm_model,
    token = hf_token
)


header = r"""
 ██████   ██████             █████  █████████                           █████     
░░██████ ██████             ░░███  ███░░░░░███                         ░░███      
 ░███░█████░███   ██████  ███████ ░███    ░░░ ████████   ██████  ██████ ░███ █████
 ░███░░███ ░███  ███░░██████░░███ ░░█████████░░███░░███ ███░░███░░░░░███░███░░███ 
 ░███ ░░░  ░███ ░███████░███ ░███  ░░░░░░░░███░███ ░███░███████  ███████░██████░  
 ░███      ░███ ░███░░░ ░███ ░███  ███    ░███░███ ░███░███░░░  ███░░███░███░░███ 
 █████     █████░░██████░░████████░░█████████ ░███████ ░░██████░░███████████ █████
░░░░░     ░░░░░  ░░░░░░  ░░░░░░░░  ░░░░░░░░░  ░███░░░   ░░░░░░  ░░░░░░░░░░░ ░░░░░ 
                                              ░███                                
                                              █████                               
                                             ░░░░░                                
                                                    from team:  _           
                                                     ___   __  | |_    __ _ 
                                                    / _ \ / _| | ' \  / _` |
                                                    \___/ \__| |_||_| \__,_|

                                I am your Virtual Medical Assistant
"""
print(header, "\n 1. Use the Medspeak (Default Voice) \n 2. Do you want to analyse the audio ? \n")
print("\t\t\t\t\tNote : If You want to exit you can prompt \"/exit\" or \"/onnula\" commands \n\n")
initial_input = input()


if initial_input == "1":
    preshnm = input("Hi your Virtual Doctor here, What brings you here today? : ")
    while True:
        if preshnm.lower() == "/onnula" or preshnm.lower() == "/exit":
            break
        else:
            kitti = ezuth(client, preshnm)
            engine = pyttsx3.init()
            engine.setProperty('rate', 170) 
            engine.setProperty('volume', 1.0) 
            text_to_speak = kitti
            engine.say(text_to_speak)
            engine.runAndWait()
            preshnm = input("vere? : ")
elif initial_input == "2":
    record_audio("kettath.wav", duration=10)
    recorded = "kettath.wav"
    response = analyze_voice(recorded)
    prompt = input("Hi your Virtual Doctor here, What brings you here today? : ")
    preshnm = prompt+". \nthis is the result after analysing the sound : "+ response
    kitti = ezuth(client, preshnm)
    engine = pyttsx3.init()
    engine.setProperty('rate', 150) 
    engine.setProperty('volume', 1.0) 
    text_to_speak = kitti
    engine.say(text_to_speak)
    engine.runAndWait()
else:
    exit()