import sounddevice as sd
import wavio
import random
import pygame
import torchaudio
import torch
import torch.nn.functional as F
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
import os
from dotenv import load_dotenv

def play_audio(file_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        # input("Press Enter to stop playback...")
        # pygame.mixer.music.stop()
    except Exception as e:
        print(f"An error occurred: {e}")


def generate_car_sentence():
    subjects = [
        "The iconic Porsche 911",
        "This legendary sports car",
        "With its unmistakable silhouette, the Porsche 911",
        "A marvel of automotive engineering, the Porsche 911"
    ]
    predicates = [
        "is powered by a rear-mounted flat-six engine",
        "delivers an exhilarating driving experience on winding roads",
        "combines timeless design with cutting-edge technology",
        "offers remarkable performance, achieving 0-60 mph in under 4 seconds"
    ]
    details = [
        "thanks to its twin-turbocharged engines and lightweight chassis",
        "with an innovative all-wheel-drive system ensuring optimal traction",
        "while maintaining its heritage of precision engineering and craftsmanship",
        "as a symbol of performance and luxury in the automotive world"
    ]
    additional_info = [
        "making it a favorite among enthusiasts worldwide.",
        "establishing its dominance on both the track and the road.",
        "continuing to set benchmarks for what a sports car can achieve.",
        "highlighting its position as a true icon in the industry."
    ]
    sentence = (
        f"  {random.choice(subjects)} {random.choice(predicates)} {random.choice(details)} "
        f"{random.choice(additional_info)}"
    )
    return sentence



def record_audio(filename: str, duration: int = 5, sample_rate: int = 44100):
    """
    Records audio for a specified duration and saves it as a WAV file.

    Parameters:
        filename (str): The path where the audio file will be saved.
        duration (int): The duration of the recording in seconds. Default is 10 seconds.
        sample_rate (int): The sample rate for the recording. Default is 44100 Hz.
    """
    print("Read this ↓ ")
    print(generate_car_sentence())
    myrecording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("Done recording.")
    
    wavio.write(filename, myrecording, sample_rate, sampwidth=2)
    print(f"Audio saved as {filename}")


def ezuth(client, UserInput):
    system_content = """
    "Act as a virtual Medical assistant.
     Your role is to assist healthcare professionals by providing accurate, evidence-based medical information, offering treatment options, and supporting patient care to the User Input. 
     Always prioritizepatient safety, provide concise answers, and state that your advice does not replace a doctor's judgment. "
    """

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": f"""
            User Input : {UserInput}
            
            Be professional and give accurate medical information. make it short and clear to common people.
            """
        }
    ]

    response = client.chat_completion(
        messages=messages,
        max_tokens=1000,
        temperature=0.1,
        top_p=0.9
    )
    med_response = response['choices'][0]['message']['content']

    return med_response


def analyze_voice(audio_file):
    # Define thresholds based on medical research or heuristic rules
    THRESHOLDS = {
        "mean_pitch_min": 80,     # Typical range for adult speech in Hz
        "mean_pitch_max": 300,
        "energy_min": 0.01,       # RMS energy minimum for healthy voices
        "spectral_centroid_max": 5000,  # High spectral centroid may indicate issues
        "zero_crossing_rate_max": 1000  # Abnormally high zero crossing rate can indicate issues
    }

    # Load audio file
    y, sr = librosa.load(audio_file)
    
    # Extract features
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    energy = librosa.feature.rms(y=y)
    zero_crossings = librosa.zero_crossings(y, pad=False)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    # Aggregate features
    mean_pitch = np.mean(pitch[pitch > 0])  # Filter out zeros
    mean_energy = np.mean(energy)
    zcr = sum(zero_crossings)
    mean_spectral_centroid = np.mean(spectral_centroid)
    
    # Compare against thresholds
    issues = []
    if mean_pitch < THRESHOLDS["mean_pitch_min"] or mean_pitch > THRESHOLDS["mean_pitch_max"]:
        issues.append("Abnormal pitch detected.")
    if mean_energy < THRESHOLDS["energy_min"]:
        issues.append("Low energy detected.")
    if mean_spectral_centroid > THRESHOLDS["spectral_centroid_max"]:
        issues.append("High spectral centroid detected.")
    if zcr > THRESHOLDS["zero_crossing_rate_max"]:
        issues.append("High zero crossing rate detected.")
    
    # Build output message
    if issues:
        output_message = "\nPotential Issues Detected:\n"
        for issue in issues:
            output_message += f"- {issue}\n"
        output_message += "\nThe user's condition may not be medically OK. Further examination is recommended."
    else:
        output_message = "\nThe user's condition appears to be medically OK."
    
    # Return output message and raw results (optional)
    return output_message
    