import time
import sounddevice as sd
from scipy.io.wavfile import write

freq = 16000
duration = 15

time.sleep(5)
recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
sd.wait()

write(f"audio_capture/test.wav", freq, recording)