import numpy as np
from scipy.io.wavfile import write as wave_write

import sounddevice as sd

FS = 48_000
CHANNELS = 1
CHUNK_SIZE = 4096

def tone(freq, duration):
    n = int(FS * duration)
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)

def gen_payload_tones(payload_bytes):
    if len(payload_bytes) != 2:
        raise ValueError("Payload bytes must be 2 bytes long.")
    payload_tones = []
    for byte in payload_bytes:
        high_nibble = (byte >> 4) & 0x0F
        low_nibble = byte & 0x0F
        payload_tones.append(6000 + high_nibble * 500)
        payload_tones.append(6000 + low_nibble * 500)
    return payload_tones

start_seq = [1000, 3000, 600, 4000]
payload_seq = gen_payload_tones(b'\x01\xCD')  # Example payload bytes
noise_seq = [200, 50, 100]
symbol_time = 0.2

# gen the waveform
waveform = np.concatenate([tone(f, symbol_time) for f in noise_seq + start_seq + payload_seq])

# play the waveform
sd.play(waveform, FS)
sd.wait()
# save to wav file
wave_write("demo.wav", FS, waveform.astype(np.float32))