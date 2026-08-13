import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
plt.figure("Analyze Sound", facecolor='black', edgecolor='black')
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.ion()

import sounddevice as sd

FS = 48_000
CHANNELS = 1
CHUNK_SIZE = 4096

symbol_time = 0.2
start_seq_def = [1000, 3000, 600, 4000]
payload_freq_min, payload_freq_max = 6000, 14000
payload_freq_range = range(payload_freq_min - 100, payload_freq_max + 100)
payload_seq_def = [6000, 9000, 13000, 14000]

cache_time_len = len(start_seq_def + payload_seq_def) * symbol_time
cache_data_len = int(cache_time_len * FS)
cache_buffer = np.zeros(cache_data_len, dtype='float32')

mic_buffer = np.zeros((CHUNK_SIZE, CHANNELS), dtype='float32')
out_buffer = np.zeros((CHUNK_SIZE, CHANNELS), dtype='float32')

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    
    mic_buffer[:] = indata
    # right concat the mic_buffer to the cache_buffer
    cache_buffer[:] = np.concatenate((cache_buffer[len(mic_buffer):], mic_buffer[:, 0]))
    outdata[:] = out_buffer

sd_io = sd.Stream(samplerate=FS,
               blocksize=CHUNK_SIZE,
               dtype='float32',
               channels=CHANNELS,
               callback=callback)
sd_io.start()
sd.sleep(1000)
print("Microphone stream started. Listening for start sequence...")

import time

def fft_analyze(buffer):
    xf = fftfreq(len(buffer), 1 / FS)
    yf = fft(buffer) * 2 / len(buffer)
    mask = xf >= 0
    xf = xf[mask]
    yf = np.abs(yf[mask])
    # find top freq
    top_indices = np.argsort(yf)[-3:][::-1]
    top_frequencies = xf[top_indices]
    top_energies = yf[top_indices]
    return xf, yf, top_frequencies, top_energies

def freq_seq_cmp(seq1, seq2, tol=50):
    if len(seq1) != len(seq2):
        return False
    for f1, f2 in zip(seq1, seq2):
        if abs(f1 - f2) > tol:
            return False
    return True

def decode_payload(seq):
    temp_payload = []
    for f in seq:
        if f not in payload_freq_range:
            print("Error to decode payload. Freq not in payload range:", f)
            return None
        baseband_freq = f - payload_freq_min
        # baseband_freq should be 0 - 8000, div it by 500, then we got 16 tiers
        temp_payload.append(baseband_freq // 500)
    # one symbol equ 4bits, 4 symbols equ 16bits. we can convert it to 2 bytes
    byte1 = (temp_payload[0] << 4) | temp_payload[1]
    byte2 = (temp_payload[2] << 4) | temp_payload[3]
    return bytes([byte1, byte2])

start_seq_time_len = len(start_seq_def) * symbol_time
start_seq_data_len = int(start_seq_time_len * FS)

while True:
    # try det the start sequence from the cache_buffer head
    cloned_buffer = cache_buffer.copy()
    window = cloned_buffer[:start_seq_data_len]
    # split it to len(start_seq) pieces, and do fft on each piece
    frags = np.split(window, len(start_seq_def))
    det_freq_seq = []
    for frag in frags:
        xf, yf, top_freq, top_energies = fft_analyze(frag)
        #print("Top frequencies: ", top_freq, "Energies: ", top_energies)
        det_freq_seq.append(top_freq[0].astype(int).item())
        # plt.clf()
        # plt.plot(xf, yf)
        # plt.xlabel("Frequency (Hz)")
        # plt.xlim(0, 1000)
        # plt.pause(0.01)
    #print("Detected freq sequence: ", det_freq_seq)
    if not freq_seq_cmp(det_freq_seq, start_seq_def):
        time.sleep(symbol_time / 20)
        continue
    print("Start sequence detected!")
    payload_chunk = cloned_buffer[start_seq_data_len:]
    # split it to len(payload_seq) pieces, and do fft on each piece
    frags = np.split(payload_chunk, len(payload_seq_def))
    det_freq_seq = []
    for i, frag in enumerate(frags):
        xf, yf, top_freq, top_energies = fft_analyze(frag)
        if i == 0 and top_freq[0] in range(start_seq_def[-1] - 50, start_seq_def[-1] + 50):
            print("Warn: unaligned payload detected. will retry...")
            break
        if top_freq[0] not in payload_freq_range:
            print("Warn: payload freq not in payload range. That might be interference by noise")
            print("Top frequencies:", top_freq, "Energies:", top_energies)
            top_freq = top_freq[1:]
        det_freq_seq.append(top_freq[0].astype(int).item())
    if len(det_freq_seq) == 0:
        time.sleep(symbol_time / 20)
        continue
    print("Detected payload freq sequence: ", det_freq_seq)
    payload_bytes = decode_payload(det_freq_seq)
    if payload_bytes is None:
        print("Warn: failed to decode payload. will retry...")
        continue
    print("Decoded payload bytes:", payload_bytes)
    time.sleep(symbol_time)