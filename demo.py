import numpy as np
from scipy.fft import fft, fftfreq
from scipy.io.wavfile import write as wave_write
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
# create left and right figures
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

import sounddevice as sd

FS = 48_000
CHANNELS = 1
CHUNK_SIZE = 4096

# mic_buffer = np.zeros((CHUNK_SIZE, CHANNELS), dtype='float32')
# out_buffer = np.zeros((CHUNK_SIZE, CHANNELS), dtype='float32')

# def callback(indata, outdata, frames, time, status):
#     if status:
#         print(status)
    
#     mic_buffer[:] = indata
#     outdata[:] = out_buffer
    
# sd_io = sd.Stream(samplerate=FS,
#                blocksize=CHUNK_SIZE,
#                dtype='float32',
#                channels=CHANNELS,
#                callback=callback)
# sd_io.start()
# sd.sleep(1000)

def tone(freq, duration):
    n = int(FS * duration)
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq * t)

start_seq = [1000, 3000, 600, 4000, 2000]
payload_seq = [4000, 1000, 4000, 1000]
noise_seq = [200, 50, 100, 300]

# gen the waveform, symbol time is 0.2s
waveform = np.concatenate([tone(f, 0.2) for f in noise_seq + start_seq + payload_seq])

# play the waveform
sd.play(waveform, FS)
sd.wait()
# save to wav file
wave_write("demo.wav", FS, waveform.astype(np.float32))

# ok. try to decode it
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

# step 1: find the start sequence. we know it's len(start_seq) * 0.2 seconds long
start_seq_time_len = len(start_seq) * 0.2
start_seq_data_len = int(start_seq_time_len * FS)
real_truck = None
# swap window to find the start sequence.
for i in range(0, len(waveform) - start_seq_data_len, int(FS * 0.01)):
    window = waveform[i:i+start_seq_data_len]
    # plt.clf()
    # plt.plot(window)
    # plt.pause(0.01)
    # split it to len(start_seq) pieces, and do fft on each piece
    frags = np.split(window, len(start_seq))
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
    print("Detected freq sequence: ", det_freq_seq)
    if freq_seq_cmp(det_freq_seq, start_seq):
        print("Found start sequence at index: ", i)
        real_truck = waveform[i:]
        break
if real_truck is None:
    print("Start sequence not found.")
    exit(1)

# good. step 2: decode the payload
payload_chunk = real_truck[start_seq_data_len:]
# split it to len(payload_seq) pieces, and do fft on each piece
frags = np.split(payload_chunk, len(payload_seq))
det_freq_seq = []
for frag in frags:
    xf, yf, top_freq, top_energies = fft_analyze(frag)
    det_freq_seq.append(top_freq[0].astype(int).item())
print("Detected payload freq sequence: ", det_freq_seq)