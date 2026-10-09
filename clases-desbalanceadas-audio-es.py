"""Banda sonora sintetizada para desbalance-con-audio.mp4.
Misma paleta sonora que procesos-gaussianos-audio.py: música ambiental en Re mayor
+ efectos sincronizados con las marcas de tiempo que escribe la escena de Manim (marks.json).
"""
import json
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 48000
MK = json.load(open("marks.json"))
DUR = MK["total"][0]
N = int(SR * DUR)
rng = np.random.default_rng(3)


def hz(note):
    """'D4' -> Hz (notación anglosajona, sostenidos con #)."""
    names = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7,
             "G#": 8, "A": 9, "A#": 10, "B": 11}
    n, o = note[:-1], int(note[-1])
    midi = 12 * (o + 1) + names[n]
    return 440.0 * 2 ** ((midi - 69) / 12)


def tt(d):
    return np.arange(int(SR * d)) / SR


def lowpass(x, fc, order=2):
    return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x)


def bandpass(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def place(buf, sig, t, gain=1.0, pan=0.0):
    """Suma una señal mono en el buffer estéreo en el instante t (s), con paneo -1..1."""
    i = int(t * SR)
    if i >= buf.shape[0]:
        return
    sig = sig[: buf.shape[0] - i] * gain
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[i:i + len(sig), 0] += sig * l * np.sqrt(2)
    buf[i:i + len(sig), 1] += sig * r * np.sqrt(2)


def reverb(x, secs=2.2, wet=0.35, seed=0):
    r = np.random.default_rng(seed)
    n = int(SR * secs)
    env = np.exp(-np.linspace(0, 7, n))
    out = np.zeros_like(x)
    for ch in range(2):
        ir = r.standard_normal(n) * env
        ir = lowpass(ir, 6000)
        ir /= np.sqrt(np.sum(ir**2))
        out[:, ch] = fftconvolve(x[:, ch], ir)[: len(x)]
    return (1 - wet) * x + wet * out


# ---------- Instrumentos ----------
def bell(f, d=2.5, bright=1.0):
    t = tt(d)
    partials = [(1, 1.0, 1.0), (2.0, 0.35 * bright, 1.8), (3.01, 0.18 * bright, 2.6), (4.2, 0.08 * bright, 3.5)]
    s = sum(a * np.sin(2 * np.pi * f * m * t) * np.exp(-t * 2.2 * k) for m, a, k in partials)
    att = np.minimum(1, t / 0.004)
    return s * att * 0.5


def pluck(f, d=1.6):
    t = tt(d)
    s = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 6)
    return s * np.exp(-t * 3.2) * np.minimum(1, t / 0.006) * 0.5


def pad_voice(f, d, detune=0.12):
    t = tt(d)
    s = np.zeros_like(t)
    for c in (-detune, 0, detune):
        ff = f * 2 ** (c / 12)
        ph = rng.uniform(0, 2 * np.pi)
        # triángulo suave aproximado con 3 armónicos impares
        s += (np.sin(2 * np.pi * ff * t + ph) - np.sin(2 * np.pi * 3 * ff * t + ph) / 9
              + np.sin(2 * np.pi * 5 * ff * t + ph) / 25)
    a, r = min(1.2, d / 3), min(1.6, d / 2.5)
    env = np.minimum(1, t / a) * np.minimum(1, (d - t) / r)
    return s * np.clip(env, 0, 1) / 3


def pop(f):
    t = tt(0.35)
    freq = f * (1 + 1.5 * np.exp(-t * 60))
    ph = 2 * np.pi * np.cumsum(freq) / SR
    s = np.sin(ph) * np.exp(-t * 14)
    click = rng.standard_normal(len(t)) * np.exp(-t * 900) * 0.3
    return (s + click) * 0.8


def whoosh(d=0.7, f0=3000, f1=500, gain=1.0):
    """Ruido filtrado con barrido de frecuencia de f0 a f1."""
    n = int(SR * d)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    blocks = 40
    edges = np.linspace(0, n, blocks + 1).astype(int)
    for b in range(blocks):
        fc = f0 * (f1 / f0) ** (b / (blocks - 1))
        seg = bandpass(noise, max(60, fc * 0.6), min(SR / 2 - 100, fc * 1.6))
        out[edges[b]:edges[b + 1]] = seg[edges[b]:edges[b + 1]]
    t = np.arange(n) / n
    env = np.sin(np.pi * t) ** 1.5
    return out * env * 0.5 * gain


def glide(f0, f1, d):
    t = tt(d)
    freq = f0 * (f1 / f0) ** (t / d)
    ph = 2 * np.pi * np.cumsum(freq) / SR
    env = np.sin(np.pi * np.minimum(1, t / d) * 0.5) * np.exp(-np.maximum(0, t - d * 0.8) * 8)
    return (np.sin(ph) + 0.2 * np.sin(2 * ph)) * env * 0.4


def wobble(f, d=1.4):
    """Tono 'indeciso': vibrato lento y amplio."""
    t = tt(d)
    freq = f * (1 + 0.03 * np.sin(2 * np.pi * 5.5 * t))
    ph = 2 * np.pi * np.cumsum(freq) / SR
    s = np.sin(ph) + 0.25 * np.sin(2 * ph)
    return s * np.exp(-t * 2.0) * np.minimum(1, t / 0.02) * 0.5


def thump(f=55):
    t = tt(0.6)
    freq = f * (1 + 2 * np.exp(-t * 30))
    ph = 2 * np.pi * np.cumsum(freq) / SR
    return np.sin(ph) * np.exp(-t * 7) * 0.9


# ---------- Música ----------
music = np.zeros((N, 2))
C = MK["caption"]
DM9 = ["D3", "A3", "C#4", "F#4", "E4"]; BM9 = ["B2", "F#3", "A3", "D4", "C#4"]; GM9 = ["G2", "D3", "F#3", "B3", "A3"]
EM9 = ["E3", "B3", "D4", "G4", "F#4"]; A7S = ["A2", "E3", "G3", "D4", "B3"]; F7 = ["F#2", "C#3", "E3", "A3", "G#3"]
chords = [
    (0.0, C[1], DM9),              # mil camiones
    (C[1], C[2], BM9),             # el acierto engaña
    (C[2], C[3], GM9),             # bajar el límite
    (C[3], C[4], EM9),             # precio a cada error
    (C[4], MK["legend"][0], A7S),  # reequilibrar
    (MK["legend"][0], DUR, DM9),   # cierre
]
for t0, t1, notes in chords:
    d = t1 - t0 + 1.2
    voice = sum(pad_voice(hz(n), d) for n in notes)
    voice = sosfilt(butter(2, 110, "high", fs=SR, output="sos"), lowpass(voice, 2000))
    place(music, voice, max(0, t0 - 0.6), gain=0.16)
    root = notes[0][:-1] + str(int(notes[0][-1]) - 1)
    b = tt(d)
    bass = np.sin(2 * np.pi * hz(root) * b) * np.minimum(1, b / 0.8) * np.minimum(1, (d - b) / 1.2)
    place(music, np.clip(bass, -1, 1), max(0, t0 - 0.6), gain=0.07)

step, t, k = 0.375, C[0], 0
pattern = [0, 2, 3, 4, 3, 2, 1, 2]
while t < DUR - 2.5:
    notes = next(n for t0, t1, n in chords if t0 <= t < t1)
    f = hz(notes[1:][pattern[k % len(pattern)] % 4]) * 2
    place(music, pluck(f), t, gain=0.10 * (1.0 if k % 4 == 0 else 0.7), pan=0.35 * np.sin(k * 0.9))
    t += step
    k += 1

music = reverb(music, 2.6, 0.45, seed=1)
env = np.ones(N)
env[: int(SR * 1.5)] = np.linspace(0, 1, int(SR * 1.5))
fo = int(SR * 2.2)
env[-fo:] = np.linspace(1, 0, fo) ** 1.5
music *= env[:, None]

# ---------- Efectos ----------
sfx = np.zeros((N, 2))
for i, n in enumerate(["D5", "F#5", "A5", "D6", "F#6"]):                 # título
    place(sfx, bell(hz(n), 2.0, 0.6), 0.05 + i * 0.22, 0.10, pan=-0.5 + i * 0.25)
place(sfx, bell(hz("A5"), 3.0), MK["subtitle"][0] + 0.05, 0.14)
place(sfx, whoosh(0.9, 2500, 300, 0.8), MK["title_out"][0] - 0.05, 0.35)
for tc in C:
    place(sfx, whoosh(0.7, 5000, 1500, 0.5), tc, 0.22)

place(sfx, whoosh(1.4, 400, 2600, 0.4), MK["grid"][0], 0.12)
for i in range(10):
    place(sfx, pop(hz(["D5", "F#5", "A5", "E5", "B5", "D6", "A5", "F#5", "E5", "D5"][i])), MK["grid"][0] + 0.2 + i * 0.11, 0.07, pan=-0.6 + 0.13 * i)
place(sfx, bell(hz("D6"), 2.2), MK["acc"][0] + 0.1, 0.12)
place(sfx, thump(55), MK["zero"][0] + 0.1, 0.14); place(sfx, wobble(hz("A4"), 1.4), MK["zero"][0] + 0.3, 0.10)
place(sfx, whoosh(1.0, 600, 2400, 0.4), MK["lim"][0], 0.12)
for i, tc in enumerate(MK["step"]):
    place(sfx, pluck(hz(["D5", "E5", "F#5", "A5", "B5"][i])), tc + 0.05, 0.11, pan=-0.4 + 0.2 * i)
for i, tc in enumerate(MK["cost"]):
    place(sfx, glide(hz("D5"), hz(["A5", "F#5", "D5"][i]), 0.6), tc + 0.05, 0.10, pan=-0.3 + 0.3 * i)
place(sfx, bell(hz("F#6"), 2.5), MK["rule"][0] + 0.1, 0.11)
place(sfx, wobble(hz("D5"), 1.2), MK["infl"][0] + 0.1, 0.10)
place(sfx, glide(hz("A5"), hz("D5"), 0.7), MK["corr"][0] + 0.1, 0.12); place(sfx, bell(hz("D6"), 2.2), MK["corr"][0] + 0.8, 0.10)

place(sfx, whoosh(0.7, 4000, 1000, 0.5), MK["legend"][0], 0.2)
for i, n in enumerate(["D5", "A5", "C#6", "E6", "F#6"]):
    place(sfx, bell(hz(n), 3.5), MK["end_text"][0] + 0.3 + i * 0.11, 0.08, pan=-0.4 + i * 0.2)
place(sfx, whoosh(1.0, 1500, 200, 0.6), MK["fade"][0], 0.25)

sfx = reverb(sfx, 1.6, 0.28, seed=2)
music *= 0.17
mix = music + sfx
mix /= np.sqrt(np.mean(mix**2)) / 10 ** (-18.7 / 20)   # mismo nivel RMS que el resto de la serie
mix = 0.95 * np.tanh(mix / 0.95)                        # limitador suave para los picos
wavfile.write("soundtrack.wav", SR, (mix * 32767).astype(np.int16))
print("ok", mix.shape, DUR)
