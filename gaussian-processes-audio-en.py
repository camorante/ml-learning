"""Synthesized soundtrack for gaussian-processes-with-audio.mp4 (38.8 s).
Música ambiental en Re mayor + efectos sincronizados con los tiempos de la escena de Manim.
"""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 48000
DUR = 38.8
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
# (inicio, fin, acorde)
chords = [
    (0.0, 11.7, ["D3", "A3", "C#4", "F#4", "E4"]),     # Dmaj9
    (11.7, 17.1, ["B2", "F#3", "A3", "D4", "C#4"]),   # Bm9
    (17.1, 22.7, ["G2", "D3", "F#3", "B3", "A3"]),    # Gmaj9
    (22.7, 26.3, ["E3", "B3", "D4", "G4", "F#4"]),    # Em9
    (26.3, 32.4, ["A2", "E3", "G3", "D4", "B3"]),     # A7sus (tensión: la duda)
    (32.4, DUR, ["D3", "A3", "C#4", "F#4", "E4"]),    # Dmaj9 (resolución)
]
for t0, t1, notes in chords:
    d = t1 - t0 + 1.2  # solape para que los acordes se fundan
    voice = sum(pad_voice(hz(n), d) for n in notes)
    voice = sosfilt(butter(2, 110, "high", fs=SR, output="sos"), lowpass(voice, 2000))
    place(music, voice, max(0, t0 - 0.6), gain=0.16)
    # bajo
    root = notes[0][:-1] + str(int(notes[0][-1]) - 1)
    b = tt(d)
    bass = np.sin(2 * np.pi * hz(root) * b) * np.minimum(1, b / 0.8) * np.minimum(1, (d - b) / 1.2)
    place(music, np.clip(bass, -1, 1), max(0, t0 - 0.6), gain=0.07)

# Arpegio suave a 80 bpm (corcheas = 0.375 s), una octava arriba de las notas del acorde
step = 0.375
t = 4.6
pattern = [0, 2, 3, 4, 3, 2, 1, 2]
k = 0
while t < 36.5:
    notes = next(n for t0, t1, n in chords if t0 <= t < t1)
    nn = notes[1:][pattern[k % len(pattern)] % 4]
    f = hz(nn) * 2
    accent = 1.0 if k % 4 == 0 else 0.7
    place(music, pluck(f), t, gain=0.11 * accent, pan=0.35 * np.sin(k * 0.9))
    t += step
    k += 1

music = reverb(music, 2.6, 0.45, seed=1)
# fundido de entrada y salida
env = np.ones(N)
env[: int(SR * 1.5)] = np.linspace(0, 1, int(SR * 1.5))
fo = int(SR * 2.2)
env[-fo:] = np.linspace(1, 0, fo) ** 1.5
music *= env[:, None]

# ---------- Efectos ----------
sfx = np.zeros((N, 2))
D5 = ["D5", "F#5", "A5", "C#6", "E6", "F#6", "A6"]

# Título: brillo ascendente + campana al subtítulo
for i, n in enumerate(["D5", "F#5", "A5", "D6", "F#6"]):
    place(sfx, bell(hz(n), 2.0, 0.6), 0.05 + i * 0.22, 0.10, pan=-0.5 + i * 0.25)
place(sfx, bell(hz("A5"), 3.0), 1.45, 0.14)
place(sfx, whoosh(0.9, 2500, 300, 0.8), 3.55, 0.35)           # salida del título
place(sfx, thump(), 4.6, 0.2)                                 # ejes
place(sfx, whoosh(1.0, 400, 2500, 0.6), 4.6, 0.25)

# Cambios de subtítulo
for tc in (5.6, 11.7, 22.7, 26.3, 33.0):
    place(sfx, whoosh(0.7, 5000, 1500, 0.5), tc, 0.22)
place(sfx, whoosh(0.9, 300, 1200, 0.5), 6.4, 0.25)            # aparece la banda

# Curvas del prior: una nota de cristal por curva (LaggedStart, lag 0.2596 s)
for i in range(7):
    place(sfx, bell(hz(D5[i]), 2.2, 1.2), 7.2 + i * 0.2596, 0.085, pan=-0.6 + i * 0.2)

# Puntos: pop afinado + barrido de "asentamiento" mientras las curvas se ajustan
point_t = [12.5, 14.8, 17.1, 19.4]
point_n = ["D5", "F#5", "A5", "D6"]
for tp, n in zip(point_t, point_n):
    place(sfx, pop(hz(n)), tp + 0.02, 0.40)
    place(sfx, bell(hz(n) * 2, 1.2, 0.5), tp + 0.02, 0.06)
    place(sfx, whoosh(1.4, 3500, 350, 0.7), tp + 0.6, 0.30)

# Línea media: tono que sube y se resuelve en campana
place(sfx, glide(hz("A4"), hz("D5"), 1.6), 23.5, 0.22)
place(sfx, bell(hz("D6"), 2.5), 25.1, 0.12)
place(sfx, bell(hz("A6"), 2.5, 0.8), 25.12, 0.06, pan=0.3)

# Zona de duda
place(sfx, whoosh(0.8, 2000, 400, 0.6), 27.1, 0.25)           # se van las curvas
place(sfx, wobble(hz("A3"), 1.6), 27.95, 0.32, pan=0.15)        # "aquí dudo más"
place(sfx, wobble(hz("E4"), 1.4), 28.05, 0.14, pan=0.15)
place(sfx, bell(hz("C#6"), 2.0), 28.95, 0.14, pan=-0.35)        # "aquí estoy seguro"
place(sfx, bell(hz("E6"), 2.0), 29.05, 0.10, pan=-0.35)

# Cierre: acorde de campanas arpegiado + barrido final
place(sfx, whoosh(0.7, 4000, 1000, 0.5), 32.4, 0.2)
for i, n in enumerate(["D5", "A5", "C#6", "E6", "F#6"]):
    place(sfx, bell(hz(n), 3.5), 33.8 + i * 0.11, 0.08, pan=-0.4 + i * 0.2)
place(sfx, whoosh(1.0, 1500, 200, 0.6), 37.8, 0.25)

sfx = reverb(sfx, 1.6, 0.28, seed=2)

music *= 0.17
mix = music + sfx
mix /= np.max(np.abs(mix)) / 0.89
wavfile.write("soundtrack.wav", SR, (mix * 32767).astype(np.int16))
print("ok", mix.shape, DUR)

def db(x): return 20*np.log10(np.sqrt(np.mean(x**2))+1e-12)
s = np.max(np.abs(music + sfx)) / 0.89
print("music RMS dB", round(db(music/s),1), " sfx RMS dB", round(db(sfx/s),1))
print("music peak", round(20*np.log10(np.max(np.abs(music/s))),1), " sfx peak", round(20*np.log10(np.max(np.abs(sfx/s))),1))
for a,b in [(0,4.6),(7,11),(12,22),(23,26),(27,32),(33,38)]:
    i,j=int(a*SR),int(b*SR)
    print(f"{a}-{b}s music {db(music[i:j]/s):.1f} sfx {db(sfx[i:j]/s):.1f}")
