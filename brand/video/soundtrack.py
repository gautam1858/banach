"""Synthesize the launch film soundtrack (no samples, no licences): an ambient pad, a
filtered shatter at the crack, a low swell as the B forms, and a bell at the logo reveal.
Usage: python soundtrack.py OUT.wav"""
import sys
import wave

import numpy as np

SR, DUR = 48000, 44.0
t = np.arange(int(SR * DUR)) / SR
rng = np.random.default_rng(3)


def env(a, b, attack, release):
    e = np.clip((t - a) / attack, 0, 1) * np.clip((b - t) / release, 0, 1)
    return e * e * (3 - 2 * e)


def onepole(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):  # fine for a few seconds of audio
        acc = (1 - a) * x[i] + a * acc
        y[i] = acc
    return y


def pad():
    # A major 9, gently detuned, breathing
    notes = [110.0, 164.81, 220.0, 277.18, 329.63, 493.88]
    out = np.zeros_like(t)
    for k, f in enumerate(notes):
        for d in (-0.6, 0.6):
            ph = rng.uniform(0, 2 * np.pi)
            out += np.sin(2 * np.pi * (f + d) * t + ph) * (0.5 if k == 0 else 0.32 / (1 + 0.3 * k))
    breathe = 0.75 + 0.25 * np.sin(2 * np.pi * t / 9.0)
    return out * breathe * env(0.0, DUR, 3.0, 3.0) * 0.06


def shatter(at):
    n = int(SR * 1.6)
    s = np.zeros_like(t)
    i0 = int(at * SR)
    noise = rng.standard_normal(n)
    hp = noise - onepole(noise, 1800.0)  # bright, glassy
    decay = np.exp(-np.arange(n) / SR * 5.0)
    tick = np.zeros(n)
    for _ in range(40):  # little shards of sound
        j = rng.integers(0, int(SR * 0.6))
        L = int(SR * 0.03)
        f = rng.uniform(2500, 6000)
        tick[j:j + L] += np.sin(2 * np.pi * f * np.arange(L) / SR) * np.exp(-np.arange(L) / (SR * 0.006)) * 0.4
    seg = (hp * 0.5 + tick) * decay
    s[i0:i0 + n] = seg[: len(s) - i0]
    return s * 0.25


def swell(a, b):
    f = 55.0
    return np.sin(2 * np.pi * f * t) * env(a, b, 1.2, 1.4) * 0.12


def bell(at, f0=659.25):
    s = np.zeros_like(t)
    i0 = int(at * SR)
    n = int(SR * 5)
    tt = np.arange(n) / SR
    tone = sum(a * np.sin(2 * np.pi * f0 * r * tt) * np.exp(-tt * d)
               for r, a, d in [(1, 1.0, 0.9), (2.01, 0.45, 1.6), (2.98, 0.25, 2.4), (4.16, 0.12, 3.5)])
    s[i0:i0 + n] = tone[: len(s) - i0]
    return s * 0.12


mix = pad() + shatter(5.9) + swell(7.4, 10.4) + bell(38.6) + bell(38.9, 987.77) * 0.6
# tiny stereo width: delay the right channel by 11 ms
right = np.concatenate([np.zeros(int(0.011 * SR)), mix])[: len(mix)]
fade = np.clip(t / 0.4, 0, 1) * np.clip((DUR - t) / 1.5, 0, 1)
st = np.stack([mix * fade, right * fade], axis=1)
st /= max(1e-9, np.abs(st).max()) / 0.8
pcm = (st * 32767).astype("<i2")
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", sys.argv[1])
