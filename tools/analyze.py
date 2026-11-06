import numpy as np, sys
SR = 22050
x = np.fromfile(sys.argv[1], dtype=np.float32)
hop = 2205            # 0.1 s
win = 4096
nfft = 16384
w = np.hanning(win)
n = (len(x) - win) // hop
freqs = np.fft.rfftfreq(nfft, 1 / SR)
band = (freqs >= 30) & (freqs <= 2000)
f0 = np.zeros(n); rms = np.zeros(n); conf = np.zeros(n); cent = np.zeros(n)
cand = np.arange(35, 470, 0.5)  # firing-frequency candidates (Hz)
for i in range(n):
    seg = x[i * hop:i * hop + win]
    rms[i] = np.sqrt(np.mean(seg ** 2))
    S = np.abs(np.fft.rfft(seg * w, nfft))
    S = np.log1p(S / (S.max() + 1e-9) * 1000)
    df = freqs[1]
    # harmonic sum: score candidate f by mean log-magnitude at k*f, k=1..6 (vs. half-harmonics)
    sc = np.zeros(len(cand))
    for k in range(1, 7):
        idx = np.clip(np.round(cand * k / df).astype(int), 0, len(S) - 1)
        idxh = np.clip(np.round(cand * (k - 0.5) / df).astype(int), 0, len(S) - 1)
        sc += S[idx] - 0.5 * S[idxh]
    j = np.argmax(sc)
    f0[i] = cand[j]; conf[i] = sc[j] / (np.median(sc) + 1e-6)
    m = S[band]; cent[i] = (freqs[band] * m).sum() / (m.sum() + 1e-9)
np.savez(sys.argv[2], f0=f0, rms=rms, conf=conf, cent=cent)
print('frames', n)
