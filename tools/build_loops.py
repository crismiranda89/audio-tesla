"""Cut constant-pitch engine loops from the GT3 recording.
For every target RPM and load (on = pulling, off = coasting/overrun) pick the cleanest
window, re-sample it in the crank-angle domain so pitch is exactly constant, then make a
seamless loop spanning a whole number of 4-stroke cycles. Also grab overrun pops."""
import numpy as np, json, subprocess, sys
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import resample_poly

MP3 = sys.argv[1]; OUT = sys.argv[2]
SR = 44100
d = np.load('gt3/rpm.npz'); rpm, sc, db = d['rpm'].astype(float), d['sc'], d['db']; hop = float(d['hop'])
r = median_filter(rpm, 7); ok = median_filter((sc > 0.95).astype(float), 5) > 0.5
dr = np.gradient(uniform_filter1d(r, 9)) / hop
x22 = np.memmap('gt3/mono22k.f32', dtype=np.float32, mode='r')

def load(t0, dur):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t0:.3f}', '-t', f'{dur:.3f}', '-i', MP3,
                        '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).astype(float)

def refine(t0, t1):
    """fine RPM track (100 Hz) by peak-picking the order-3 line around the coarse estimate"""
    y = np.array(x22[int(t0 * 22050) - 2048:int(t1 * 22050) + 2048], dtype=float)
    ts = np.arange(t0, t1, 0.01); out = []
    for t in ts:
        c = int((t - t0) * 22050) + 2048; seg = y[c - 2048:c + 2048] * np.hanning(4096)
        S = np.abs(np.fft.rfft(seg, 1 << 16)); f = np.fft.rfftfreq(1 << 16, 1 / 22050)
        rc = np.interp(t, np.arange(len(r)) * hop + 2048 / 22050, r)
        est = []
        for o in (3, 6):
            fc = rc / 60 * o; m = (f > fc * 0.94) & (f < fc * 1.06)
            i = np.argmax(S * m); a, b, cc = np.log(S[i - 1:i + 2] + 1e-12)
            dlt = 0.5 * (a - cc) / (a - 2 * b + cc + 1e-12)
            est.append(((i + dlt) * f[1] * 60 / o, S[i]))
        out.append(sum(e * w for e, w in est) / sum(w for _, w in est))
    return ts, median_filter(np.array(out), 9)

def pick(B, kind, used):
    L = int(0.8 / hop); cands = []
    for i in range(0, len(r) - L, 2):
        w = slice(i, i + L); rr = r[w]; m = rr.mean()
        if not ok[w].all() or abs(m - B) > B * 0.08: continue
        if any(abs(i * hop - u) < 3 for u in used): continue
        if np.abs(np.diff(rr)).max() > B * 0.035: continue
        s = dr[w].mean()
        if kind == 'on' and not (s > -100 and db[w].mean() > -16): continue
        if kind == 'off' and not (-1500 < s < -120): continue
        if kind == 'on' and s > 2500: continue
        pk = np.abs(x22[int(i * hop * 22050):int((i + L) * hop * 22050)]).max()
        if pk > 0.97: continue
        span = (rr.max() - rr.min()) / m
        score = sc[w].mean() * 2 - span * 10 - np.std(db[w]) * 0.2 - abs(m - B) / B * 10
        if kind == 'on': score += (db[w].mean() + 16) * 0.06
        cands.append((score, i * hop, m, s))
    cands.sort(key=lambda c: -c[0]); out = []
    for c in cands:
        if all(abs(c[1] - o[1]) > 2 for o in out): out.append(c)
        if len(out) >= 12: break
    return out

def validate(loop, R0):
    fr = 2048; hopv = 1024; f = np.fft.rfftfreq(8192, 1 / SR); comb = np.zeros(len(f), bool)
    for k in range(1, 25): comb |= np.abs(f - k * R0 / 120) < 4
    low = f < 1500; ratios = []; rms = []
    yy = np.tile(loop, 2)
    for i in range(0, len(loop), hopv):
        seg = yy[i:i + fr] * np.hanning(fr); P = np.abs(np.fft.rfft(seg, 8192)) ** 2
        ratios.append(P[comb & low].sum() / P[low].sum()); rms.append(np.sqrt((seg ** 2).mean()))
    rms = np.array(rms)
    return min(ratios), rms.max() / (rms.min() + 1e-9)

def make_loop(t0, dur=0.8):
    pad = 0.15
    y = load(t0 - pad, dur + 2 * pad)
    ts, rp = refine(t0 - pad + 0.05, t0 + dur + pad - 0.05)
    R0 = float(np.median(rp))
    # crank angle (revolutions) of the source as a function of source time
    tsrc = np.arange(len(y)) / SR + t0 - pad
    rr = np.interp(tsrc, ts, rp)
    phi = np.cumsum(rr / 60) / SR
    # output: constant R0. Source time for each output sample via inverse of phi
    phi_out = np.arange(phi[0], phi[-1], R0 / 60 / SR)
    up = 4; yu = resample_poly(y, up, 1)
    src_idx = np.interp(phi_out, phi, np.arange(len(y))) * up
    z = np.interp(src_idx, np.arange(len(yu)), yu)
    # loop over whole number of 4-stroke cycles (2 revs)
    cyc = 2 * 60 / R0 * SR
    ncyc = max(4, int(round(0.62 * SR / cyc)))
    Lp = int(round(ncyc * cyc)); X = int(0.06 * SR)
    st = int(pad * SR)
    seg = z[st:st + Lp + X]
    if len(seg) < Lp + X: raise RuntimeError('short')
    loop = seg[:Lp].copy()
    fade = np.sin(np.linspace(0, np.pi / 2, X)) ** 2
    loop[:X] = seg[:X] * fade + seg[Lp:Lp + X] * (1 - fade)   # tail flows into head
    loop -= loop.mean()
    rms = np.sqrt((loop ** 2).mean()); loop *= 0.18 / rms
    return loop, R0, float(rms), ncyc

ON = [2500, 2700, 3300, 3900, 4500, 5100, 5700, 6300, 6900, 7500, 8100, 8700, 9000]
OFF = [2500, 2700, 3300, 3900, 4500, 5100, 5700, 6300, 6900, 7500, 8100, 8700]
bank = []; used = []
for kind, bands in (('on', ON), ('off', OFF)):
    for B in bands:
        got = None
        for p in pick(B, kind, used):
            loop, R0, lvl, nc = make_loop(p[1], 0.8)
            q, dyn = validate(loop, R0)
            if q > 0.2 and dyn < 1.9 and all(abs(R0 - b['rpm']) > R0 * 0.04 for b in bank if b['kind'] == kind):
                got = p; break
        if got is None: print('none', kind, B); continue
        used.append(p[1]); print(f'   q={q:.2f} dyn={dyn:.2f}', end='')
        bank.append(dict(kind=kind, rpm=round(R0, 1), src=round(p[1], 2), level=round(20 * np.log10(lvl), 1), cycles=nc, data=loop))
        print(f'{kind} {B}: src {p[1]:7.1f}s  R0 {R0:6.0f}  slope {p[3]:6.0f}  {len(loop)/SR:.3f}s  lvl {20*np.log10(lvl):5.1f} dB')

# overrun pops: sharp transients during coasting
env = 20 * np.log10(uniform_filter1d(np.abs(np.array(x22, dtype=float)) ** 2, 220)[::220] + 1e-12) / 2  # 10 ms
base = median_filter(env, 51)
cand = np.where((env - base > 9) & (env > -26))[0]
pops = []; last = -1e9
for c in cand:
    t = c * 0.01
    if t - last < 2: continue
    k = int(t / hop)
    if not (-6000 < dr[min(k, len(dr) - 1)] < -200): continue
    last = t; pops.append(t)
print('pop candidates', len(pops))
pops = pops[:: max(1, len(pops) // 10)][:10]
for t in pops:
    y = load(t - 0.01, 0.18); i = np.argmax(np.abs(y[:int(0.05 * SR)]))
    y = y[max(0, i - int(0.004 * SR)):][:int(0.14 * SR)]
    y = y * np.minimum(1, np.linspace(0, 40, len(y))) * np.exp(-np.arange(len(y)) / (0.05 * SR))
    y /= np.abs(y).max() + 1e-9; y *= 0.8
    bank.append(dict(kind='pop', rpm=0, src=round(t, 2), level=0, cycles=0, data=y))

# pack: one mono track, 50 ms silence between clips
gap = np.zeros(int(0.05 * SR)); parts = []; meta = []; pos = len(gap); parts.append(gap)
for b in bank:
    meta.append(dict(kind=b['kind'], rpm=b['rpm'], start=pos / SR, len=len(b['data']) / SR, src=b['src']))
    parts += [b['data'], gap]; pos += len(b['data']) + len(gap)
pack = np.concatenate(parts).astype(np.float32)
pack /= max(1, np.abs(pack).max() / 0.98)
pack.tofile(OUT + '.f32')
json.dump(dict(sr=SR, clips=meta), open(OUT + '.json', 'w'), indent=1)
print('total', len(pack) / SR, 's')
