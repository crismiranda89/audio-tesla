"""Encrypt the GT3 sample bank with a PIN so it can live in the public repo.

usage: python pack_gt3.py <bank.f32> <bank.json> <PIN> <out.bin>

Format: b'GT3E' | u32 iterations | 16-byte salt | 12-byte IV | AES-256-GCM ciphertext
Plaintext: u32 json_len | json (padded to even) | int16 mono PCM at 32 kHz
Key = PBKDF2-SHA256(PIN, salt, iterations). The browser does the same with WebCrypto.
The audio itself is never committed unencrypted.
"""
import sys, json, struct, os
import numpy as np
from scipy.signal import resample_poly
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

f32, js, pin, out = sys.argv[1:5]
ITER = 600_000
OUT_SR = 32000
# loops rejected after inspection (seam glitches / mis-tracked pitch)
DROP = {('on', 7699), ('on', 8580), ('off', 4474), ('off', 6739)}

meta = json.load(open(js)); x = np.fromfile(f32, np.float32).astype(float); SR = meta['sr']
clips = []; parts = []; pos = 0
gap = np.zeros(int(0.03 * OUT_SR))
for c in meta['clips']:
    if (c['kind'], round(c['rpm'])) in DROP: continue
    a = int(round(c['start'] * SR)); n = int(round(c['len'] * SR))
    y = x[a:a + n]
    if c['kind'] != 'pop':           # resample a 3x tiled copy so the loop seam stays clean
        z = resample_poly(np.tile(y, 3), OUT_SR // 100, SR // 100)
        L = len(z) / 3; z = z[int(round(L)):int(round(2 * L))]
    else:
        z = resample_poly(y, OUT_SR // 100, SR // 100)
    parts += [gap, z]; pos += len(gap)
    clips.append(dict(kind=c['kind'], rpm=round(c['rpm']), start=pos, len=len(z)))
    pos += len(z)
parts.append(gap)
pcm = np.concatenate(parts)
pcm = np.clip(pcm / max(1.0, np.abs(pcm).max() / 0.98), -1, 1)
pcm16 = (pcm * 32767).astype('<i2').tobytes()
hdr = json.dumps(dict(sr=OUT_SR, clips=clips), separators=(',', ':')).encode()
if len(hdr) % 2: hdr += b' '
plain = struct.pack('<I', len(hdr)) + hdr + pcm16

salt = os.urandom(16); iv = os.urandom(12)
key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(pin.encode())
ct = AESGCM(key).encrypt(iv, plain, None)
open(out, 'wb').write(b'GT3E' + struct.pack('<I', ITER) + salt + iv + ct)
print(f'{len(clips)} clips, {len(pcm) / OUT_SR:.1f}s audio, {len(ct) / 1024:.0f} KiB -> {out}')
for c in clips: print(' ', c['kind'], c['rpm'])
