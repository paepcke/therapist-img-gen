#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-11-17 10:29:19
# @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/manual_corrections/round3_fear_and_checks.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-11-17 10:29:19
# **********************************************************

'''
Round 3 of the Nov 2026 client-image fixes. Same --out as rounds 1 and 2.
Needs: round 2 done (<out>/round2/sources), the Fear candidates made by
    python src/manual_corrections/generate_driving_candidates.py --out <out> --emotions Fear
(<out>/extra_driving/Fear--a..d.png), the OpenAI key and a GPU.

Does four things, writing only into <out>/round3:
 1. Fear candidates a-d on the neutralized shoulders source, 1.0/1.5/2.0.
 2. Head-bow retry: one GPT edit per identity that lowers the head much more
    than round 2 did ("bowed2"), then Sadness-c on it at 1.5 and 2.0.
 3. The picked Anxiety-d at 1.0 on the shoulders source (round 2 had 1.5 only).
 4. Emotions Jon has not seen on the neutralized neutral source, with the
    library's current driving photos and strengths: Contempt 2.0, Disgust 1.5,
    Happiness 0.5, Surprise 1.5.

Run on quintus, from the project root, in the therapist-img-gen env:
    python src/manual_corrections/round3_fear_and_checks.py --gpu 0 --out ~/imgproto_2026-11-16
'''
import argparse
import base64
import html
import json
import shutil
import sys
from pathlib import Path

IDENTITIES = ['asian_female_20s-30s_00', 'hispanic_male_40s-50s_00']

KEEP = ("Keep the person's identity, skin tone, hairstyle, clothing, the office "
        "background, lighting, camera angle and framing exactly unchanged. "
        "Give the person a completely neutral, relaxed facial expression: mouth "
        "closed with the lips relaxed and level, no smile, no smirk, eyes open "
        "normally. ")
BOW2 = ("Lower the head clearly: tilt it forward and down about 20 degrees so "
        "the chin moves toward the chest and the head hangs slightly, as in "
        "quiet sorrow. Keep the shoulders, arms and everything else as they "
        "are.")
OTHERS = {'Contempt': 2.0, 'Disgust': 1.5, 'Happiness': 0.5, 'Surprise': 1.5}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--proj-root', default='.')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--quality', default='high')
    a = ap.parse_args()
    root = Path(a.proj_root).resolve()
    out = Path(a.out).expanduser().resolve()
    if out == root or root in out.parents:
        sys.exit('--out must be outside the project tree')
    r2, r3 = out / 'round2', out / 'round3'
    (r3 / 'sources').mkdir(parents=True, exist_ok=True)
    (r3 / 'driving').mkdir(exist_ok=True)
    for need in (r2 / 'sources', out / 'extra_driving' / 'Fear--a.png'):
        if not need.exists():
            sys.exit(f'{need} missing: see this script\'s docstring')

    # 2a. Stronger head bow
    sys.path.insert(0, str(root / 'src'))
    from cloud_img_procurement.posture_variant_generator import PostureVariantGenerator
    gen = PostureVariantGenerator(quality=a.quality)
    for ident in IDENTITIES:
        dest = r3 / 'sources' / f'{ident}__bowed2.png'
        if dest.exists():
            print(f'reuse {dest.name}')
            continue
        bucket = ident.rsplit('_', 1)[0]
        src = root / 'assets' / 'client_library' / bucket / f'{ident}.png'
        try:
            with open(src, 'rb') as fd:
                resp = gen.client.images.edit(model=gen.model, image=fd,
                                              prompt=KEEP + BOW2,
                                              quality=a.quality, n=1)
            dest.write_bytes(base64.b64decode(resp.data[0].b64_json))
            print(f'wrote {dest.name}')
        except Exception as exc:
            print(f'FAILED {dest.name}: {type(exc).__name__}: {exc}')

    # Private copies of the library's other driving photos (pkl caches stay here):
    for emo in OTHERS:
        shutil.copy2(root / 'assets' / f'{emo}.png', r3 / 'driving' / f'{emo}.png')

    from image_gen.live_portrait_service import LivePortraitService
    svc = LivePortraitService(a.gpu)

    def src_of(ident, key):
        if key == 'bowed2':
            return r3 / 'sources' / f'{ident}__bowed2.png'
        return r2 / 'sources' / f'{ident}__{key}.png'

    jobs = []   # (group, label, source key, driving path, multiplier)
    for lab in 'abcd':
        for m in (1.0, 1.5, 2.0):
            jobs.append(('Fear', f'Fear-{lab} x{m}', 'shoulders',
                         out / 'extra_driving' / f'Fear--{lab}.png', m))
    for m in (1.5, 2.0):
        jobs.append(('Sadness', f'Sadness-c on bowed2 x{m}', 'bowed2',
                     out / 'extra_driving' / 'Sadness--c.png', m))
    jobs.append(('Anxiety', 'Anxiety-d on shoulders x1.0', 'shoulders',
                 out / 'extra_driving' / 'Anxiety--d.png', 1.0))
    for emo, m in OTHERS.items():
        jobs.append(('Others', f'{emo} on neutral x{m}', 'neutral',
                     r3 / 'driving' / f'{emo}.png', m))

    log, rows = [], {i: [] for i in IDENTITIES}
    for ident in IDENTITIES:
        for group, label, key, drv, m in jobs:
            src = src_of(ident, key)
            if not (src.exists() and drv.exists()):
                print(f'skip {ident} {label}: missing {src if not src.exists() else drv}')
                continue
            wdir = r3 / 'work' / ident / key
            wdir.mkdir(parents=True, exist_ok=True)
            dest = r3 / ident / (label.replace(' ', '_') + '.jpg')
            dest.parent.mkdir(exist_ok=True)
            try:
                res = svc.generate(src, drv, wdir, driving_multiplier=m)
                shutil.copy2(res['wfp'], dest)
                rows[ident].append((group, label, dest))
                log.append({'ident': ident, 'label': label, 'source': key,
                            'driving': str(drv), 'multiplier': m, 'file': str(dest)})
            except Exception as exc:
                print(f'FAILED {ident} {label}: {exc}')
                log.append({'ident': ident, 'label': label, 'error': str(exc)})
    (r3 / 'results.json').write_text(json.dumps(log, indent=1))

    parts = ['<!doctype html><meta charset="utf-8"><title>Round 3</title>'
             '<style>body{font:14px system-ui;margin:12px}.row{display:flex;'
             'flex-wrap:wrap;gap:8px;margin:6px 0 18px}.c{width:220px}'
             '.c img{width:220px;display:block}</style>']
    for ident in IDENTITIES:
        parts.append(f'<h2>{html.escape(ident)}</h2>')
        b2 = r3 / 'sources' / f'{ident}__bowed2.png'
        if b2.exists():
            parts.append(f'<div class="row"><div class="c"><img src="sources/{b2.name}">'
                         f'<div>bowed2 source</div></div></div>')
        for group in ('Fear', 'Sadness', 'Anxiety', 'Others'):
            parts.append(f'<h4>{group}</h4><div class="row">')
            for g, label, p in rows[ident]:
                if g == group:
                    parts.append(f'<div class="c"><img src="{ident}/{p.name}">'
                                 f'<div>{html.escape(label)}</div></div>')
            parts.append('</div>')
    (r3 / 'index.html').write_text('\n'.join(parts))
    print(f'Done: {len([l for l in log if "file" in l])} clips. See {r3}/index.html')


if __name__ == '__main__':
    main()
