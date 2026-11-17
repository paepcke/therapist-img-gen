#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-11-16 15:23:03
# @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/manual_corrections/expression_prototype.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-11-16 20:09:41
# **********************************************************

'''
Prototype for the Nov 2026 client-image fixes after Jon's review.

For two identities, regenerates the (emotion, posture) cells the trainer
actually shows -- Anger/fists_clenched, Anxiety/shoulders_raised,
Fear/shoulders_raised, Sadness/neutral -- at several driving_multiplier
values, and optionally with alternative driving photos. Writes ONLY into
--out (never into the repo's assets or library), and builds
<out>/index.html: one row per identity x emotion, with the neutral
source, the clip as Jon reviewed it, and every variant, labeled with its
multiplier.

Alternative driving photos: put files named <Emotion>--<label>.png (or
.jpg) into <out>/extra_driving/ before running, e.g. Anger--glare1.png.
Each runs at strengths 1.0, 1.5 and 2.0.

Run on quintus, from the project root, in the therapist-img-gen env:
    python src/manual_corrections/expression_prototype.py --gpu 0 --out ~/imgproto_2026-11-16
'''
import argparse
import html
import json
import shutil
import sys
from pathlib import Path

# Baseline multipliers (posture_expression_combiner.DRIVING_MULTIPLIERS;
# Fear had no calibrated value, 1.0) first, then the variants to try:
CELLS = {
    'Anger':   ('fists_clenched',  [2.0, 1.5, 3.0]),
    'Anxiety': ('shoulders_raised', [1.5, 1.0, 2.0]),
    'Fear':    ('shoulders_raised', [1.0, 1.5, 2.0]),
    'Sadness': ('neutral',          [2.0, 1.0, 3.0]),
    }
IDENTITIES = ['asian_female_20s-30s_00', 'hispanic_male_40s-50s_00']


def source_photo(lib, ident, posture):
    bucket = ident.rsplit('_', 1)[0]
    name = ident if posture == 'neutral' else f'{ident}_{posture}'
    return lib / bucket / f'{name}.png'


def reviewed_clip(animated, ident, posture, emotion):
    bucket = ident.rsplit('_', 1)[0]
    p = animated / bucket / ident / posture / f'{ident}--{emotion}.jpg'
    return p if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--proj-root', default='.', help='therapist-img-gen root')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    root = Path(a.proj_root).resolve()
    out = Path(a.out).expanduser().resolve()
    if out == root or root in out.parents:
        sys.exit('--out must be outside the project tree')
    (out / 'driving').mkdir(parents=True, exist_ok=True)
    (out / 'extra_driving').mkdir(exist_ok=True)

    # Imported only now, and the service is built before anything else
    # touches torch (it pins the GPU):
    from image_gen.live_portrait_service import LivePortraitService
    svc = LivePortraitService(a.gpu)

    lib = root / 'assets' / 'client_library'
    animated = root / 'assets' / 'client_library_animated'
    # Private copies of the driving photos, so LivePortrait's cached .pkl
    # files land in --out, not in the repo's assets/:
    drv = {}
    for emo in CELLS:
        src = root / 'assets' / f'{emo}.png'
        dst = out / 'driving' / src.name
        shutil.copy2(src, dst)
        drv[emo] = dst
    extras = []
    for p in sorted((out / 'extra_driving').glob('*')):
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg') and '--' in p.stem:
            extras.append((p.stem.split('--', 1)[0], p.stem.split('--', 1)[1], p))

    rows, log = [], []
    for ident in IDENTITIES:
        for emo, (posture, mults) in CELLS.items():
            src = source_photo(lib, ident, posture)
            if not src.exists():
                print(f'MISSING source {src}')
                continue
            row = {'ident': ident, 'emotion': emo, 'posture': posture,
                   'cells': [('source (no expression)', src)]}
            rev = reviewed_clip(animated, ident, posture, emo)
            if rev:
                row['cells'].append((f'as Jon reviewed (x{mults[0]})', rev))
            jobs = [(f'x{m}', drv[emo], m) for m in mults[1:]]
            jobs += [(f'{label} x{m2}', p, m2)
                     for e, label, p in extras if e == emo
                     for m2 in (1.0, 1.5, 2.0)]
            for label, driving, m in jobs:
                wdir = out / 'work' / ident / posture
                wdir.mkdir(parents=True, exist_ok=True)
                try:
                    res = svc.generate(src, driving, wdir, driving_multiplier=m)
                    dest = out / ident / f'{emo}__{label.replace(" ", "_")}.jpg'
                    dest.parent.mkdir(exist_ok=True)
                    shutil.copy2(res['wfp'], dest)
                    row['cells'].append((label, dest))
                    log.append({'ident': ident, 'emotion': emo, 'label': label,
                                'multiplier': m, 'driving': str(driving),
                                'file': str(dest)})
                except Exception as exc:
                    print(f'FAILED {ident} {emo} {label}: {exc}')
                    log.append({'ident': ident, 'emotion': emo, 'label': label,
                                'error': f'{type(exc).__name__}: {exc}'})
            rows.append(row)

    (out / 'results.json').write_text(json.dumps(log, indent=1))
    parts = ['<!doctype html><meta charset="utf-8"><title>Expression prototype</title>'
             '<style>body{font:14px system-ui;margin:12px}.row{display:flex;gap:8px;'
             'margin:6px 0 18px}.c{width:220px}.c img{width:220px;display:block}'
             'h3{margin:14px 0 2px}</style>']
    for r in rows:
        parts.append(f'<h3>{html.escape(r["ident"])} &mdash; {r["emotion"]} '
                     f'({r["posture"]})</h3><div class="row">')
        for label, p in r['cells']:
            rel = Path(p).resolve()
            try:
                rel = rel.relative_to(out)
            except ValueError:
                # Library images live outside --out: copy them in.
                rel = Path('ref') / f'{r["ident"]}_{r["emotion"]}_{Path(p).name}'
                (out / rel).parent.mkdir(exist_ok=True)
                shutil.copy2(p, out / rel)
            parts.append(f'<div class="c"><img src="{rel}"><div>{html.escape(label)}</div></div>')
        parts.append('</div>')
    (out / 'index.html').write_text('\n'.join(parts))
    print(f'Done: {len(log)} clips. Open {out}/index.html')


if __name__ == '__main__':
    main()
