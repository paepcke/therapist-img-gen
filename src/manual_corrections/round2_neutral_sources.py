#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-11-16 20:28:41
# @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/manual_corrections/round2_neutral_sources.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-11-16 20:28:41
# **********************************************************

'''
Round 2 of the Nov 2026 client-image fixes (after Jon's Nov 13 review).

Round 1 (expression_prototype.py) found that the library's source photos
are not neutral: the woman has a closed-mouth smile, the man a smirk, and
the smile bleeds through the animated expression (her Anger came out as a
smile with teeth). This script:

 1. Makes neutralized sources with GPT Image edits of the ORIGINAL library
    photos (never an edit of an edit): the neutral, fists_clenched and
    shoulders_raised photos with a relaxed, mouth-closed, non-smiling
    face, plus a neutral photo with the head bowed slightly (for Sadness).
 2. Animates them with the round-1 driving photos (expects <out>/driving
    and <out>/extra_driving from round 1, same --out):
        Anger   on fists source:       Anger--a, b, c at 1.5, 2.0, 2.5
        Anxiety on shoulders source:   Anxiety--a, d at 1.0, 1.5, 2.0
        Sadness on bowed head source:  Sadness--a, c at 1.5, 2.0
        Sadness on neutral source:     Sadness--a at 1.5, 2.0 (for comparison)
        Fear    on shoulders source:   Fear (assets) at 1.5
 3. Writes <out>/round2/index.html: per identity, the original and
    neutralized sources, then the clips.

Writes ONLY into --out/round2. Needs the OpenAI key (as for round 1) and
a GPU. Existing neutralized sources are reused, so reruns are cheap; use
--skip-generate to only (re)animate.

Run on quintus, from the project root, in the therapist-img-gen env:
    python src/manual_corrections/round2_neutral_sources.py --gpu 0 --out ~/imgproto_2026-11-16
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
        "background, lighting, camera angle and framing exactly unchanged. ")
NEUTRAL_FACE = ("Give the person a completely neutral, relaxed facial expression: "
                "mouth closed with the lips relaxed and level, no smile and no "
                "smirk, no tension, eyes open normally, eyebrows in a resting "
                "position, looking at the camera. ")
KEEP_BODY = "Keep the body, arms and hands exactly as they are. "

# source key -> (library file suffix, extra instruction)
SOURCES = {
    'neutral':   ('',                  KEEP_BODY),
    'fists':     ('_fists_clenched',   KEEP_BODY + "The hands stay clenched into fists. "),
    'shoulders': ('_shoulders_raised', KEEP_BODY + "The shoulders stay raised. "),
    'bowed':     ('',                  "Also lower the head slightly so the chin tips down "
                                       "a little toward the chest, as if the person is "
                                       "looking down; the shoulders and everything else "
                                       "stay as they are. "),
    }

# (emotion, source key, driving label or None for assets/<Emotion>.png, strengths)
JOBS = [
    ('Anger',   'fists',     ['a', 'b', 'c'], [1.5, 2.0, 2.5]),
    ('Anxiety', 'shoulders', ['a', 'd'],      [1.0, 1.5, 2.0]),
    ('Sadness', 'bowed',     ['a', 'c'],      [1.5, 2.0]),
    ('Sadness', 'neutral',   ['a'],           [1.5, 2.0]),
    ('Fear',    'shoulders', [None],          [1.5]),
    ]


def library_photo(root, ident, suffix):
    bucket = ident.rsplit('_', 1)[0]
    return root / 'assets' / 'client_library' / bucket / f'{ident}{suffix}.png'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--proj-root', default='.', help='therapist-img-gen root')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True, help='the round-1 output folder')
    ap.add_argument('--skip-generate', action='store_true')
    ap.add_argument('--quality', default='high')
    a = ap.parse_args()
    root = Path(a.proj_root).resolve()
    out = Path(a.out).expanduser().resolve()
    if out == root or root in out.parents:
        sys.exit('--out must be outside the project tree')
    r2 = out / 'round2'
    (r2 / 'sources').mkdir(parents=True, exist_ok=True)
    for need in ('driving', 'extra_driving'):
        if not (out / need).is_dir():
            sys.exit(f'{out / need} missing: run round 1 first, same --out')

    # 1. Neutralized sources
    if not a.skip_generate:
        sys.path.insert(0, str(root / 'src'))
        from cloud_img_procurement.posture_variant_generator import PostureVariantGenerator
        gen = PostureVariantGenerator(quality=a.quality)
        for ident in IDENTITIES:
            for key, (suffix, extra) in SOURCES.items():
                dest = r2 / 'sources' / f'{ident}__{key}.png'
                if dest.exists():
                    print(f'reuse {dest.name}')
                    continue
                src = library_photo(root, ident, suffix)
                if not src.exists():
                    print(f'MISSING {src}')
                    continue
                prompt = KEEP + NEUTRAL_FACE + extra
                try:
                    with open(src, 'rb') as fd:
                        resp = gen.client.images.edit(
                            model=gen.model, image=fd, prompt=prompt,
                            quality=a.quality, n=1)
                    dest.write_bytes(base64.b64decode(resp.data[0].b64_json))
                    print(f'wrote {dest.name}')
                except Exception as exc:
                    print(f'FAILED {dest.name}: {type(exc).__name__}: {exc}')

    # 2. Animate (service built before anything else imports torch)
    from image_gen.live_portrait_service import LivePortraitService
    svc = LivePortraitService(a.gpu)
    log, rows = [], {}
    for ident in IDENTITIES:
        rows[ident] = []
        for emo, key, labels, mults in JOBS:
            src = r2 / 'sources' / f'{ident}__{key}.png'
            if not src.exists():
                print(f'no source {src.name}; skipping {emo}')
                continue
            for label in labels:
                if label is None:
                    drv = out / 'driving' / f'{emo}.png'
                    tag = emo
                else:
                    drv = out / 'extra_driving' / f'{emo}--{label}.png'
                    tag = f'{emo}-{label}'
                if not drv.exists():
                    print(f'no driving photo {drv}')
                    continue
                for m in mults:
                    wdir = r2 / 'work' / ident / key
                    wdir.mkdir(parents=True, exist_ok=True)
                    dest = r2 / ident / f'{emo}__{key}__{label or "std"}_x{m}.jpg'
                    dest.parent.mkdir(exist_ok=True)
                    try:
                        res = svc.generate(src, drv, wdir, driving_multiplier=m)
                        shutil.copy2(res['wfp'], dest)
                        rows[ident].append((f'{tag} on {key} x{m}', dest))
                        log.append({'ident': ident, 'emotion': emo, 'source': key,
                                    'driving': str(drv), 'multiplier': m,
                                    'file': str(dest)})
                    except Exception as exc:
                        print(f'FAILED {ident} {tag} {key} x{m}: {exc}')
                        log.append({'ident': ident, 'emotion': emo, 'source': key,
                                    'multiplier': m,
                                    'error': f'{type(exc).__name__}: {exc}'})
    (r2 / 'results.json').write_text(json.dumps(log, indent=1))

    # 3. Sheet
    parts = ['<!doctype html><meta charset="utf-8"><title>Round 2</title>'
             '<style>body{font:14px system-ui;margin:12px}.row{display:flex;'
             'flex-wrap:wrap;gap:8px;margin:6px 0 18px}.c{width:220px}'
             '.c img{width:220px;display:block}</style>']
    for ident in IDENTITIES:
        parts.append(f'<h2>{html.escape(ident)}</h2><h4>Sources (neutralized)</h4><div class="row">')
        for key in SOURCES:
            p = r2 / 'sources' / f'{ident}__{key}.png'
            if p.exists():
                parts.append(f'<div class="c"><img src="sources/{p.name}"><div>{key}</div></div>')
        parts.append('</div><h4>Clips</h4><div class="row">')
        for label, p in rows[ident]:
            parts.append(f'<div class="c"><img src="{ident}/{p.name}"><div>{html.escape(label)}</div></div>')
        parts.append('</div>')
    (r2 / 'index.html').write_text('\n'.join(parts))
    print(f'Done: {len([l for l in log if "file" in l])} clips. See {r2}/index.html')


if __name__ == '__main__':
    main()
