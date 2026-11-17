#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-11-17 13:56:19
# @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/manual_corrections/round4_anger_gaze.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-11-17 13:56:19
# **********************************************************

'''
Round 4 of the Nov 2026 client-image fixes: Anger gaze. Same --out as
rounds 1-3. Jon asked for a direct glare at the viewer. In rounds 1-3 the
Anger clips had lowered brows but irises above the camera line, because
LivePortrait copies gaze from the Anger driving photo.

Needs: round 2 done (<out>/round2/sources/<id>__fists.png and __neutral.png),
the OpenAI key and a GPU.

Does, writing only into <out>/round4:
 1. 12 new Anger driving photos (6 prompts x 2 samples, labels e1,e2 ... j1,j2),
    GPT edits of the neutral reference photo, all stressing a level, centered,
    unblinking stare into the lens. They also go to <out>/extra_driving/.
 2. Each one animated onto the neutralized fists source of both identities at
    2.0 and 2.5.
 3. A gaze score for every clip, and for the neutral source as the reference:
    mean deviation of the irises from the eye-box center, horizontally and
    vertically (0 = centered on the camera; looking up gives a negative
    vertical value). Needs mediapipe; without it clips are made unscored
    and you judge from the eye strips in index.html.
 4. index.html: per identity, clips sorted by score, each with an eye-region
    strip beside the full frame.

Run on quintus, from the project root, in the therapist-img-gen env:
    python src/manual_corrections/round4_anger_gaze.py --gpu 0 --out ~/imgproto_2026-11-16
'''
import argparse
import base64
import html
import json
import shutil
import sys
from pathlib import Path

IDENTITIES = ['asian_female_20s-30s_00', 'hispanic_male_40s-50s_00']
STRENGTHS = [2.0, 2.5]

KEEP = ("Keep the person's identity, skin tone, hairstyle, clothing, the office "
        "background, lighting, camera angle, framing, head position and body "
        "pose exactly unchanged. Change ONLY the facial expression: ")
GAZE = ("Both eyes are level and perfectly centered, the irises exactly in the "
        "middle of the eye openings, pupils pointed straight into the camera "
        "lens, not looking up, down or to either side. ")

PROMPTS = {
    'e': GAZE + ("Anger. The eyebrows are lowered and drawn together, the "
                 "upper eyelids tense, the jaw clenched, the lips pressed "
                 "firmly together. An unblinking, direct glare at the viewer."),
    'f': GAZE + ("Cold, hard anger. A fixed stare straight at the viewer, "
                 "eyebrows pulled down and together, lower eyelids tight, "
                 "nostrils slightly flared, mouth closed and firm. The chin is "
                 "slightly lowered but the eyes stay level on the lens."),
    'g': GAZE + ("Furious anger. Eyebrows sharply lowered and knitted, a "
                 "piercing direct glare into the lens, lips parted slightly "
                 "showing clenched teeth."),
    'h': GAZE + ("Restrained anger. Steady, intense eye contact with the "
                 "camera, eyebrows lowered, lips thin and compressed. No "
                 "raised eyebrows, no wide-open eyes."),
    'i': GAZE + ("Angry confrontation. The person stares straight into the "
                 "camera as if challenging the viewer: brows low and "
                 "together, eyes narrowed slightly, jaw set."),
    'j': GAZE + ("Seething anger. Brows lowered, eyes narrowed and locked "
                 "directly on the camera, lips pressed hard together, the "
                 "muscles at the jaw tight."),
    }
SAMPLES = (1, 2)


def gaze_scorer():
    '''Returns f(image_path) -> (dx, dy, score) or None if mediapipe is
    missing. dx, dy are iris offsets from the eye-box center as a fraction of
    eye width/height (dy < 0: iris high, looking up); score = |dx|+|dy|,
    lower is better. Averaged over both eyes.'''
    try:
        import cv2
        import mediapipe as mp
        mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                               max_num_faces=1,
                                               refine_landmarks=True)
    except Exception as exc:
        print(f'No gaze scoring ({type(exc).__name__}: {exc}); clips unscored.')
        return None
    # (outer corner, inner corner, upper lid, lower lid, iris center)
    eyes = [(33, 133, 159, 145, 468), (263, 362, 386, 374, 473)]

    def score(path):
        img = cv2.imread(str(path))
        if img is None:
            return None
        res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        dxs, dys = [], []
        for o, i, u, l, c in eyes:
            w = lm[i].x - lm[o].x
            h = lm[l].y - lm[u].y
            if abs(w) < 1e-6 or abs(h) < 1e-6:
                return None
            dxs.append((lm[c].x - (lm[o].x + lm[i].x) / 2) / abs(w))
            dys.append((lm[c].y - (lm[u].y + lm[l].y) / 2) / abs(h))
        dx, dy = sum(dxs) / 2, sum(dys) / 2
        return dx, dy, abs(dx) + abs(dy)
    return score


def eye_strip(path, dest, height=150):
    '''Crops the eye region and saves it (fixed boxes, as in the manual
    checks: x .30-.70 of the width, y from .10 (face crops) to .16).'''
    try:
        from PIL import Image
        im = Image.open(path).convert('RGB')
        w, h = im.size
        y0 = int(h * 0.12)
        cr = im.crop((int(w * .30), y0, int(w * .70), y0 + int(w * .40 * .47)))
        cr = cr.resize((int(height * cr.width / cr.height), height))
        cr.save(dest, quality=90)
        return True
    except Exception as exc:
        print(f'eye strip failed for {path}: {exc}')
        return False


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
    r2, r4 = out / 'round2', out / 'round4'
    ed = out / 'extra_driving'
    for sub in ('strips', 'work'):
        (r4 / sub).mkdir(parents=True, exist_ok=True)
    ed.mkdir(parents=True, exist_ok=True)
    for ident in IDENTITIES:
        for key in ('fists', 'neutral'):
            need = r2 / 'sources' / f'{ident}__{key}.png'
            if not need.exists():
                sys.exit(f'{need} missing: run round 2 first')

    # 1. New Anger driving photos
    ref = root / 'assets' / 'clientInTherapyOfficeIsolated.png'
    if not ref.exists():
        sys.exit(f'Reference photo not found: {ref}')
    sys.path.insert(0, str(root / 'src'))
    from cloud_img_procurement.posture_variant_generator import PostureVariantGenerator
    gen = PostureVariantGenerator(quality=a.quality)
    drivers = []     # (label, path)
    for lab, desc in PROMPTS.items():
        for n in SAMPLES:
            name = f'{lab}{n}'
            dest = ed / f'Anger--{name}.png'
            if not dest.exists():
                try:
                    with open(ref, 'rb') as fd:
                        resp = gen.client.images.edit(
                            model=gen.model, image=fd, prompt=KEEP + desc,
                            quality=a.quality, n=1)
                    dest.write_bytes(base64.b64decode(resp.data[0].b64_json))
                    print(f'wrote {dest.name}')
                except Exception as exc:
                    print(f'FAILED Anger--{name}: {type(exc).__name__}: {exc}')
                    continue
            drivers.append((name, dest))
    (r4 / 'prompts.json').write_text(json.dumps(
        {k: KEEP + v for k, v in PROMPTS.items()}, indent=1))

    # 2. Animate onto the fists source; 3. score
    from image_gen.live_portrait_service import LivePortraitService
    svc = LivePortraitService(a.gpu)
    scorer = gaze_scorer()
    (r4 / 'driving').mkdir(exist_ok=True)

    log, rows = [], {i: [] for i in IDENTITIES}
    for ident in IDENTITIES:
        base = {}
        if scorer:
            base = scorer(r2 / 'sources' / f'{ident}__neutral.png')
        rows[ident].append(('neutral source (reference)',
                            r2 / 'sources' / f'{ident}__neutral.png', base))
        fists = r2 / 'sources' / f'{ident}__fists.png'
        for name, drv in drivers:
            # private copy so the .pkl motion cache stays out of extra_driving
            pdrv = r4 / 'driving' / drv.name
            if not pdrv.exists():
                shutil.copy2(drv, pdrv)
            for m in STRENGTHS:
                label = f'Anger-{name} x{m}'
                wdir = r4 / 'work' / ident / f'{name}_{m}'
                wdir.mkdir(parents=True, exist_ok=True)
                dest = r4 / ident / f'Anger-{name}_x{m}.jpg'
                dest.parent.mkdir(exist_ok=True)
                try:
                    res = svc.generate(fists, pdrv, wdir, driving_multiplier=m)
                    shutil.copy2(res['wfp'], dest)
                except Exception as exc:
                    print(f'FAILED {ident} {label}: {exc}')
                    log.append({'ident': ident, 'label': label, 'error': str(exc)})
                    continue
                sc = scorer(dest) if scorer else None
                eye_strip(dest, r4 / 'strips' / f'{ident}__{dest.stem}.jpg')
                rows[ident].append((label, dest, sc))
                log.append({'ident': ident, 'label': label, 'driving': str(drv),
                            'multiplier': m, 'file': str(dest),
                            'dx': sc[0] if sc else None,
                            'dy': sc[1] if sc else None,
                            'score': sc[2] if sc else None})
        eye_strip(r2 / 'sources' / f'{ident}__neutral.png',
                  r4 / 'strips' / f'{ident}__neutral_source.jpg')
    (r4 / 'results.json').write_text(json.dumps(log, indent=1))

    # 4. Sheet, best gaze first
    parts = ['<!doctype html><meta charset="utf-8"><title>Round 4: Anger gaze</title>'
             '<style>body{font:14px system-ui;margin:12px}.row{display:flex;'
             'flex-wrap:wrap;gap:10px;margin:6px 0 18px}.c{width:300px}'
             '.c img{width:300px;display:block}.s{font-size:12px;color:#444}'
             '</style><p>Score = |dx|+|dy| of iris offset from eye center '
             '(0 is looking at the camera); dy&lt;0 means looking up. The '
             'neutral source is the reference.</p>']
    for ident in IDENTITIES:
        parts.append(f'<h2>{html.escape(ident)}</h2><div class="row">')
        ref_row, clips = rows[ident][0], rows[ident][1:]
        clips.sort(key=lambda r: (r[2] is None, r[2][2] if r[2] else 0))
        for label, p, sc in [ref_row] + clips:
            rel = p.relative_to(out).as_posix()
            stem = 'neutral_source' if 'reference' in label else Path(p).stem
            s = (f'dx {sc[0]:+.2f} dy {sc[1]:+.2f} score {sc[2]:.2f}'
                 if sc else 'unscored')
            parts.append(f'<div class="c"><img src="strips/{ident}__{stem}.jpg">'
                         f'<img src="{rel}"><div>{html.escape(label)}</div>'
                         f'<div class="s">{s}</div></div>')
        parts.append('</div>')
    (r4 / 'index.html').write_text('\n'.join(parts))
    print(f'Done: {len([l for l in log if "file" in l])} clips. See {r4}/index.html')


if __name__ == '__main__':
    main()
