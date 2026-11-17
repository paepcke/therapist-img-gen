#!/usr/bin/env python
# **********************************************************
# @Author: Andreas Paepcke
# @Date:   2026-11-16 19:55:34
# @File:   /Users/paepcke/VSCodeWorkspaces/therapist-img-gen/src/manual_corrections/generate_driving_candidates.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-11-17 10:29:00
# **********************************************************

'''
Makes candidate LivePortrait *driving* photos for Anger, Anxiety, Sadness
and Fear, from Jon's Nov 13 2026 review: the current driving photos make
Anger look away/up, Anxiety look angry, and Sadness look like affect
suppression. Each candidate is a GPT Image edit of one NEUTRAL reference
photo (assets/clientInTherapyOfficeIsolated.png), changing only the
facial expression, worded after Jon's comments.

Writes ONLY into --out:
    <out>/extra_driving/<Emotion>--<label>.png   (what expression_prototype.py
                                                  picks up)
    <out>/candidates.json                        (prompts, for the record)
    <out>/candidates.html                        (quick look sheet)

Reuses the project's key handling and image model
(cloud_img_procurement.posture_variant_generator), so the OpenAI key file
must be on this machine. Costs one image edit per candidate (12 by default).

Run on quintus, in the therapist-img-gen env, from the project root:
    python src/manual_corrections/generate_driving_candidates.py --out ~/imgproto_2026-11-16
'''
import argparse
import base64
import html
import json
import sys
from pathlib import Path

KEEP = ("Keep the person's identity, skin tone, hairstyle, clothing, the office "
        "background, lighting, camera angle, framing, head position and body "
        "pose exactly unchanged. Change ONLY the facial expression: ")

CANDIDATES = {
    'Anger': {
        'a': ("a clear expression of anger. The eyes glare directly into the "
              "camera. The eyebrows are lowered and drawn together, the upper "
              "eyelids are tense, the jaw is clenched and the lips are pressed "
              "firmly together."),
        'b': ("intense, controlled anger. A hard, direct stare straight into "
              "the camera, eyebrows pulled sharply down and together, lower "
              "eyelids raised and tight, nostrils slightly flared, mouth "
              "closed and firm."),
        'c': ("anger with the teeth just visible. Glaring straight into the "
              "camera, eyebrows lowered and drawn together, lips parted "
              "slightly to show clenched teeth, chin pushed slightly forward."),
        'd': ("restrained, cold anger. Steady direct eye contact with the "
              "camera, eyebrows lowered, a hard stare, lips thin and "
              "compressed, no raised eyebrows and no wide-open eyes."),
        },
    'Anxiety': {
        'a': ("anxious fear, clearly NOT anger. The inner ends of the eyebrows "
              "are raised and drawn together in a worried shape, the eyes are "
              "wide with the eyes looking at the camera, the lips are slightly "
              "parted and tense. The brows are not lowered, there is no glare "
              "and the jaw is not clenched."),
        'b': ("anxiety and apprehension. Worried, raised and slightly knitted "
              "eyebrows, wide eyes with visible whites above the iris, "
              "looking toward the camera, lips pressed together and pulled "
              "slightly sideways. No frown and no angry look."),
        'c': ("anxious distress. Eyebrows raised with the inner ends tilted up, "
              "eyes wide and slightly watery, mouth slightly open with the "
              "lower lip tense, as if bracing for bad news. Not angry."),
        'd': ("fearful anxiety. Eyes opened wide with more of the white "
              "showing, eyebrows raised high and pulled together, upper eyelids "
              "lifted, mouth tense with the lips stretched slightly back."),
        },
    'Sadness': {
        'a': ("sadness. The eyes are looking down, the eyelids lowered, the "
              "inner ends of the eyebrows raised, the mouth slightly open with "
              "relaxed lips, the corners of the mouth turned down. No pressed "
              "lips and no smile."),
        'b': ("deep sadness and grief. The gaze is cast downward, the eyes are "
              "glistening, the inner eyebrows are lifted, the lower lip is "
              "slightly trembling and the mouth is a little open."),
        'c': ("quiet sorrow. Eyes looking down and away, heavy upper eyelids, "
              "the eyebrows drawn up in the middle, the mouth slightly open "
              "and soft, the corners of the mouth pulled down."),
        'd': ("about to cry. Eyes cast down and wet, eyebrows raised in the "
              "middle, a slightly open mouth with the chin starting to "
              "crumple. The lips are not pressed together."),
        },
    'Fear': {
        'a': ("acute fear, clearly NOT surprise and NOT anger. The eyebrows are "
              "raised and drawn together, the upper eyelids lifted so the "
              "whites show above the iris, the eyes wide, the mouth slightly "
              "open with the lips stretched horizontally back, and the lower "
              "face tense."),
        'b': ("terror. Eyes very wide with whites showing, brows raised and "
              "pulled together in a flat, tense line, the mouth open a little "
              "with the lips pulled back and the lower lip tense, the chin "
              "drawn back slightly."),
        'c': ("frightened and frozen. Wide eyes, brows raised and drawn "
              "together, the lips parted and stretched sideways, a tense "
              "bracing look, as if something threatening is approaching. Not a "
              "surprised look."),
        'd': ("fear held in. Wide eyes with whites showing, brows raised and "
              "knit, the mouth closed but the lips stretched thin and pulled "
              "back sideways, the chin tense and drawn back, the face rigid."),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--proj-root', default='.', help='therapist-img-gen root')
    ap.add_argument('--out', required=True)
    ap.add_argument('--source', help='neutral photo to edit '
                    '(default assets/clientInTherapyOfficeIsolated.png)')
    ap.add_argument('--emotions', nargs='*', default=list(CANDIDATES))
    ap.add_argument('--quality', default='high')
    a = ap.parse_args()
    root = Path(a.proj_root).resolve()
    out = Path(a.out).expanduser().resolve()
    if out == root or root in out.parents:
        sys.exit('--out must be outside the project tree')
    src = Path(a.source) if a.source else root / 'assets' / 'clientInTherapyOfficeIsolated.png'
    if not src.exists():
        sys.exit(f'Source photo not found: {src}')
    ed = out / 'extra_driving'
    ed.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root / 'src'))
    from cloud_img_procurement.posture_variant_generator import PostureVariantGenerator
    gen = PostureVariantGenerator(quality=a.quality)

    record = []
    for emo in a.emotions:
        for label, desc in CANDIDATES[emo].items():
            dest = ed / f'{emo}--{label}.png'
            prompt = KEEP + desc
            if dest.exists():
                print(f'skip existing {dest.name}')
            else:
                try:
                    with open(src, 'rb') as fd:
                        resp = gen.client.images.edit(
                            model=gen.model, image=fd, prompt=prompt,
                            quality=a.quality, n=1)
                    dest.write_bytes(base64.b64decode(resp.data[0].b64_json))
                    print(f'wrote {dest.name}')
                except Exception as exc:
                    print(f'FAILED {emo}--{label}: {type(exc).__name__}: {exc}')
                    record.append({'emotion': emo, 'label': label,
                                   'prompt': prompt, 'error': str(exc)})
                    continue
            record.append({'emotion': emo, 'label': label, 'prompt': prompt,
                           'file': str(dest)})

    (out / 'candidates.json').write_text(json.dumps(record, indent=1))
    parts = ['<!doctype html><meta charset="utf-8"><title>Driving candidates</title>'
             '<style>body{font:14px system-ui;margin:12px}.row{display:flex;gap:8px;'
             'margin:6px 0 18px}.c{width:250px}.c img{width:250px;display:block}'
             '.p{font-size:11px;color:#555}</style>']
    for emo in a.emotions:
        parts.append(f'<h3>{emo}</h3><div class="row">')
        for r in record:
            if r['emotion'] == emo and 'file' in r:
                parts.append(f'<div class="c"><img src="extra_driving/{Path(r["file"]).name}">'
                             f'<div>{emo}--{r["label"]}</div>'
                             f'<div class="p">{html.escape(r["prompt"][len(KEEP):])}</div></div>')
        parts.append('</div>')
    (out / 'candidates.html').write_text('\n'.join(parts))
    print(f'Done. Look at {out}/candidates.html, then run expression_prototype.py '
          f'with the same --out.')


if __name__ == '__main__':
    main()
