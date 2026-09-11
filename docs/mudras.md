# Mudra Reference & Recording Protocol

## Starter set (5 mudras)

| Label (used in code/filenames) | Sanskrit name | Type | Notes |
|---|---|---|---|
| `pataka` | Pataka | static | Flat hand, fingers together, thumb bent in — baseline shape |
| `tripataka` | Tripataka | static | Pataka with ring finger bent — confusable with Pataka, good stress-test |
| `mushti` | Mushti | static | Clenched fist, thumb across fingers — visually distinct, sanity-check shape |
| `suchi` | Suchi | transition | Neutral hand → index finger extended, others curled |
| `kartarimukha` | Kartarimukha | transition | Neutral hand → index + middle finger extended in a "scissor" shape — confusable with Suchi mid-transition |

Watch credentialed teacher reference videos for each mudra before recording. Note the
canonical held shape and common technique deviations for each.

## Directory / filename convention

Raw recordings go under `data/raw_videos/<mudra_label>/<clip_type>/`, where
`clip_type` is `static` or `transition`.

Filename format (fields separated by double underscore):

```
<subject>__<angle>__<distance>__<lighting>__<clip_idx>.mp4
```

- `subject`: `me` for the initial solo dataset (keep this field even though it's
  constant for now — it makes adding other people later a pure append, not a
  schema change)
- `angle`: `front`, `left30`, `right30`, `above`, `below` (pick a consistent short set)
- `distance`: `close`, `medium`, `far`
- `lighting`: `bright`, `dim`
- `clip_idx`: zero-padded two-digit take number, e.g. `01`, `02`

Example: `data/raw_videos/pataka/static/me__front__medium__bright__01.mp4`

`src/extract_landmarks.py` parses this filename to populate the metadata columns
in the landmark CSV — keep the convention consistent or extraction will
mis-tag clips.

## Recording checklist per mudra

- [ ] At least 2 angles (front + one side) × 2 distances × 2 lighting conditions
- [ ] Static mudras: a few seconds of the held pose per clip
- [ ] Transition mudras: neutral hand → into the shape → hold, captured as one
      continuous clip (don't pre-cut to just the final pose)
- [ ] Do a pilot recording (1 mudra, all condition variants) first and run it
      through `extract_landmarks.py` before recording the full set, to confirm
      MediaPipe tracking holds up at your worst-case angle/lighting/distance
