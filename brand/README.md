# Brand

## The mark

A "B" made of two equal spheres on a stem. The lower sphere is cut along three chords into
seven shards; the upper sphere is built from the same seven shards turned half a turn. That
is the Banach-Tarski paradox: one ball, cut into pieces and reassembled, becomes two. It is
also what Banach does with a model too big for your machine.

| File | Use |
|---|---|
| `logo-mark.svg`, `logo-512.png`, `logo-1024.png` | app icon, avatars (dark tile) |
| `logo-mark-transparent.svg`, `logo-transparent-512.png` | on dark backgrounds |
| `logo-lockup.svg`, `logo-lockup.png` | README header, slides, social cards |

Colours: background `#0A0B14`, ink `#F4F5FB`, muted `#8C90A8`, and the gradient
`#9A8BFF` (violet) to `#5BC8FF` (cyan) to `#6FFFD2` (mint). Type: Inter Display for
headings, Inter for text, a monospace for anything typed at a terminal.

## Launch film

`video/index.html` draws the film on a canvas from a single `render(t)` function, so every
frame is reproducible. `video/render.js` steps through it with Playwright and encodes with
ffmpeg; `video/soundtrack.py` synthesizes the audio (no samples, no licences).

```bash
python brand/video/soundtrack.py soundtrack.wav
node brand/video/render.js banach-16x9.mp4 1920 1080 30 --audio soundtrack.wav
node brand/video/render.js banach-9x16.mp4 1080 1920 30 --audio soundtrack.wav
```

Open `video/index.html` in a browser for a live preview.

Every number on screen comes from docs/VALIDATION.md. Predictions are labelled as
predictions, the streaming results name the model they were measured on, and the fidelity
chart is labelled as an illustration of the method.
