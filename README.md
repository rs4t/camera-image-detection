# Face + Gesture Reaction Board (MediaPipe)

This script opens your webcam and shows:

- Left side: live camera feed with face and hand tracking overlays
- Right side: output image based on your reaction

Current built-in reactions:

- `thumbs_up` -> shows `images/thumbs-up.png`
- `shocked face + both hands on/near head` -> shows `images/shocked.png`
- `victory` -> shows `images/victory.png`
- `pointing_up` -> shows `images/pointing-up.jpg`
- `two open palms` -> shows `images/two-open-palms.jpg`
- `one open palm` -> shows `images/one-open-palm.jpg`
- `pointing_up near your torso/center` -> shows `images/pointing-at-yourself.jpg`
- `two closed fists held up` -> shows `images/two-closed-fists-up.jpg`
- `one closed fist` -> shows `images/one-closed-fist.jpg`
- `idle (no movement for a short time)` -> shows `images/idle.jpg`
- `middle finger (one hand)` -> shows `images/middle-finger.png`
- `middle finger (both hands)` -> shows `images/double-middle-finger.png`
- `two-hand heart` -> shows `images/two-hand-heart.png`
- `T sign with both hands` -> shows `images/t-sign.jpg`
- `OK sign` -> shows `images/ok-sign.jpg`
- `tongue out` -> shows `images/tongue-out.jpg`
- `eyes closed for 0.5s` -> shows `images/sleep.jpg`
- `shush sign (index finger in front of mouth)` -> shows `images/shush-sign.jpg`
- fallback -> shows `images/neutral.png` (if present)

If those images are missing, the app falls back to idle or a placeholder card.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

On first run, the script auto-downloads two MediaPipe task models into `assets/models/`.

3. Run:

```bash
python app.py
```

Phone/virtual camera users can pick camera index and backend:

```bash
python app.py --list-cameras
python app.py --camera-index 1
python app.py --camera-index 1 --backend dshow
```

Useful flags:

- `--camera-index` camera number to open (default `0`)
- `--list-cameras` scans and prints working indices before launch
- `--scan-max-index` max index for scanning (default `8`)
- `--backend` one of `auto`, `any`, `dshow`, `msmf`
- `--hide-landmarks` starts with face/hand landmarks hidden

Press `Q` or `Esc` to quit.
Press `H` to toggle landmarks on/off while running.

## Add your own images

Create an `images` folder next to `app.py` and add:

- `images/neutral.png` (optional fallback)
- `images/thumbs-up.png`
- `images/shocked.png`
- `images/victory.png`
- `images/pointing-up.jpg` (used for pointing-up gesture)
- `images/two-open-palms.jpg` (two open palms)
- `images/one-open-palm.jpg` (one open palm)
- `images/pointing-at-yourself.jpg` (pointing at yourself)
- `images/two-closed-fists-up.jpg` (two closed fists held up)
- `images/one-closed-fist.jpg` (one closed fist)
- `images/idle.jpg` (idle/no movement)
- `images/middle-finger.png` (single middle finger)
- `images/double-middle-finger.png` (double middle finger)
- `images/two-hand-heart.png` (two-hand heart)
- `images/t-sign.jpg` (T sign)
- `images/ok-sign.jpg` (OK sign)
- `images/tongue-out.jpg` (tongue out)
- `images/sleep.jpg` (eyes closed)
- `images/shush-sign.jpg` (shush sign)

The current files in the `images` folder are placeholders.
You can replace any image with your own, as long as you keep the same filename.

Use any image size; the app auto-fits to the right panel without stretching.
Non-square images are letterboxed to preserve the full image.
If an action image is missing, the app silently falls back to the idle image.
If the idle image is missing too, it uses a text placeholder card instead of erroring.

You can also set panel size (each side is square):

```bash
python app.py --square-size 360
```

## Notes

- Shocked requires a shocked face plus two hands touching/near the head.
- You can tune values in `detect_shocked()` if it is too sensitive or not sensitive enough.
- Thumbs up is detected from MediaPipe Gesture Recognizer categories.
- The camera side keeps its original aspect ratio and is resized to the configured panel height (no square crop).
