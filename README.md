# Aether — Hand-Driven Particle Field

A 3,000-particle 3D sphere that reacts to your hands in real time through your
webcam. Two implementations, same gestures:

- **`aether.py`** — Python, using OpenCV + MediaPipe + pygame
- **`web/index.html`** — a single self-contained HTML file, using Three.js + MediaPipe Hands, no build step, no install

| Gesture | Effect |
|---|---|
| One hand | Move and rotate the sphere |
| Pinch (thumb + index) and hold | Gold ring charges up |
| Release the pinch | Particles burst outward, then reform |
| Two hands | Spread apart to grow the sphere, bring together to shrink it |
| Finger count (0–5) | Shifts the particle color |
| "Webcam Background" button / `B` key | Toggles your camera feed as a fullscreen backdrop |

---

## Option A — Python version

### 1. Requirements
- Python 3.9–3.12
- A webcam

### 2. Set up in VS Code

Open this folder in VS Code, then open a terminal (`` Ctrl+` `` / `` Cmd+` ``) and run:

```bash
# create a virtual environment (keeps dependencies isolated)
python -m venv venv

# activate it
# macOS/Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt
```

If VS Code prompts "Select Interpreter," pick the one inside `venv`
(bottom-right of the window, or `Ctrl+Shift+P` → "Python: Select Interpreter").

### 3. Run it

```bash
python aether.py
```

or press `F5` / the ▶ Run button in VS Code with `aether.py` open.

The first run downloads a small hand-tracking model file (`hand_landmarker.task`,
a few MB) into the project folder — needs internet once, then it works offline.

### 4. Controls
- `Esc` or close the window to quit
- `B` or click **Webcam Background** (top right) to toggle the camera backdrop

### Troubleshooting
- **Camera doesn't open**: make sure no other app (Zoom, Teams, browser tab) is using it, and check `CAM_INDEX = 0` near the top of `aether.py` — try `1` if you have multiple cameras.
- **Low frame rate**: lower `PARTICLE_COUNT` near the top of the file (1500 still looks good).
- **`pip install mediapipe` fails**: mediapipe support lags behind the newest Python releases — Python 3.10 or 3.11 is the safest bet if 3.12+ gives you trouble.

---

## Option B — Browser version

No installation needed.

1. Open the `web/` folder in VS Code.
2. Install the **Live Server** extension (Ritwick Dey) — VS Code will suggest it automatically via `.vscode/extensions.json`.
3. Right-click `web/index.html` → **Open with Live Server**.
4. Allow camera access when prompted.

This serves the page over `http://localhost`, which browsers require for reliable camera permissions. Opening the file directly (`file://`) usually also works, but `localhost` is the safe path.

---

## Project structure

```
.
├── aether.py           # Python implementation
├── requirements.txt    # Python dependencies
├── web/
│   └── index.html       # Browser implementation (Three.js + MediaPipe Hands)
├── .vscode/
│   └── extensions.json # suggests Python + Live Server extensions
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
