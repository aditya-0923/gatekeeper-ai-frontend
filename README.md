# GateKeeper AI 🛡️

Context-aware virtual fencing and access control in the browser. Draw a restricted zone on a live camera feed, enroll the people who are allowed inside, and get an alert when anyone else steps in.

> This repository is a fork of [vikassinngh123/gatekeeper-ai](https://github.com/vikassinngh123/gatekeeper-ai). The original project is a Streamlit app. This fork adds a **FastAPI backend and a custom web frontend**, person tracking, and a simpler setup. Credit for the original idea and the detection pipeline goes to the upstream authors.

<!-- Add a screenshot or GIF here: docs/demo.gif -->

## What it does

- **Draw a zone** by clicking on the camera view. Drag corners to adjust. The zone is saved in your browser.
- **Enroll faces** from your webcam. Enrolled faces are kept between restarts.
- **Watch the feed.** People outside the zone are blue, authorized people inside are green, unknown people inside are red.
- **Get alerts** in an event log, as a red flash on the video, and optionally as a sound.
- **Try it without a backend.** Open `index.html` on its own and you get a simulated demo scene.

## How it works

```
Browser camera ──frame (JPEG)──▶ POST /api/analyze
                                      │
                      YOLOv8n + ByteTrack: find people, give each a track ID
                                      │
              foot point (bottom-centre of box) inside the zone?
                          │ no                 │ yes
                     blue box        already recognised? ── yes ─▶ reuse identity
                                                 │ no
                                      YuNet finds the face in the head area
                                      SFace turns it into a 128-number embedding
                                      cosine match against enrolled faces
                                          │                    │
                                     match: green        no match: red + alert
```

Why it is built this way:

1. **One point, not the whole box.** Only a single point of a person's box is tested against the zone, so an arm or head leaning over the line does not trigger it. By default (*Auto*) that point is their feet. If the box touches the bottom of the frame, as with a laptop webcam where feet are not visible, the body centre is used instead. You can force either one in the Live monitor panel.
2. **Faces only when needed.** Face recognition runs only for people inside the zone, and stops once someone is recognised. The tracker ID carries their identity, which saves compute.
3. **Alert after a short confirmation.** An unknown person must stay unrecognised for a few frames before the alarm fires, so a turned head does not cause a false alert.

## What changed from upstream

| Area | Upstream | This fork |
|---|---|---|
| Interface | Streamlit tabs | Custom web UI served by FastAPI |
| Zone drawing | Sliders / inputs | Click to place, drag to move, saved in the browser |
| People | Detected frame by frame | Tracked with ByteTrack, identity remembered per track |
| Enrolled faces | Session only | Saved to disk and reloaded on restart (can be turned off) |
| Alerts | Box colours | Event log, red flash, optional sound, live counts |
| Setup | Several requirements files | One `requirements.txt`, env-var settings, Dockerfile |
| Input checks | None | Image size, name length and zone shape are validated |

The original Streamlit prototype was removed from this fork. You can still find it in the [upstream repository](https://github.com/vikassinngh123/gatekeeper-ai).

## Quick start

You need **Python 3.11** and a webcam.

```bash
git clone https://github.com/aditya-0923/gatekeeper-ai-frontend.git
cd gatekeeper-ai-frontend

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate

pip install -r requirements.txt
python server.py
```

Then open <http://localhost:8000>. Browsers only allow camera access on `localhost` or HTTPS.

### Download the models

Put these three files next to `server.py` (they are already in this repo, but if you cloned without them):

| File | Source |
|---|---|
| `yolov8n.pt` | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo: YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `face_recognition_sface_2021dec.onnx` | [OpenCV Zoo: SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) |

### Using it

1. **Enroll faces.** Type a name, click *Open camera*, then *Capture photo*. Use a front-facing photo in good light.
2. **Draw zone.** Click at least three corners on the video. Or use *Use sample zone*.
3. **Live monitor.** Click *Start monitoring* and allow the camera.

## Configuration

Set these as environment variables. None are required.

| Variable | Default | Meaning |
|---|---|---|
| `HOST` | `127.0.0.1` | Address to listen on. Use `0.0.0.0` in Docker. |
| `PORT` | `8000` | Port to listen on. |
| `MODEL_DIR` | next to `server.py` | Folder containing the three model files. |
| `DATA_DIR` | `./data` | Where enrolled face data is saved. |
| `PERSIST_FACES` | `1` | `0` keeps faces in memory only. |
| `COSINE_MATCH` | `0.363` | Face match threshold. Higher is stricter. |
| `PERSON_CONF` | `0.5` | Minimum YOLO confidence for a person. |

Example (PowerShell): `$env:COSINE_MATCH="0.4"; python server.py`

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check the server is up |
| `GET` | `/api/people` | List enrolled names |
| `POST` | `/api/enroll` | Enroll a face: `{ "name": "...", "image": "<base64 JPEG>" }` |
| `DELETE` | `/api/people/{name}` | Remove someone |
| `POST` | `/api/reset` | Forget tracked people (called when monitoring starts) |
| `POST` | `/api/analyze` | Analyze a frame: `{ "image": "<base64 JPEG>", "zone": [[x, y], ...], "anchor": "auto" }` (`anchor` is `auto`, `feet` or `center`) |

`/api/analyze` returns each person's `track_id`, `box`, `point` (the point tested against the zone), `inside`, `name` and match `score`, plus the server time `ms`. Zone points use the pixel coordinates of the image you send.

## Deploy

### Docker

```bash
docker build -t gatekeeper-ai .
docker run -p 7860:7860 gatekeeper-ai
```

### Hugging Face Space (Docker)

1. Create a new Space and choose **Docker** as the SDK.
2. Push this repo to the Space.
3. Add this block to the very top of the Space's `README.md`:

   ```yaml
   ---
   title: GateKeeper AI
   sdk: docker
   app_port: 7860
   ---
   ```

4. Open the Space's direct `*.hf.space` URL (not the embedded view) so the browser can ask for camera access.

Spaces have temporary storage, so enrolled faces are lost when the Space restarts.

## Project layout

```
server.py            FastAPI backend: tracking, face matching, API
index.html           Web frontend (no build step)
requirements.txt     Dependencies for the web app
Dockerfile           Container build (Hugging Face compatible)
ARCHITECTURE.md      Design notes
*.onnx, yolov8n.pt   Model weights
```

## Limitations

- **No liveness check.** A printed photo or a phone screen held up to the camera can be matched as a real face. Do not use this as the only control for real security.
- **Face accuracy depends on the camera.** Low light, small faces, masks and extreme angles reduce matches. The face is searched for in the top 40% of each person's box.
- **One camera, one user at a time.** Tracking state lives on the server, so two browsers analysing at once would confuse each other.
- **Zone check point.** *Auto* switches to the body centre when feet are cut off at the bottom of the frame. If someone is partly hidden in other ways (behind a desk, say), choose *Body centre* manually.

## Privacy

Enrolled faces are stored as 128-number embeddings in `data/enrolled.npz`, not as photos. Even so, they are biometric data. Only enroll people who have agreed, and delete the file (or use *Remove*) when you no longer need it. Set `PERSIST_FACES=0` to keep nothing on disk.

## Credits and licenses

- Original project: [vikassinngh123/gatekeeper-ai](https://github.com/vikassinngh123/gatekeeper-ai)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) is licensed under AGPL-3.0, which matters if you distribute or host a modified version. Check it against this repo's license.
- YuNet and SFace models come from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo).

See [LICENSE](LICENSE) for this repository's license.
