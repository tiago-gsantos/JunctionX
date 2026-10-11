"""Stock Sense: the staff app. Upload camera photos, run the model, see what to restock or fix first.

    .venv/bin/python app/server.py                 # then open http://localhost:8000
    .venv/bin/python app/server.py --load <dir>    # also process a photo folder at start-up
    .venv/bin/python app/server.py --demo          # demo model + its own results folder (scripts/build_demo.py)

Photos are uploaded as a folder laid out <camera_id>/<unix_timestamp>.jpg (or a .zip of one). The
newest upload replaces the previous one; it lives in data/runs/current/ (photos, crops, features,
predictions.json, details.json, plus state.json and corrections.json for what staff did). The UI is
static, in app/static/; everything it shows comes from GET /api/dashboard (see shelfmon/tasks.py).

API:
    POST /api/upload                      multipart files (filename = relative path) or a zip body
    GET  /api/status                      {state: idle|processing|ready|error, step, progress, message, has_run}
    GET  /api/dashboard                   summary, cameras (area polygons + status), sorted tasks, groups
    POST /api/tasks/<area_id>/done        mark done
    POST /api/tasks/<area_id>/undo        clear the done mark and any staff correction
    POST /api/tasks/<area_id>/feedback    {"stock_level"?: empty|half|full, "planogram_compliant"?: bool}
    GET  /api/predictions.json            the submission file (model output, without staff corrections)
    GET  /files/<path>                    photos of the current upload
"""
import argparse
import io
import json
import re
import shutil
import sys
import threading
import traceback
import zipfile
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from predict import run_pipeline  # noqa: E402

from shelfmon.io import write_json  # noqa: E402
from shelfmon.model import load_bundle  # noqa: E402
from shelfmon.planogram import load_planogram  # noqa: E402
from shelfmon.tasks import (apply_corrections, build_dashboard, correction_entry, latest_images,  # noqa: E402
                            load_display_names)

STATIC = Path(__file__).resolve().parent / "static"
MODEL = ROOT / "data/models/final.joblib"
REFERENCES = ROOT / "data/references/references.npz"
RUNS = ROOT / "data/runs"
DEMO = {"model": ROOT / "data/models/demo.joblib", "references": ROOT / "data/demo/references/references.npz",
        "runs": ROOT / "data/demo/runs"}
MAX_UPLOAD = 2 * 1024 ** 3
PHOTO_NAME = re.compile(r"^\d{9,11}\.jpe?g$", re.IGNORECASE)
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript",
                 ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
                 ".jpeg": "image/jpeg", ".woff2": "font/woff2", ".ico": "image/x-icon", ".webp": "image/webp"}


class App:
    """The current upload and what staff did with it. All access goes through the lock."""

    def __init__(self, model: Path = MODEL, references: Path = REFERENCES, runs: Path = RUNS):
        self.model, self.references = model, references
        self.current, self.incoming = runs / "current", runs / "incoming"
        self.lock = threading.Lock()
        self.planogram = load_planogram(ROOT / "planogram.json")
        self.names = load_display_names(ROOT / "analysis/product_display.csv")
        self.thresholds = load_bundle(model)["confidence"]
        self.status = {"state": "idle", "step": "", "progress": 0.0, "message": ""}
        self.run = None
        self.load_run()

    # --- current run -------------------------------------------------------------------------------
    def load_run(self):
        current = self.current
        if not (current / "predictions.json").exists():
            self.run = None
            return
        read = lambda name, default: json.loads((current / name).read_text(encoding="utf-8")) \
            if (current / name).exists() else default  # noqa: E731
        self.run = {"predictions": read("predictions.json", {}), "details": read("details.json", {}),
                    "corrections": read("corrections.json", {}), "done": set(read("state.json", {}).get("done", []))}
        self.status = {"state": "ready", "step": "", "progress": 1.0, "message": ""}

    def save_state(self):
        write_json(self.current / "state.json", {"done": sorted(self.run["done"])})
        write_json(self.current / "corrections.json", self.run["corrections"])

    def dashboard(self) -> dict:
        run = self.run
        return build_dashboard(
            self.planogram, run["predictions"], run["details"], self.thresholds, self.names,
            corrections=run["corrections"], done=run["done"],
            image_url=lambda cam, img: f"/files/images/{cam}/{quote(img)}")

    def task_image(self, area_id: str) -> tuple[str, str]:
        """(camera, newest image) of a task's area; raises KeyError for unknown areas."""
        cam = area_id.split("_", 1)[0]
        img = latest_images(self.run["predictions"])[cam]
        if area_id not in self.run["predictions"][cam][img]:
            raise KeyError(area_id)
        return cam, img

    # --- processing --------------------------------------------------------------------------------
    def start(self, photos: dict[str, bytes]):
        """photos: {"<camera_id>/<file>.jpg": bytes}, already validated. Runs the model in the background."""
        shutil.rmtree(self.incoming, ignore_errors=True)
        for rel, data in photos.items():
            path = self.incoming / "images" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.status = {"state": "processing", "step": "Starting", "progress": 0.0, "message": ""}
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        def progress(step, fraction):
            self.status |= {"step": step, "progress": round(fraction, 3)}
        try:
            run_pipeline(self.incoming / "images", self.incoming, self.model, self.references, progress=progress)
            with self.lock:
                shutil.rmtree(self.current, ignore_errors=True)
                self.incoming.rename(self.current)
                self.load_run()
        except Exception as e:  # shown to the user; the previous upload stays available
            traceback.print_exc()
            with self.lock:
                self.status = {"state": "error", "step": "", "progress": 0.0, "message": str(e)}


def photos_from_upload(content_type: str, body: bytes, cameras: set[str]) -> tuple[dict[str, bytes], list[str]]:
    """Accepted photos {"cam/file.jpg": bytes} and the names that were skipped."""
    files = []
    if content_type.startswith("multipart/form-data"):
        msg = BytesParser(policy=HTTP).parsebytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
        files = [(part.get_filename() or "", part.get_payload(decode=True) or b"") for part in msg.iter_parts()]
    else:
        files = [("upload.zip", body)]
    expanded = []
    for name, data in files:
        if name.lower().endswith(".zip") or data[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                expanded += [(m, z.read(m)) for m in z.namelist() if not m.endswith("/")]
        else:
            expanded.append((name, data))
    photos, skipped = {}, []
    for name, data in expanded:
        parts = PurePosixPath(name.replace("\\", "/")).parts
        if len(parts) >= 2 and parts[-2] in cameras and PHOTO_NAME.match(parts[-1]):
            photos[f"{parts[-2]}/{parts[-1].rsplit('.', 1)[0]}.jpg"] = data
        elif not parts or not parts[-1].startswith("."):
            skipped.append(name)
    return photos, skipped


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code: int, body: bytes, ctype: str = "application/json", headers: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def json(self, code: int, data):
            self.send(code, json.dumps(data, ensure_ascii=False).encode())

        def file(self, base: Path, rel: str):
            f = (base / rel).resolve()
            if f.is_relative_to(base.resolve()) and f.is_file():
                self.send(200, f.read_bytes(), CONTENT_TYPES.get(f.suffix.lower(), "application/octet-stream"))
            else:
                self.json(404, {"error": "not found"})

        def do_GET(self):
            path = unquote(self.path.split("?")[0])
            with app.lock:
                if path == "/api/status":
                    return self.json(200, app.status | {"has_run": app.run is not None})
                if path == "/api/dashboard":
                    return self.json(200, app.dashboard()) if app.run else self.json(404, {"error": "no photos yet"})
                if path == "/api/predictions.json":
                    if not app.run:
                        return self.json(404, {"error": "no photos yet"})
                    return self.send(200, json.dumps(app.run["predictions"], indent=1).encode(), headers={
                        "Content-Disposition": 'attachment; filename="predictions.json"'})
                if path.startswith("/files/"):
                    return self.file(app.current, path[len("/files/"):])
            if path.startswith("/api/"):
                return self.json(404, {"error": "not found"})
            self.file(STATIC, "index.html" if path == "/" else path.lstrip("/"))

        def do_POST(self):
            path = unquote(self.path.split("?")[0])
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_UPLOAD:
                return self.json(413, {"error": "upload too large"})
            body = self.rfile.read(length)

            if path == "/api/upload":
                with app.lock:
                    if app.status["state"] == "processing":
                        return self.json(409, {"error": "still processing the previous upload"})
                    photos, skipped = photos_from_upload(self.headers.get("Content-Type", ""), body,
                                                         set(app.planogram))
                    if not photos:
                        return self.json(400, {"error": "no camera photos found: expected "
                                                        "<camera_id>/<unix_timestamp>.jpg", "skipped": skipped[:20]})
                    app.start(photos)
                cams = {p.split("/")[0] for p in photos}
                return self.json(202, {"photos": len(photos), "cameras": len(cams), "skipped": skipped[:20]})

            m = re.fullmatch(r"/api/tasks/([0-9]+_[0-9]+)/(done|undo|feedback)", path)
            if not m:
                return self.json(404, {"error": "not found"})
            area_id, action = m.groups()
            with app.lock:
                if not app.run:
                    return self.json(404, {"error": "no photos yet"})
                try:
                    cam, img = app.task_image(area_id)
                except KeyError:
                    return self.json(404, {"error": f"unknown area {area_id}"})
                run = app.run
                if action == "done":
                    run["done"].add(area_id)
                elif action == "undo":
                    run["done"].discard(area_id)
                    run["corrections"].get(cam, {}).get(img, {}).pop(area_id, None)
                else:
                    try:
                        req = json.loads(body or b"{}")
                        current = apply_corrections(run["predictions"], run["corrections"])[cam][img][area_id]
                        entry = correction_entry(current, req.get("stock_level"), req.get("planogram_compliant"))
                    except (ValueError, AttributeError) as e:
                        return self.json(400, {"error": str(e)})
                    run["corrections"].setdefault(cam, {}).setdefault(img, {})[area_id] = entry
                app.save_state()
                return self.json(200, app.dashboard())

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open it from a tablet on the same network")
    ap.add_argument("--load", type=Path, help="process this photo folder at start-up")
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--references", type=Path, default=REFERENCES, help="saved 'usual shelf' references")
    ap.add_argument("--runs", type=Path, default=RUNS, help="where uploads and their results are kept")
    ap.add_argument("--demo", action="store_true", help="use the demo model, references and results folder "
                    "(build them with scripts/build_demo.py)")
    args = ap.parse_args()
    if args.demo:
        args.model, args.references, args.runs = DEMO["model"], DEMO["references"], DEMO["runs"]
    for path, how in [(args.model, "scripts/train_eval.py --model hgb --features handcrafted radio --override proto "
                                   "--save-model data/models/final.joblib --no-log"),
                      (args.references, "scripts/compute_features.py")]:
        if not path.exists():
            sys.exit(f"{path} not found: run {'scripts/build_demo.py' if args.demo else how}")

    app = App(args.model, args.references, args.runs)
    print(f"model: {args.model}\nresults: {args.runs}")
    if args.load:
        cameras = set(app.planogram)
        photos = {f"{p.parent.name}/{p.name}": p.read_bytes() for p in sorted(args.load.glob("*/*.jpg"))
                  if p.parent.name in cameras and PHOTO_NAME.match(p.name)}
        if not photos:
            sys.exit(f"no <camera_id>/<timestamp>.jpg photos under {args.load}")
        app.start(photos)
        print(f"processing {len(photos)} photos from {args.load} ...")
    print(f"Stock Sense on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer((args.host, args.port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()
