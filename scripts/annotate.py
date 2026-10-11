"""Local web tool for hand-labelling planogram areas in the annotations.json format of the brief.

Works on the output of scripts/extract_areas.py (run that first): for each timestamp it shows that
frame's labelled _overview.jpg with clickable areas, and the rectified crop of the selected area.

On start, every area of every image is filled with the default
{"planogram_compliant": true, "stock_level": "half", "stock_value": 0.5}
unless it already has a label, so existing work is never overwritten.

    .venv/bin/python scripts/annotate.py            # then open http://localhost:8000

Output:
    <out>                 annotations.json, exactly the submission/ground-truth format
    <out minus .json>.reviewed.json   {camera_id: [image_file, ...]} images marked as checked
"""
import argparse
import csv
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelfmon.io import write_json  # noqa: E402
from shelfmon.planogram import load_planogram  # noqa: E402

DEFAULT = {"planogram_compliant": True, "stock_level": "half", "stock_value": 0.5}
STOCK_VALUES = {"empty": 0, "half": 0.5, "full": 1}
HTML = Path(__file__).with_name("annotate.html")


def load_manifest(areas_dir: Path) -> dict:
    """{camera_id: {image_file: {"overview": path, "crops": {area_id: path}}}} with paths relative to areas_dir."""
    frames = {}
    with open(areas_dir / "manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            crop = row["crop_path"]
            frame = frames.setdefault(row["camera_id"], {}).setdefault(row["image_file"], {
                "overview": (Path(crop).parent / "_overview.jpg").as_posix(), "crops": {}})
            frame["crops"][row["area_id"]] = crop
    return frames


def init_annotations(frames: dict, planogram, path: Path) -> dict:
    ann = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for cam, areas in planogram.items():
        for img in sorted(frames.get(cam, {})):
            entry = ann.setdefault(cam, {}).setdefault(img, {})
            for area in areas:
                entry.setdefault(area.area_id, dict(DEFAULT))
    write_json(path, ann)
    return ann


def valid_label(label) -> bool:
    return (isinstance(label, dict)
            and isinstance(label.get("planogram_compliant"), bool)
            and label.get("stock_level") in STOCK_VALUES
            and label.get("stock_value") == STOCK_VALUES[label["stock_level"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--areas", type=Path, default=Path("data/areas"), help="output dir of extract_areas.py")
    ap.add_argument("--planogram", type=Path, default=Path("planogram.json"))
    ap.add_argument("--out", type=Path, default=Path("annotations/annotations.json"))
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if not (args.areas / "manifest.csv").exists():
        sys.exit(f"{args.areas / 'manifest.csv'} not found: run scripts/extract_areas.py first")
    planogram = load_planogram(args.planogram)
    frames = load_manifest(args.areas)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ann = init_annotations(frames, planogram, args.out)
    reviewed_path = args.out.with_name(args.out.stem + ".reviewed.json")
    reviewed = json.loads(reviewed_path.read_text()) if reviewed_path.exists() else {}

    areas_json = {cam: [{"area_id": a.area_id, "product_id": a.product_id, "product_name": a.product_name,
                         "quad": a.quad.tolist()} for a in areas] for cam, areas in planogram.items()}
    files_root = args.areas.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(self.path.split("?")[0])
            if path == "/":
                self.send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                body = {"areas": areas_json, "frames": frames, "annotations": ann, "reviewed": reviewed}
                self.send(200, json.dumps(body, ensure_ascii=False).encode())
            elif path.startswith("/files/"):
                f = (files_root / path[len("/files/"):]).resolve()
                if f.is_relative_to(files_root) and f.is_file():
                    self.send(200, f.read_bytes(), "image/jpeg")
                else:
                    self.send(404, b"{}")
            else:
                self.send(404, b"{}")

        def do_POST(self):
            if self.path != "/api/save":
                return self.send(404, b"{}")
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            cam, img, labels = req["camera_id"], req["image_file"], req["labels"]
            if img not in ann.get(cam, {}) or set(labels) != set(ann[cam][img]) \
                    or not all(valid_label(v) for v in labels.values()):
                return self.send(400, b'{"error": "invalid labels"}')
            ann[cam][img] = labels
            done = set(reviewed.get(cam, []))
            (done.add if req.get("reviewed") else done.discard)(img)
            reviewed[cam] = sorted(done)
            write_json(args.out, ann)
            write_json(reviewed_path, reviewed)
            self.send(200, b'{"ok": true}')

    n = sum(len(v) for imgs in ann.values() for v in imgs.values())
    print(f"{args.out}: {n} area labels. Open http://localhost:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
