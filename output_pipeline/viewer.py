#!/usr/bin/env python3
"""Interactive 3D mesh viewer with synced detection gallery.

Split-panel:
  Left  – Three.js mesh rendered in browser (TrackballControls)
  Right – Detection image gallery; arrow navigation syncs camera frustum
          and ray-cast hit points in the 3D view.

Usage:
    python3 viewer.py [--port 8080]
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
MESH_PLY     = BASE / "car_final_mesh_v2.ply"   # displayed in browser
RAY_MESH_PLY = BASE / "car_final_mesh_v2.ply"   # used for ray casting
DETS_DIR     = BASE / "detections"
MANIFEST_PATH = BASE / ".manifest_cache.json"

TRAJ_NPZ  = Path("/isilon/Automotive/RnD/elad.e/uv-3d/sessions/demo_room_test"
                 "/run_2026-03-18T09-29-58-448462/02_trajectory/trajectory.npz")
CALIB_YAML = Path("/isilon/Automotive/RnD/elad.e/uv-3d/sessions/demo_room_test"
                  "/run_2026-03-18T09-29-58-448462/inputs/calibration.yaml")
DETS_JSON  = Path("/isilon/Automotive/RnD/elad.e/uv-3d/sessions/demo_room_test"
                  "/run_2026-03-18T09-29-58-448462/04_clustering_pi3"
                  "/damaged_detections_with_uuid.json")

FRUSTUM_DEPTH = 0.4   # metres
GRID_N        = 5     # rays per bbox edge
MAX_HIT_DIST  = 4.0   # max ray-hit distance (metres) to reject bleed-through
MAX_HIT_PTS   = 60    # cap per detection

# ── Camera colour palette (matches 13 camera names) ───────────────────────────
CAM_COLORS: dict[str, list[int]] = {
    "at_cam_01":  [230,  25,  75],
    "at_cam_02":  [ 60, 180,  75],
    "at_cam_03":  [255, 225,  25],
    "at_cam_04":  [  0, 130, 200],
    "at_cam_05":  [245, 130,  48],
    "at_cam_06":  [145,  30, 180],
    "at_cam_07":  [ 70, 240, 240],
    "at_cam_08":  [240,  50, 230],
    "at_cam_09":  [210, 245,  60],
    "at_front_00":[250, 190, 212],
    "at_front_01":[  0, 128, 128],
    "at_rear_00": [220, 190, 255],
    "at_rear_01": [170, 110,  40],
}
DEFAULT_COLOR = [128, 128, 128]


# ── Calibration loading ───────────────────────────────────────────────────────

def load_calibration(path: Path) -> dict[str, dict]:
    """Return dict cam_name → {fx,fy,cx,cy,dist,W,H}."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    cams: dict[str, dict] = {}
    for key, val in raw.items():
        if not isinstance(val, dict) or "cam_name" not in val:
            continue
        name = val["cam_name"]
        intr = val["intrinsics"]      # [fx, fy, cx, cy]
        dist = val.get("distortion_coeffs", [0, 0, 0, 0])
        res  = val["resolution"]      # [W, H]
        cams[name] = {
            "fx": intr[0], "fy": intr[1], "cx": intr[2], "cy": intr[3],
            "dist": np.array(dist[:4], dtype=np.float32),
            "W": res[0], "H": res[1],
        }
    return cams


# ── Trajectory loading ────────────────────────────────────────────────────────

def load_trajectory(path: Path) -> dict[tuple[str, int], np.ndarray]:
    """Return dict (cam_name, frame_idx) → 4×4 world-from-cam matrix."""
    d = np.load(str(path))
    cameras = d["pose_cache_cameras"]
    frames  = d["pose_cache_frames"]
    values  = d["pose_cache_values"]
    return {(cameras[i], int(frames[i])): values[i] for i in range(len(cameras))}


# ── Ray helpers ───────────────────────────────────────────────────────────────

def _bbox_rays(
    bbox: list[float],
    cam: dict,
    pose: np.ndarray,
    grid_n: int = GRID_N,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate rays (origins, directions) from a bbox grid in world space."""
    x1_n, y1_n, x2_n, y2_n = bbox
    W, H = cam["W"], cam["H"]
    x1, y1, x2, y2 = x1_n * W, y1_n * H, x2_n * W, y2_n * H

    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    K  = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dc = cam["dist"]

    # Grid of pixel points inside bbox
    xs = np.linspace(x1, x2, grid_n)
    ys = np.linspace(y1, y2, grid_n)
    pts = np.array([[x, y] for x in xs for y in ys], dtype=np.float32)

    # Undistort → normalised camera coords
    undist = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dc)  # (N,1,2)
    undist = undist.reshape(-1, 2)

    # Ray directions in camera space [x_n, y_n, 1]
    dirs_c = np.column_stack([undist, np.ones(len(undist))])
    dirs_c /= np.linalg.norm(dirs_c, axis=1, keepdims=True)

    # Transform to world
    R = pose[:3, :3]
    t = pose[:3, 3]
    dirs_w   = (R @ dirs_c.T).T
    dirs_w  /= np.linalg.norm(dirs_w, axis=1, keepdims=True)
    origins  = np.tile(t, (len(dirs_w), 1))
    return origins.astype(np.float32), dirs_w.astype(np.float32)


def _cast_bbox(
    bbox: list[float],
    cam: dict,
    pose: np.ndarray,
    mesh: trimesh.Trimesh,
) -> np.ndarray:
    """Ray-cast bbox against mesh; return 3-D hit points."""
    origins, dirs_w = _bbox_rays(bbox, cam, pose)
    try:
        locs, ray_idx, _ = mesh.ray.intersects_location(
            origins, dirs_w, multiple_hits=True
        )
    except Exception:
        return np.empty((0, 3))

    if len(locs) == 0:
        return np.empty((0, 3))

    # Keep nearest hit per ray within MAX_HIT_DIST
    best: dict[int, tuple[float, np.ndarray]] = {}
    for loc, ridx in zip(locs, ray_idx):
        dist = float(np.linalg.norm(loc - origins[ridx]))
        if dist > MAX_HIT_DIST:
            continue
        if ridx not in best or dist < best[ridx][0]:
            best[ridx] = (dist, loc)

    if not best:
        return np.empty((0, 3))
    return np.array([v for _, v in best.values()])


# ── Frustum geometry ──────────────────────────────────────────────────────────

def _make_frustum(
    cam: dict,
    pose: np.ndarray,
    depth: float = FRUSTUM_DEPTH,
) -> list[list[float]]:
    """Return [apex, tl, tr, br, bl] in world space (5 points)."""
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    W, H = cam["W"], cam["H"]

    corners_pix = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    K  = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dc = cam["dist"]
    undist = cv2.undistortPoints(corners_pix.reshape(-1, 1, 2), K, dc).reshape(-1, 2)

    dirs_c = np.column_stack([undist, np.ones(4)])
    dirs_c /= np.linalg.norm(dirs_c, axis=1, keepdims=True)
    dirs_w = (pose[:3, :3] @ dirs_c.T).T

    apex    = pose[:3, 3]
    corners = apex + dirs_w * depth
    pts = [apex] + list(corners)
    return [p.tolist() for p in pts]


# ── Detections loading ────────────────────────────────────────────────────────

def load_detections(path: Path) -> list[dict]:
    """Flatten nested detection JSON into a list with cam/frame/bbox/label."""
    with open(path) as f:
        raw = json.load(f)

    result = []
    for _outer_label, sides in raw.items():
        if not isinstance(sides, dict):
            continue
        for _side, dets in sides.items():
            if not isinstance(dets, list):
                continue
            for d in dets:
                cam = d.get("camera", "")
                frame_raw = d.get("frame_index", "")  # e.g. "frame_0041.png"
                # parse frame idx
                try:
                    frame_idx = int(frame_raw.split("_")[1].split(".")[0])
                except Exception:
                    continue
                bbox  = d.get("bbox", [])
                label = d.get("label", "")
                result.append({
                    "camera": cam,
                    "frame_idx": frame_idx,
                    "bbox": bbox,
                    "label": label,
                })
    return result


# ── Manifest precomputation ───────────────────────────────────────────────────

def build_manifest() -> dict:
    log.info("Loading mesh for ray casting: %s", RAY_MESH_PLY)
    ray_mesh = trimesh.load(str(RAY_MESH_PLY), process=False, force="mesh")
    log.info("  vertices=%d  faces=%d", len(ray_mesh.vertices), len(ray_mesh.faces))

    # Mesh centre (for centring in Three.js)
    mn = ray_mesh.vertices.min(axis=0)
    mx = ray_mesh.vertices.max(axis=0)
    center = ((mn + mx) / 2).tolist()
    log.info("  mesh centre: %s", center)

    calib = load_calibration(CALIB_YAML)
    traj  = load_trajectory(TRAJ_NPZ)
    dets  = load_detections(DETS_JSON)

    # Group detections by (camera, frame_idx)
    from collections import defaultdict
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for d in dets:
        grouped[(d["camera"], d["frame_idx"])].append(d)

    # Build detection image names from detections dir
    img_names: dict[tuple[str, int], str] = {}
    for img_path in sorted(DETS_DIR.iterdir()):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        name = img_path.stem  # e.g. "at_cam_07__frame_0041"
        # parse cam and frame
        parts = name.split("__")
        if len(parts) == 2:
            cam_name, frame_part = parts
            try:
                frame_idx = int(frame_part.split("_")[1])
            except Exception:
                continue
            img_names[(cam_name, frame_idx)] = img_path.name
        # also handle single underscore format e.g. "at_cam_01_frame_0032"
        elif len(parts) == 1:
            # try splitting on "_frame_"
            if "_frame_" in name:
                cam_part, frame_part = name.split("_frame_", 1)
                try:
                    frame_idx = int(frame_part.split(".")[0])
                    img_names[(cam_part, frame_idx)] = img_path.name
                except Exception:
                    pass

    log.info("Found %d detection images", len(img_names))

    c = np.array(center)
    frames_out: list[dict] = []

    total_keys = len(img_names)
    for idx, ((cam_name, frame_idx), img_file) in enumerate(sorted(img_names.items())):
        if idx % 20 == 0:
            log.info("  Processing frame %d/%d", idx, total_keys)

        cam = calib.get(cam_name)
        pose = traj.get((cam_name, frame_idx))

        if cam is None or pose is None:
            log.warning("  Missing cam/pose for %s frame %d", cam_name, frame_idx)
            frames_out.append({
                "image": img_file, "camera": cam_name, "frame_idx": frame_idx,
                "labels": [], "cam_color": DEFAULT_COLOR,
                "frustum": [], "hit_points": [],
            })
            continue

        cam_color = CAM_COLORS.get(cam_name, DEFAULT_COLOR)
        frustum = _make_frustum(cam, pose)
        # shift by mesh centre
        frustum_c = [(np.array(p) - c).tolist() for p in frustum]

        # Ray-cast all detections for this (cam, frame)
        all_hits: list[list[float]] = []
        labels: list[str] = []
        for det in grouped.get((cam_name, frame_idx), []):
            hits = _cast_bbox(det["bbox"], cam, pose, ray_mesh)
            if len(hits) > 0:
                hits_c = (hits - c)
                # cap
                if len(hits_c) > MAX_HIT_PTS:
                    step = max(1, len(hits_c) // MAX_HIT_PTS)
                    hits_c = hits_c[::step][:MAX_HIT_PTS]
                all_hits.extend(hits_c.tolist())
            labels.append(det["label"])

        frames_out.append({
            "image": img_file,
            "camera": cam_name,
            "frame_idx": frame_idx,
            "labels": labels,
            "cam_color": cam_color,
            "frustum": frustum_c,
            "hit_points": all_hits,
        })

    manifest = {"mesh_center": center, "frames": frames_out}
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f)
    log.info("Manifest saved → %s  (%d frames)", MANIFEST_PATH, len(frames_out))
    return manifest


def load_or_build_manifest() -> dict:
    if MANIFEST_PATH.exists():
        log.info("Loading cached manifest: %s", MANIFEST_PATH)
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return build_manifest()


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>UV-3D Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { display: flex; height: 100vh; overflow: hidden;
       background: #111; color: #eee; font-family: monospace; }
#left  { flex: 1; position: relative; }
#right { width: 42%; display: flex; flex-direction: column;
         border-left: 2px solid #333; }
#info  { padding: 8px 12px; font-size: 12px; background: #1a1a1a;
         border-bottom: 1px solid #333; line-height: 1.6; }
#gallery { flex: 1; display: flex; flex-direction: column;
           align-items: center; justify-content: center;
           padding: 10px; gap: 8px; overflow: hidden; }
#det-img { max-width: 100%; max-height: calc(100vh - 160px);
           object-fit: contain; border: 1px solid #444; }
#nav { display: flex; gap: 16px; align-items: center; }
.nav-btn { font-size: 22px; padding: 4px 14px; cursor: pointer;
           background: #333; border: 1px solid #555; color: #eee;
           border-radius: 4px; }
.nav-btn:hover { background: #555; }
#counter { font-size: 13px; color: #aaa; }
#caption { font-size: 11px; color: #aaa; text-align: center; max-width: 100%; }
canvas { display: block; }
</style>
</head>
<body>
<div id="left"></div>
<div id="right">
  <div id="info">
    <b>3D Viewer</b> – drag: rotate | scroll: zoom | right-drag: pan<br/>
    ← → arrows: navigate detections
  </div>
  <div id="gallery">
    <div id="nav">
      <button class="nav-btn" id="prev">&#8592;</button>
      <span id="counter">0 / 0</span>
      <button class="nav-btn" id="next">&#8594;</button>
    </div>
    <img id="det-img" src="" alt="no image"/>
    <div id="caption"></div>
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://esm.sh/three@0.163.0",
    "three/addons/": "https://esm.sh/three@0.163.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { PLYLoader }         from 'three/addons/loaders/PLYLoader.js';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';

// ── Scene setup ──────────────────────────────────────────────────────────────
const container = document.getElementById('left');
const renderer  = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirL = new THREE.DirectionalLight(0xffffff, 0.9);
dirL.position.set(5, 10, 7);
scene.add(dirL);

const camera = new THREE.PerspectiveCamera(
    45, container.clientWidth / container.clientHeight, 0.01, 500);
camera.position.set(0, 0, 6);

const controls = new TrackballControls(camera, renderer.domElement);
controls.rotateSpeed  = 4.0;
controls.zoomSpeed    = 1.5;
controls.panSpeed     = 0.8;
controls.noZoom       = false;
controls.noPan        = false;
controls.staticMoving = false;
controls.dynamicDampingFactor = 0.15;

window.addEventListener('resize', () => {
    const w = container.clientWidth, h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    controls.handleResize();
});

// ── Load mesh ────────────────────────────────────────────────────────────────
const loader = new PLYLoader();
loader.load('/mesh.ply', geo => {
    geo.computeVertexNormals();
    // Centre geometry (bounding box)
    geo.computeBoundingBox();
    const c = new THREE.Vector3();
    geo.boundingBox.getCenter(c);
    geo.translate(-c.x, -c.y, -c.z);

    let obj;
    if (geo.index === null && geo.getAttribute('position').count > 0 &&
        geo.attributes.position.count === geo.getAttribute('position').count &&
        !geo.groups.length) {
        // Point cloud (no faces)
        const mat = new THREE.PointsMaterial({ size: 0.005, vertexColors: true });
        obj = new THREE.Points(geo, mat);
    } else {
        const mat = new THREE.MeshStandardMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            roughness: 0.7, metalness: 0.1,
        });
        obj = new THREE.Mesh(geo, mat);
    }
    scene.add(obj);
}, xhr => {
    // progress
}, err => console.error('PLY load error', err));

// ── Manifest & gallery ───────────────────────────────────────────────────────
let manifest = null;
let frames   = [];
let curIdx   = 0;
let detGroup = null;

fetch('/manifest.json')
    .then(r => r.json())
    .then(m => {
        manifest = m;
        frames   = m.frames.filter(f => f.image);
        document.getElementById('counter').textContent = `1 / ${frames.length}`;
        if (frames.length > 0) showFrame(0);
    });

function showFrame(idx) {
    curIdx = idx;
    const f = frames[idx];

    // Gallery image
    document.getElementById('det-img').src = '/detections/' + f.image;
    document.getElementById('caption').textContent =
        `${f.image}  |  cam: ${f.camera}  |  frame: ${f.frame_idx}` +
        (f.labels.length ? `\n${f.labels.join(', ')}` : '');
    document.getElementById('counter').textContent =
        `${idx + 1} / ${frames.length}`;

    // 3-D overlays
    if (detGroup) scene.remove(detGroup);
    detGroup = new THREE.Group();

    const color = new THREE.Color(
        f.cam_color[0]/255, f.cam_color[1]/255, f.cam_color[2]/255);

    // Frustum
    if (f.frustum && f.frustum.length === 5) {
        const [apex, tl, tr, br, bl] = f.frustum.map(p => new THREE.Vector3(...p));
        const pts = [apex,tl, apex,tr, apex,br, apex,bl, tl,tr, tr,br, br,bl, bl,tl];
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
        detGroup.add(new THREE.LineSegments(geo, mat));
    }

    // Hit points
    if (f.hit_points && f.hit_points.length > 0) {
        const geo = new THREE.BufferGeometry();
        const pos = new Float32Array(f.hit_points.flat());
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        const mat = new THREE.PointsMaterial({ color: 0xffff00, size: 0.03 });
        detGroup.add(new THREE.Points(geo, mat));
    }

    scene.add(detGroup);
}

document.getElementById('prev').addEventListener('click', () => {
    if (frames.length === 0) return;
    showFrame((curIdx - 1 + frames.length) % frames.length);
});
document.getElementById('next').addEventListener('click', () => {
    if (frames.length === 0) return;
    showFrame((curIdx + 1) % frames.length);
});
document.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft')  document.getElementById('prev').click();
    if (e.key === 'ArrowRight') document.getElementById('next').click();
});

// ── Render loop ──────────────────────────────────────────────────────────────
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

_MANIFEST: dict | None = None


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence per-request logs

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            self._send(HTML.encode(), "text/html")
        elif path == "/mesh.ply":
            data = MESH_PLY.read_bytes()
            self._send(data, "application/octet-stream", cache=True)
        elif path == "/manifest.json":
            global _MANIFEST
            data = json.dumps(_MANIFEST).encode()
            self._send(data, "application/json")
        elif path.startswith("/detections/"):
            img_name = path[len("/detections/"):]
            img_path = DETS_DIR / img_name
            if img_path.exists():
                self._send(img_path.read_bytes(), "image/png", cache=True)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def _send(self, data: bytes, ct: str, cache: bool = False):
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=3600")
        else:
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rebuild", action="store_true",
                        help="Force manifest rebuild even if cache exists")
    args = parser.parse_args()

    if args.rebuild and MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    log.info("Building/loading manifest …")
    _MANIFEST = load_or_build_manifest()
    log.info("Manifest ready: %d frames", len(_MANIFEST["frames"]))

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    log.info("Serving on http://0.0.0.0:%d  (open http://127.0.0.1:%d in browser)",
             args.port, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
