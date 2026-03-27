#!/usr/bin/env python3
"""Interactive 3D mesh viewer with synced detection gallery.

Split-panel:
  Left  – Three.js mesh (TrackballControls). Live raycasting via three-mesh-bvh.
  Right – Detection image gallery with arrow navigation.
           Navigating a frame: camera frustum + bbox projections update instantly.

Manifest is built in ~1 second (no Python ray casting — all done live in JS).

Usage:
    python3 viewer.py [--port 8080]
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent
MESH_PLY      = BASE / "car_final_mesh.ply"
DETS_DIR      = BASE / "detections"
MANIFEST_PATH = BASE / ".manifest_cache.json"

TRAJ_NPZ   = Path("/isilon/Automotive/RnD/elad.e/uv-3d/sessions/demo_room_test"
                  "/run_2026-03-18T09-29-58-448462/02_trajectory/trajectory.npz")
CALIB_YAML = Path("/isilon/Automotive/RnD/elad.e/uv-3d/sessions/demo_room_test"
                  "/run_2026-03-18T09-29-58-448462/inputs/calibration.yaml")
DETS_JSON  = Path("/isilon/Automotive/RnD/elad.e/uv-3d/sessions/demo_room_test"
                  "/run_2026-03-18T09-29-58-448462/04_clustering_pi3"
                  "/damaged_detections_with_uuid.json")

FRUSTUM_DEPTH = 0.4   # metres (frustum wireframe depth)

# ── Camera colours ─────────────────────────────────────────────────────────────
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


# ── Data loading ───────────────────────────────────────────────────────────────

def load_calibration(path: Path) -> dict[str, dict]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cams: dict[str, dict] = {}
    for val in raw.values():
        if not isinstance(val, dict) or "cam_name" not in val:
            continue
        name = val["cam_name"]
        intr = val["intrinsics"]      # [fx, fy, cx, cy]
        dist = val.get("distortion_coeffs", [0, 0, 0, 0])
        res  = val["resolution"]      # [W, H]
        cams[name] = {
            "fx": intr[0], "fy": intr[1], "cx": intr[2], "cy": intr[3],
            "dist": dist[:4],
            "W": res[0], "H": res[1],
        }
    return cams


def load_trajectory(path: Path) -> dict[tuple[str, int], np.ndarray]:
    d = np.load(str(path))
    cameras = d["pose_cache_cameras"]
    frames  = d["pose_cache_frames"]
    values  = d["pose_cache_values"]
    return {(cameras[i], int(frames[i])): values[i] for i in range(len(cameras))}


def load_detections(path: Path) -> dict[tuple[str, int], list[dict]]:
    """Return dict (cam, frame_idx) → list of {bbox, label}."""
    with open(path) as f:
        raw = json.load(f)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for _outer, sides in raw.items():
        if not isinstance(sides, dict):
            continue
        for _side, dets in sides.items():
            if not isinstance(dets, list):
                continue
            for d in dets:
                cam = d.get("camera", "")
                frame_raw = d.get("frame_index", "")
                try:
                    frame_idx = int(frame_raw.split("_")[1].split(".")[0])
                except Exception:
                    continue
                grouped[(cam, frame_idx)].append({
                    "bbox":  d.get("bbox", []),
                    "label": d.get("label", ""),
                })
    return grouped


def parse_img_names(dets_dir: Path) -> dict[tuple[str, int], str]:
    """Map (cam_name, frame_idx) → image filename."""
    result: dict[tuple[str, int], str] = {}
    for p in sorted(dets_dir.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        name = p.stem
        if "__" in name:
            cam, frame_part = name.split("__", 1)
        elif "_frame_" in name:
            cam, frame_part = name.split("_frame_", 1)
            frame_part = "frame_" + frame_part
        else:
            continue
        try:
            frame_idx = int(frame_part.split("_")[1].split(".")[0])
        except Exception:
            continue
        result[(cam, frame_idx)] = p.name
    return result


def mesh_center(ply_path: Path) -> list[float]:
    """Compute bounding-box centre of the mesh (without loading all data)."""
    import struct
    # Fast path: parse PLY header to find vertex count, then read x,y,z
    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        header_str = header.decode("ascii", errors="ignore")
    # Extract n_vertices
    n_verts = 0
    for line in header_str.splitlines():
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
            break
    if n_verts == 0:
        return [0.0, 0.0, 0.0]
    # Use open3d for speed
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        verts = np.asarray(mesh.vertices)
        c = ((verts.min(0) + verts.max(0)) / 2).tolist()
        return c
    except Exception:
        return [0.0, 0.0, 0.0]


# ── Manifest ───────────────────────────────────────────────────────────────────

def build_manifest() -> dict:
    log.info("Loading calibration …")
    calib = load_calibration(CALIB_YAML)
    log.info("Loading trajectory …")
    traj = load_trajectory(TRAJ_NPZ)
    log.info("Loading detections …")
    grouped = load_detections(DETS_JSON)
    img_names = parse_img_names(DETS_DIR)

    log.info("Computing mesh centre …")
    center = mesh_center(MESH_PLY)
    log.info("  centre: %s", [round(x, 4) for x in center])
    cx, cy, cz = center

    frames_out: list[dict] = []
    for (cam_name, frame_idx), img_file in sorted(img_names.items()):
        cam  = calib.get(cam_name)
        pose = traj.get((cam_name, frame_idx))
        dets = grouped.get((cam_name, frame_idx), [])

        cam_color = CAM_COLORS.get(cam_name, DEFAULT_COLOR)

        if cam is None or pose is None:
            frames_out.append({
                "image": img_file, "camera": cam_name, "frame_idx": frame_idx,
                "labels": [d["label"] for d in dets],
                "cam_color": cam_color,
                "pose": None, "intrinsics": None, "bboxes": [],
            })
            continue

        # Store pose as nested list (4×4, row-major)
        pose_list = pose.tolist()

        frames_out.append({
            "image":      img_file,
            "camera":     cam_name,
            "frame_idx":  frame_idx,
            "labels":     [d["label"] for d in dets],
            "cam_color":  cam_color,
            "pose":       pose_list,            # world-from-cam 4×4
            "intrinsics": {                     # pinhole params
                "fx": cam["fx"], "fy": cam["fy"],
                "cx": cam["cx"], "cy": cam["cy"],
                "W":  cam["W"],  "H":  cam["H"],
            },
            "bboxes": [d["bbox"] for d in dets],  # normalised [x1,y1,x2,y2]
        })

    manifest = {
        "mesh_center":    center,
        "frustum_depth":  FRUSTUM_DEPTH,
        "frames":         frames_out,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f)
    log.info("Manifest saved → %s  (%d frames)", MANIFEST_PATH, len(frames_out))
    return manifest


def load_or_build_manifest() -> dict:
    if MANIFEST_PATH.exists():
        log.info("Loading cached manifest …")
        with open(MANIFEST_PATH) as f:
            m = json.load(f)
        # Rebuild if old format (had precomputed hit_points)
        if m.get("frames") and "hit_points" in (m["frames"][0] if m["frames"] else {}):
            log.info("Old manifest format detected — rebuilding …")
            MANIFEST_PATH.unlink()
            return build_manifest()
        return m
    return build_manifest()


# ── HTML ───────────────────────────────────────────────────────────────────────

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
#status  { position: absolute; top: 8px; left: 8px; font-size: 11px;
           color: #aaa; background: rgba(0,0,0,0.5); padding: 3px 8px;
           border-radius: 3px; pointer-events: none; }
canvas { display: block; }
</style>
</head>
<body>
<div id="left">
  <div id="status">Loading mesh …</div>
</div>
<div id="right">
  <div id="info">
    <b>UV-3D Viewer</b> – drag: rotate &nbsp;|&nbsp; scroll: zoom &nbsp;|&nbsp;
    right-drag: pan<br/>← → arrows or buttons: navigate detections
  </div>
  <div id="gallery">
    <div id="nav">
      <button class="nav-btn" id="prev">&#8592;</button>
      <span id="counter">– / –</span>
      <button class="nav-btn" id="next">&#8594;</button>
    </div>
    <img id="det-img" src="" alt=""/>
    <div id="caption"></div>
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "three":           "https://esm.sh/three@0.163.0",
    "three/addons/":   "https://esm.sh/three@0.163.0/examples/jsm/",
    "three-mesh-bvh":  "https://esm.sh/three-mesh-bvh@0.7.3"
  }
}
</script>
<script type="module">
import * as THREE          from 'three';
import { PLYLoader }       from 'three/addons/loaders/PLYLoader.js';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';
import {
  computeBoundsTree, disposeBoundsTree, acceleratedRaycast
} from 'three-mesh-bvh';

// Patch Three.js with BVH for fast raycasting
THREE.BufferGeometry.prototype.computeBoundsTree  = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree  = disposeBoundsTree;
THREE.Mesh.prototype.raycast                      = acceleratedRaycast;

const GRID_N       = 5;     // rays per edge of bbox (5×5 = 25 rays)
const MAX_RAY_DIST = 8.0;   // metres – ignore hits farther than this
const FRUSTUM_DEPTH = 0.4;  // metres

// ── Scene ─────────────────────────────────────────────────────────────────────
const container = document.getElementById('left');
const renderer  = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
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
controls.staticMoving = false;
controls.dynamicDampingFactor = 0.15;

window.addEventListener('resize', () => {
    const w = container.clientWidth, h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    controls.handleResize();
});

// ── Mesh ──────────────────────────────────────────────────────────────────────
let meshObj = null;
let meshCenter = new THREE.Vector3();

const loader = new PLYLoader();
loader.load('/mesh.ply', geo => {
    geo.computeVertexNormals();
    geo.computeBoundingBox();
    const c = new THREE.Vector3();
    geo.boundingBox.getCenter(c);
    geo.translate(-c.x, -c.y, -c.z);
    meshCenter.set(c.x, c.y, c.z);

    // Build BVH for fast raycasting
    geo.computeBoundsTree();

    const hasFaces = geo.index !== null ||
        (geo.groups && geo.groups.length > 0) ||
        (geo.attributes.position && geo.attributes.position.count > 0 &&
         geo.attributes.position.count % 3 === 0 &&
         !geo.isPoints);

    const mat = new THREE.MeshStandardMaterial({
        vertexColors: geo.hasAttribute('color'),
        color: geo.hasAttribute('color') ? 0xffffff : 0x888888,
        side: THREE.DoubleSide,
        roughness: 0.7, metalness: 0.1,
    });
    meshObj = new THREE.Mesh(geo, mat);
    scene.add(meshObj);
    document.getElementById('status').textContent = 'BVH ready – raycasting live';
    setTimeout(() => document.getElementById('status').style.display = 'none', 2000);

    // If manifest already loaded, render the current frame
    if (manifest && frames.length > 0) showFrame(curIdx);
}, () => {}, err => console.error('PLY error', err));

// ── Manifest + gallery ────────────────────────────────────────────────────────
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

// ── Live raycasting helpers ───────────────────────────────────────────────────

function worldRayDir(px, py, intr, poseRows) {
    // Normalised camera coords (pinhole, no distortion)
    const xn = (px - intr.cx) / intr.fx;
    const yn = (py - intr.cy) / intr.fy;
    const len = Math.sqrt(xn*xn + yn*yn + 1);
    // Rotate from camera to world space using pose rotation (rows 0-2, cols 0-2)
    const dx = (poseRows[0][0]*xn + poseRows[0][1]*yn + poseRows[0][2]) / len;
    const dy = (poseRows[1][0]*xn + poseRows[1][1]*yn + poseRows[1][2]) / len;
    const dz = (poseRows[2][0]*xn + poseRows[2][1]*yn + poseRows[2][2]) / len;
    return new THREE.Vector3(dx, dy, dz);
}

function castBboxRays(bbox, intr, poseRows, camPosWorld) {
    if (!meshObj) return [];
    const [x1n, y1n, x2n, y2n] = bbox;
    const { W, H } = intr;
    const x1 = x1n * W, y1 = y1n * H, x2 = x2n * W, y2 = y2n * H;

    const raycaster = new THREE.Raycaster();
    raycaster.far   = MAX_RAY_DIST;
    raycaster.firstHitOnly = true;  // three-mesh-bvh: return nearest hit only

    const hits = [];
    for (let i = 0; i <= GRID_N; i++) {
        for (let j = 0; j <= GRID_N; j++) {
            const px = x1 + (x2 - x1) * i / GRID_N;
            const py = y1 + (y2 - y1) * j / GRID_N;
            const dir = worldRayDir(px, py, intr, poseRows);
            raycaster.set(camPosWorld, dir);
            const isects = raycaster.intersectObject(meshObj, false);
            if (isects.length > 0) hits.push(isects[0].point.clone());
        }
    }
    return hits;
}

function makeFrustum(intr, poseRows, camPosWorld, depth, color) {
    const { fx, fy, cx, cy, W, H } = intr;
    // 4 image corners
    const corners = [[0,0],[W,0],[W,H],[0,H]].map(([px,py]) => {
        const dir = worldRayDir(px, py, intr, poseRows);
        return camPosWorld.clone().add(dir.multiplyScalar(depth));
    });
    const [tl, tr, br, bl] = corners;
    const apex = camPosWorld;
    const pts = [
        apex,tl,  apex,tr,  apex,br,  apex,bl,
        tl,tr,    tr,br,    br,bl,    bl,tl,
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color }));
}

// ── Show frame ────────────────────────────────────────────────────────────────

function showFrame(idx) {
    curIdx = idx;
    const f = frames[idx];

    document.getElementById('det-img').src = '/detections/' + f.image;
    document.getElementById('caption').textContent =
        `${f.image}  |  cam: ${f.camera}  frame: ${f.frame_idx}` +
        (f.labels.length ? `\n${f.labels.join(', ')}` : '');
    document.getElementById('counter').textContent = `${idx+1} / ${frames.length}`;

    // Remove previous overlays
    if (detGroup) { scene.remove(detGroup); detGroup.clear(); }
    detGroup = new THREE.Group();

    if (!f.pose || !f.intrinsics) { scene.add(detGroup); return; }

    const pose  = f.pose;   // 4×4 list-of-lists, row-major
    const intr  = f.intrinsics;
    const color = new THREE.Color(
        f.cam_color[0]/255, f.cam_color[1]/255, f.cam_color[2]/255);
    const depth = manifest.frustum_depth;

    // Camera position in centred space (subtract mesh centre)
    const camPos = new THREE.Vector3(
        pose[0][3] - meshCenter.x,
        pose[1][3] - meshCenter.y,
        pose[2][3] - meshCenter.z,
    );

    // Frustum
    detGroup.add(makeFrustum(intr, pose, camPos, depth, color));

    // Live ray-cast each bbox
    const allHits = [];
    for (const bbox of f.bboxes) {
        const hits = castBboxRays(bbox, intr, pose, camPos);
        allHits.push(...hits);
    }

    if (allHits.length > 0) {
        const pos = new Float32Array(allHits.flatMap(p => [p.x, p.y, p.z]));
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        detGroup.add(new THREE.Points(geo,
            new THREE.PointsMaterial({ color: 0xffff00, size: 0.025 })));
    }

    scene.add(detGroup);
}

document.getElementById('prev').onclick = () =>
    frames.length && showFrame((curIdx - 1 + frames.length) % frames.length);
document.getElementById('next').onclick = () =>
    frames.length && showFrame((curIdx + 1) % frames.length);
document.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft')  document.getElementById('prev').click();
    if (e.key === 'ArrowRight') document.getElementById('next').click();
});

// ── Render loop ───────────────────────────────────────────────────────────────
(function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
})();
</script>
</body>
</html>
"""

# ── HTTP handler ───────────────────────────────────────────────────────────────

_MANIFEST: dict | None = None


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(HTML.encode(), "text/html")
        elif path == "/mesh.ply":
            self._send(MESH_PLY.read_bytes(), "application/octet-stream", cache=True)
        elif path == "/manifest.json":
            self._send(json.dumps(_MANIFEST).encode(), "application/json")
        elif path.startswith("/detections/"):
            img = DETS_DIR / path[len("/detections/"):]
            if img.exists():
                self._send(img.read_bytes(), "image/png", cache=True)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def _send(self, data: bytes, ct: str, cache: bool = False):
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-cache")
        self.end_headers()
        self.wfile.write(data)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global _MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    type=int, default=8080)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.rebuild and MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    _MANIFEST = load_or_build_manifest()
    log.info("Manifest ready: %d frames", len(_MANIFEST["frames"]))

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    log.info("Serving on http://0.0.0.0:%d  →  open http://127.0.0.1:%d",
             args.port, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
