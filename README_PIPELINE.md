# Car Reconstruction Pipeline

End-to-end pipeline that reconstructs a 3D point cloud and triangle mesh of a
car from multi-camera video using Pi3X, then provides an interactive web viewer
for inspecting 3D detection projections.

---

## Table of Contents

1. [Conda Environment Setup](#1-conda-environment-setup)
2. [Pipeline Inputs](#2-pipeline-inputs)
3. [Running the Pipeline](#3-running-the-pipeline)
4. [Pipeline Outputs](#4-pipeline-outputs)
5. [Running the 3D Viewer](#5-running-the-3d-viewer)

---

## 1. Conda Environment Setup

The pipeline runs inside the `env_pi3` conda environment.

### Create the environment (first time only)

```bash
conda create -n env_pi3 python=3.10 -y
conda activate env_pi3

# Core ML / vision dependencies
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 pillow opencv-python plyfile safetensors huggingface_hub

# 3D processing (mesh reconstruction + ray casting)
pip install open3d trimesh rtree pyyaml

# Install the Pi3X package itself (from repo root)
pip install -e .
```

### Activate the environment

```bash
conda activate env_pi3
```

> **Note:** All `python3` commands below assume `env_pi3` is active.

---

## 2. Pipeline Inputs

The pipeline requires three inputs:

### 2a. Scan directory (`--scan_dir`)

A directory containing multi-camera frames and segmentation masks with this structure:

```
scan_dir/
├── extrinsics.yaml          # Camera rig extrinsics (relative poses between cameras)
├── at_cam_01/               # One folder per camera
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
├── at_cam_02/
│   └── ...
├── ...
└── masks/
    ├── at_cam_01/
    │   ├── frame_0040_mask.png   # Binary car segmentation mask per frame
    │   └── ...
    ├── at_cam_02/
    └── ...
```

- **Frames**: RGB images named `frame_XXXX.png`
- **Masks**: Binary PNG masks (white = car) named `frame_XXXX_mask.png`
- **`extrinsics.yaml`**: Camera rig calibration with intrinsics and relative transforms

### 2b. Trajectory file (`--trajectory`)

A `.npz` file from the UV-3D pipeline containing the camera poses for every frame:

```
trajectory.npz
├── pose_cache_cameras  – camera names per entry  (string array)
├── pose_cache_frames   – frame indices per entry  (int32 array)
└── pose_cache_values   – 4×4 world-from-cam matrices (float32, shape N×4×4)
```

Default path used:
```
/isilon/Automotive/RnD/elad.e/uv-3d/sessions/demo_room_test/
  run_2026-03-18T09-29-58-448462/02_trajectory/trajectory.npz
```

### 2c. Output directory (`--output_dir`)

Where all outputs are written. Created automatically if it does not exist.

Default: `output_pipeline/`

---

## 3. Running the Pipeline

### Minimal run (all defaults)

```bash
cd /isilon/Automotive/RnD/elad.e/pi3
conda activate env_pi3

python3 run_car_pipeline.py
```

This runs all 5 stages on frames 40–80 (every 2nd frame) using cameras
`at_cam_01`, `at_cam_02`, `at_cam_03`, `at_cam_07`, `at_cam_08`, `at_cam_09`.

---

### Full argument reference

```
python3 run_car_pipeline.py [OPTIONS]
```

#### Input / output

| Argument | Default | Description |
|---|---|---|
| `--scan_dir PATH` | `mpsfm_data/…` | Multi-camera scan directory |
| `--trajectory PATH` | `uv-3d/…/trajectory.npz` | Camera trajectory NPZ |
| `--output_dir PATH` | `output_pipeline/` | Where outputs are written |

#### Frame / camera selection

| Argument | Default | Description |
|---|---|---|
| `--cameras CAM [CAM …]` | 6 side cameras | Camera names to use |
| `--frame_start N` | `40` | First frame index |
| `--frame_end N` | `80` | Last frame index |
| `--frame_skip N` | `2` | Step between frames |

#### Stage flags

| Argument | Default | Description |
|---|---|---|
| `--skip_inference` | off | Skip Stage 1 — reuse existing per-frame PLYs |
| `--enable_reproj` | off | Enable Stage 3 reprojection filter (off by default) |
| `--reproj_ratio R` | `0.7` | Min car-mask hit ratio to keep a point (Stage 3) |
| `--skip_mesh` | off | Skip Stage 5 — do not build the Poisson mesh |

#### Bounding box padding (Stage 4)

| Argument | Default | Description |
|---|---|---|
| `--pad_front M` | `0.15` | Extra padding in front of car (metres) |
| `--pad_rear M` | `0.10` | Extra padding behind car |
| `--pad_top M` | `0.10` | Extra padding above roof |
| `--pad_bottom M` | `0.00` | Extra padding below floor |
| `--pad_left M` | `0.05` | Extra padding on left side |
| `--pad_right M` | `0.05` | Extra padding on right side |

#### Mesh reconstruction (Stage 5)

| Argument | Default | Description |
|---|---|---|
| `--mesh_depth N` | `9` | Poisson octree depth — higher = more detail |
| `--mesh_density_quantile Q` | `0.01` | Remove vertices below this density percentile |
| `--mesh_voxel_size M` | auto | Voxel size for downsampling before normal estimation. Auto-set to 0.02 m for clouds > 2M points. |

---

### Common usage examples

**Re-run only the mesh step** (point cloud already exists):

```bash
python3 run_car_pipeline.py --skip_inference
```

**Different frame range and cameras:**

```bash
python3 run_car_pipeline.py \
    --frame_start 30 --frame_end 90 --frame_skip 2 \
    --cameras at_cam_01 at_cam_07 at_cam_08
```

**Higher-quality mesh** (finer Poisson depth):

```bash
python3 run_car_pipeline.py --skip_inference --mesh_depth 10
```

**Skip mesh generation** (output only the point cloud):

```bash
python3 run_car_pipeline.py --skip_mesh
```

---

## 4. Pipeline Outputs

All files are written to `--output_dir` (default: `output_pipeline/`).

### Per-stage outputs

| Stage | File | Description |
|---|---|---|
| **1** | `frame_XXXX_caronly.ply` | Per-frame Pi3X point cloud (car only, local coords) |
| **2** | `stitched_raw.ply` | All frames stitched into world coordinates |
| **3** | `stitched_reproj_70.ply` | After reprojection filter (only if `--enable_reproj`) |
| **4** | `car_final.ply` | **Final point cloud** — cropped to car bounding box |
| **4** | `car_final_viz.ply` | Same as above with red bounding box edges drawn in |
| **5** | `car_final_mesh.ply` | **Triangle mesh** via Poisson reconstruction |

### Final outputs (what you care about)

```
output_pipeline/
├── car_final.ply          ← full-density point cloud (~12M pts, ~320 MB)
├── car_final_mesh.ply     ← triangle mesh (~440k vertices, ~34 MB)
└── car_final_viz.ply      ← point cloud + bbox wireframe for debugging
```

### Typical run statistics

| Metric | Value |
|---|---|
| Frames | 21 (40–80, skip 2) |
| Points after stitch | ~12.5 M |
| Points after bbox crop | ~12.3 M (98.6%) |
| Mesh vertices | ~440 k |
| Mesh triangles | ~876 k |
| Total runtime | ~4 min (GPU) |

---

## 5. Running the 3D Viewer

The viewer is an interactive web application that shows the reconstructed mesh
on the left and detection images on the right. Navigating between detection
images updates the camera frustum and projected bounding boxes live in the 3D
view (raycasting runs in the browser using a BVH).

### Prerequisites

The viewer needs extra Python packages in `env_pi3` (already installed if you
followed Section 1):

```bash
pip install pyyaml open3d
```

### Start the viewer

```bash
conda activate env_pi3

python3 /isilon/Automotive/RnD/elad.e/pi3/output_pipeline/viewer.py
```

By default the server listens on port **8080**.

```bash
# Custom port:
python3 output_pipeline/viewer.py --port 9090

# Force manifest rebuild:
python3 output_pipeline/viewer.py --rebuild
```

### Open in browser

- **Local machine:** http://127.0.0.1:8080
- **Remote (SSH / tunnel):** http://10.2.0.35:8080  *(replace with actual IP)*

> If `localhost` gives "Unable to connect", use the explicit IPv4 address
> (`127.0.0.1` or the machine's LAN IP) — browsers sometimes prefer IPv6.

### What the viewer requires

The viewer reads these files at startup from `output_pipeline/`:

| File | Purpose |
|---|---|
| `car_final_mesh.ply` | 3D mesh displayed and used for raycasting |
| `detections/` | Detection images (red-tinted bounding boxes) |

And these external files (UV-3D pipeline outputs):

| File | Purpose |
|---|---|
| `…/02_trajectory/trajectory.npz` | Camera poses per frame |
| `…/inputs/calibration.yaml` | Camera intrinsics |
| `…/04_clustering_pi3/damaged_detections_with_uuid.json` | Detection bboxes |

### Controls

| Action | Control |
|---|---|
| Rotate mesh | Left-click + drag |
| Zoom | Scroll wheel |
| Pan | Right-click + drag |
| Previous detection | ← arrow key or **←** button |
| Next detection | → arrow key or **→** button |

### How the projection works

When you navigate to a detection frame the browser:
1. Reads the camera pose (4×4 world-from-cam matrix) and intrinsics from the manifest
2. Constructs a pinhole ray through each point of a 5×5 grid inside the bounding box
3. Intersects those 25 rays with the loaded mesh using `three-mesh-bvh` (milliseconds)
4. Renders the hit points as yellow dots and the camera as a coloured wireframe frustum
