#!/usr/bin/env python3
"""Convert a point cloud PLY to a triangle mesh using Poisson reconstruction.

Algorithm
---------
1. Load point cloud (Open3D).
2. Estimate normals if not present (KDTree radius=0.1m, max 30 neighbours).
3. Orient normals consistently (orient_normals_consistent_tangent_plane, k=30).
4. Screened Poisson Surface Reconstruction (depth=9).
5. Remove low-density vertices (below 1st percentile).
6. Save as binary PLY.

Usage
-----
conda run -n uv-3d python3 point_cloud_to_mesh.py \
    --input  output_pipeline/car_final_translated.ply \
    --output output_pipeline/car_final_translated_mesh.ply

Optional flags
--------------
  --depth D              Poisson octree depth (default 9)
  --density-quantile Q   Remove vertices below this quantile (default 0.01)
  --normal-radius R      KDTree radius for normal estimation (default 0.1)
  --normal-nn N          Max neighbours for normal estimation (default 30)
  --orient-k K           k for orient_normals_consistent_tangent_plane (default 30)
  --reestimate-normals   Force normal re-estimation even if normals are present
  --voxel-size V         Downsample to this voxel size before normal estimation (metres).
                         Auto-applied (0.005 m) when the cloud has >2M points.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Point cloud → triangle mesh via Poisson reconstruction."
    )
    parser.add_argument("--input",  required=True,
                        help="Input point cloud PLY")
    parser.add_argument("--output", required=True,
                        help="Output mesh PLY")
    parser.add_argument("--depth",  type=int, default=9,
                        help="Poisson octree depth (default 9)")
    parser.add_argument("--density-quantile", type=float, default=0.01,
                        help="Remove vertices below this density quantile (default 0.01)")
    parser.add_argument("--normal-radius", type=float, default=0.1,
                        help="KDTree radius for normal estimation (default 0.1)")
    parser.add_argument("--normal-nn", type=int, default=30,
                        help="Max neighbours for normal estimation (default 30)")
    parser.add_argument("--orient-k", type=int, default=30,
                        help="k for orient_normals_consistent_tangent_plane (default 30)")
    parser.add_argument("--reestimate-normals", action="store_true",
                        help="Force normal re-estimation even if normals are present")
    parser.add_argument("--voxel-size", type=float, default=None,
                        help="Downsample voxel size for normal estimation (metres). "
                             "Auto-set to 0.005 for clouds >2M points.")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"Loading point cloud: {input_path} …")
    t0 = time.time()
    pcd = o3d.io.read_point_cloud(str(input_path))
    print(f"  {len(pcd.points):,} points  ({time.time()-t0:.1f}s)")

    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty.")

    # ── 2. Normals ────────────────────────────────────────────────────────────
    need_normals = args.reestimate_normals or not pcd.has_normals()
    if need_normals:
        n_pts = len(pcd.points)
        voxel = args.voxel_size
        if voxel is None and n_pts > 2_000_000:
            voxel = 0.005
            print(f"Large cloud ({n_pts:,} pts) — downsampling to {voxel} m voxels for normal estimation …")
        if voxel is not None:
            pcd_for_normals = pcd.voxel_down_sample(voxel)
            print(f"  Downsampled to {len(pcd_for_normals.points):,} pts")
        else:
            pcd_for_normals = pcd

        print("Estimating normals …")
        t0 = time.time()
        pcd_for_normals.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.normal_radius, max_nn=args.normal_nn
            )
        )
        pcd_for_normals.orient_normals_consistent_tangent_plane(args.orient_k)
        print(f"  Done ({time.time()-t0:.1f}s)")
        pcd = pcd_for_normals
    else:
        print("Normals already present — skipping estimation.")

    # ── 3. Poisson reconstruction ─────────────────────────────────────────────
    print(f"Running Poisson surface reconstruction (depth={args.depth}) …")
    t0 = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth, width=0, scale=1.1, linear_fit=False
    )
    print(f"  Initial mesh: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles  ({time.time()-t0:.1f}s)")

    # ── 4. Density filtering ──────────────────────────────────────────────────
    print(f"Removing low-density vertices (quantile={args.density_quantile}) …")
    threshold = np.quantile(densities, args.density_quantile)
    mesh.remove_vertices_by_mask(np.asarray(densities) < threshold)
    print(f"  After filtering: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    print(f"Saving → {output_path} …")
    t0 = time.time()
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    print(f"  Saved  ({output_path.stat().st_size/1e6:.1f} MB, {time.time()-t0:.1f}s)")
    print("\nDone.")


if __name__ == "__main__":
    main()
