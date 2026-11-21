"""Variant G POC: re-render the slime_chibi_demo slime with a REAL 3D engine.

Instead of sprite stacking, this extracts a smooth polygon mesh from the same
analytic SDF the demo sculpts (surface nets -> Taubin smoothing -> smooth
normals), transfers the demo's own per-voxel colors (cel shading + face paint)
onto the vertices by alpha-weighted trilinear sampling of the RGBA volume that
sculpt_spirit_frame / sculpt_star_companion produce, and embeds the meshes
plus a pinned three.js (r128 UMD) into ONE self-contained HTML file with zero
runtime external requests.

Output: variants/slime-G-threejs.html
"""
import base64
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from build_demo import sculpt_spirit_frame, sculpt_star_companion, evaluate_demo_pose
from toolkit.gpu_sdf import make_grid, smin, sd_sphere, sd_ellipsoid, sd_capsule

THREE_JS_PATH = os.path.join(REPO_ROOT, "build_threejs", "three.min.js")
OUT_HTML = os.path.join(REPO_ROOT, "variants", "slime-G-threejs.html")
SIZE_BUDGET = 12 * 1024 * 1024

GRID = 128          # field resolution (128^3, coords x0.5 -> same 64-unit world)
VOX = 0.5           # world units per voxel at GRID=128
CENTER_Z = 24.0     # demo-space z subtracted so the slime sits near origin


# ---------------------------------------------------------------- SDF fields
def spirit_sdf(X, Y, Z, pose):
    """Exact copy of the SDF composition inside build_demo.sculpt_spirit_frame,
    plus the demo's occupancy floor cut (Z >= 0.45) expressed as a half-space
    intersection so the sign field matches the baked occupancy."""
    sx, sy, sz = pose.sx, pose.stretch_y, pose.squash_z
    rb = pose.root_bob
    PX = X / sx
    PY = Y / sy
    PZ = (Z - rb) / sz
    d_body = sd_ellipsoid(PX, PY, PZ, 0.0, 0.0, 22.0, 16.5, 15.5, 18.0)
    d_ear = sd_capsule(np.abs(PX), PY, PZ, 12.0, -2.0, 32.0, 22.5, -5.0, 37.0 + pose.ear_flap_l, 4.2, 2.2)
    d_ahoge = sd_capsule(PX, PY, PZ, 0.0, 0.0, 39.0, -6.0, 4.0, 46.0 + pose.ear_flap_l * 0.4, 2.2, 1.0)
    d_all = smin(smin(d_body, d_ear, 2.2), d_ahoge, 1.4)
    return np.maximum(d_all, 0.45 - Z)


def star_sdf(X, Y, Z, f, total=24):
    """Exact copy of the SDF composition inside build_demo.sculpt_star_companion."""
    ph = (f / float(total)) * math.tau
    cz = 16.0 + 1.6 * math.sin(ph)
    d0 = sd_sphere(X, Y, Z, 0.0, 0.0, cz, 7.2)
    d1 = sd_ellipsoid(X, Y, Z, 0.0, 0.0, cz, 11.2, 4.2, 4.2)
    d2 = sd_ellipsoid(X, Y, Z, 0.0, 0.0, cz, 4.2, 11.2, 4.2)
    d3 = sd_ellipsoid(X, Y, Z, 0.0, 0.0, cz, 4.2, 4.2, 11.2)
    return smin(smin(d0, d1, 1.4), smin(d2, d3, 1.4), 1.4)


# ------------------------------------------------------------ surface nets
def surface_nets(d):
    """Naive surface nets on field d indexed [z, y, x] (inside: d <= 0).

    Returns (verts_idx, tris) where verts_idx are fractional grid-point-index
    coordinates in (x, y, z) order, tris are indices into verts_idx.
    Fully vectorized: edge crossings are accumulated into adjacent cells with
    np.add.at; quads are gathered per sign-changing edge.
    """
    inside = d <= 0
    nz, ny, nx = d.shape
    cz_, cy_, cx_ = nz - 1, ny - 1, nx - 1

    c = [inside[:-1, :-1, :-1], inside[:-1, :-1, 1:], inside[:-1, 1:, :-1], inside[:-1, 1:, 1:],
         inside[1:, :-1, :-1], inside[1:, :-1, 1:], inside[1:, 1:, :-1], inside[1:, 1:, 1:]]
    any_in = c[0] | c[1] | c[2] | c[3] | c[4] | c[5] | c[6] | c[7]
    all_in = c[0] & c[1] & c[2] & c[3] & c[4] & c[5] & c[6] & c[7]
    cell_mask = any_in & ~all_in
    n_cells = int(cell_mask.sum())
    cell_id = -np.ones((cz_, cy_, cx_), dtype=np.int32)
    cell_id[cell_mask] = np.arange(n_cells, dtype=np.int32)

    cell_size = cz_ * cy_ * cx_
    pos_sum = np.zeros((cell_size, 3), dtype=np.float64)  # (z, y, x) index coords
    pos_cnt = np.zeros(cell_size, dtype=np.float64)

    def accumulate(edge_mask, d0, d1, start, axis):
        """Add crossing points of sign-changing edges to their <=4 adjacent cells.
        start: (z, y, x) int arrays of edge start points; axis: 0=z,1=y,2=x."""
        if not np.any(edge_mask):
            return
        zs, ys, xs = (a[edge_mask] for a in start)
        v0, v1 = d0[edge_mask], d1[edge_mask]
        t = v0 / (v0 - v1)
        p = np.stack([zs, ys, xs], axis=1).astype(np.float64)
        p[:, axis] += t
        # adjacent cells: this cell coord along `axis` is the edge start; the
        # other two axes take offsets {0, -1}.
        others = [a for a in (0, 1, 2) if a != axis]
        for o1 in (0, -1):
            for o2 in (0, -1):
                cell = p.astype(np.int64).copy()
                cell[:, others[0]] += o1
                cell[:, others[1]] += o2
                valid = (
                    (cell[:, 0] >= 0) & (cell[:, 0] < cz_)
                    & (cell[:, 1] >= 0) & (cell[:, 1] < cy_)
                    & (cell[:, 2] >= 0) & (cell[:, 2] < cx_)
                )
                if not np.any(valid):
                    continue
                flat = ((cell[valid, 0] * cy_ + cell[valid, 1]) * cx_ + cell[valid, 2])
                np.add.at(pos_sum, flat, p[valid])
                np.add.at(pos_cnt, flat, 1.0)

    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    # x-edges (axis=2): between (z,y,x) and (z,y,x+1)
    accumulate(inside[:, :, :-1] != inside[:, :, 1:], d[:, :, :-1], d[:, :, 1:],
               (zz[:, :, :-1], yy[:, :, :-1], xx[:, :, :-1]), axis=2)
    # y-edges (axis=1)
    accumulate(inside[:, :-1, :] != inside[:, 1:, :], d[:, :-1, :], d[:, 1:, :],
               (zz[:, :-1, :], yy[:, :-1, :], xx[:, :-1, :]), axis=1)
    # z-edges (axis=0)
    accumulate(inside[:-1, :, :] != inside[1:, :, :], d[:-1, :, :], d[1:, :, :],
               (zz[:-1, :, :], yy[:-1, :, :], xx[:-1, :, :]), axis=0)

    cnt = pos_cnt[cell_mask.ravel()]
    assert np.all(cnt > 0), "surface cell without edge crossing"
    verts_zyx = pos_sum[cell_mask.ravel()] / cnt[:, None]
    verts_idx = verts_zyx[:, ::-1].copy()  # -> (x, y, z) index coords

    # Quads around each sign-changing edge.
    quads = []

    def gather_quads(edge_mask, cell_of):
        if not np.any(edge_mask):
            return
        zs, ys, xs = np.nonzero(edge_mask)
        ids = np.stack([cell_of(zs, ys, xs, dz, dy, dx) for (dz, dy, dx) in QUAD_OFFSETS], axis=1)
        valid = np.all(ids >= 0, axis=1)
        if np.any(valid):
            quads.append(ids[valid])

    # For an edge, the 4 adjacent cells in cyclic order. cell_of receives the
    # edge start arrays plus per-corner offsets in (z, y, x).
    return verts_idx, cell_id, quads, gather_quads


QUAD_OFFSETS = []  # filled per-direction below (cyclic order around the edge)


def build_faces(d, cell_id):
    inside = d <= 0
    nz, ny, nx = d.shape
    quads = []

    def cells_at(zs, ys, xs, offs):
        out = []
        for (dz, dy, dx) in offs:
            z2, y2, x2 = zs + dz, ys + dy, xs + dx
            ok = (z2 >= 0) & (z2 < cell_id.shape[0]) & (y2 >= 0) & (y2 < cell_id.shape[1]) & (x2 >= 0) & (x2 < cell_id.shape[2])
            ids = np.full(zs.shape, -1, dtype=np.int32)
            if np.any(ok):
                ids[ok] = cell_id[z2[ok], y2[ok], x2[ok]]
            out.append(ids)
        return np.stack(out, axis=1)

    # x-edges: cells (cz in {z-1,z}, cy in {y-1,y}, cx = x), cyclic in (cy, cz)
    zs, ys, xs = np.nonzero(inside[:, :, :-1] != inside[:, :, 1:])
    if len(zs):
        q = cells_at(zs, ys, xs, [(-1, -1, 0), (-1, 0, 0), (0, 0, 0), (0, -1, 0)])
        quads.append(q[np.all(q >= 0, axis=1)])
    # y-edges: cells (cz in {z-1,z}, cy = y, cx in {x-1,x}), cyclic in (cz, cx)
    zs, ys, xs = np.nonzero(inside[:, :-1, :] != inside[:, 1:, :])
    if len(zs):
        q = cells_at(zs, ys, xs, [(-1, 0, -1), (-1, 0, 0), (0, 0, 0), (0, 0, -1)])
        quads.append(q[np.all(q >= 0, axis=1)])
    # z-edges: cells (cz = z, cy in {y-1,y}, cx in {x-1,x}), cyclic in (cx, cy)
    zs, ys, xs = np.nonzero(inside[:-1, :, :] != inside[1:, :, :])
    if len(zs):
        q = cells_at(zs, ys, xs, [(0, -1, -1), (0, -1, 0), (0, 0, 0), (0, 0, -1)])
        quads.append(q[np.all(q >= 0, axis=1)])

    quads = np.concatenate(quads, axis=0)
    tris = np.empty((len(quads) * 2, 3), dtype=np.int32)
    tris[0::2] = quads[:, [0, 1, 2]]
    tris[1::2] = quads[:, [0, 2, 3]]
    return tris


def extract_mesh(d):
    """Field d [z,y,x] -> (verts_idx (x,y,z) grid coords, tris)."""
    # Reuse surface_nets for vertex placement only.
    inside = d <= 0
    nz, ny, nx = d.shape
    cz_, cy_, cx_ = nz - 1, ny - 1, nx - 1
    c = [inside[:-1, :-1, :-1], inside[:-1, :-1, 1:], inside[:-1, 1:, :-1], inside[:-1, 1:, 1:],
         inside[1:, :-1, :-1], inside[1:, :-1, 1:], inside[1:, 1:, :-1], inside[1:, 1:, 1:]]
    any_in = c[0] | c[1] | c[2] | c[3] | c[4] | c[5] | c[6] | c[7]
    all_in = c[0] & c[1] & c[2] & c[3] & c[4] & c[5] & c[6] & c[7]
    cell_mask = any_in & ~all_in
    n_cells = int(cell_mask.sum())
    cell_id = -np.ones((cz_, cy_, cx_), dtype=np.int32)
    cell_id[cell_mask] = np.arange(n_cells, dtype=np.int32)

    cell_size = cz_ * cy_ * cx_
    pos_sum = np.zeros((cell_size, 3), dtype=np.float64)
    pos_cnt = np.zeros(cell_size, dtype=np.float64)
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")

    def accumulate(edge_mask, d0, d1, start, axis):
        if not np.any(edge_mask):
            return
        zs, ys, xs = (a[edge_mask] for a in start)
        v0, v1 = d0[edge_mask], d1[edge_mask]
        t = (v0 / (v0 - v1)).astype(np.float64)
        p = np.stack([zs, ys, xs], axis=1).astype(np.float64)
        p[:, axis] += t
        others = [a for a in (0, 1, 2) if a != axis]
        base = p.astype(np.int64)
        for o1 in (0, -1):
            for o2 in (0, -1):
                cell = base.copy()
                cell[:, others[0]] += o1
                cell[:, others[1]] += o2
                valid = ((cell[:, 0] >= 0) & (cell[:, 0] < cz_)
                         & (cell[:, 1] >= 0) & (cell[:, 1] < cy_)
                         & (cell[:, 2] >= 0) & (cell[:, 2] < cx_))
                if not np.any(valid):
                    continue
                flat = (cell[valid, 0] * cy_ + cell[valid, 1]) * cx_ + cell[valid, 2]
                np.add.at(pos_sum, flat, p[valid])
                np.add.at(pos_cnt, flat, 1.0)

    accumulate(inside[:, :, :-1] != inside[:, :, 1:], d[:, :, :-1], d[:, :, 1:],
               (zz[:, :, :-1], yy[:, :, :-1], xx[:, :, :-1]), axis=2)
    accumulate(inside[:, :-1, :] != inside[:, 1:, :], d[:, :-1, :], d[:, 1:, :],
               (zz[:, :-1, :], yy[:, :-1, :], xx[:, :-1, :]), axis=1)
    accumulate(inside[:-1, :, :] != inside[1:, :, :], d[:-1, :, :], d[1:, :, :],
               (zz[:-1, :, :], yy[:-1, :, :], xx[:-1, :, :]), axis=0)

    cnt = pos_cnt[cell_mask.ravel()]
    assert np.all(cnt > 0), "surface cell without edge crossing"
    verts_zyx = pos_sum[cell_mask.ravel()] / cnt[:, None]
    verts_idx = verts_zyx[:, ::-1].copy()
    tris = build_faces(d, cell_id)
    return verts_idx, tris


# ------------------------------------------------------------- mesh cleanup
def taubin_smooth(verts, tris, iterations=3, lam=0.4, mu=-0.43):
    n = len(verts)
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    row = np.asarray(A.sum(axis=1)).ravel()
    row[row == 0] = 1.0
    An = sp.diags(1.0 / row) @ A
    v = verts.astype(np.float64)
    for _ in range(iterations):
        v = v + lam * (An @ v - v)
        v = v + mu * (An @ v - v)
    return v


def vertex_normals(verts, tris):
    n = np.zeros_like(verts)
    f = verts[tris]
    fn = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
    for i in range(3):
        np.add.at(n, tris[:, i], fn)
    ln = np.linalg.norm(n, axis=1)
    ln[ln == 0] = 1.0
    return n / ln[:, None]


def fix_winding(verts, tris):
    """Ensure positive signed volume (outward CCW in right-handed space)."""
    f = verts[tris].astype(np.float64)
    vol = np.einsum("ij,ij->i", f[:, 0], np.cross(f[:, 1], f[:, 2])).sum() / 6.0
    if vol < 0:
        tris = tris[:, [0, 2, 1]].copy()
    return tris, vol


# ------------------------------------------------------------ color sampling
def sample_colors(vol, pts_zyx_idx):
    """Alpha-weighted trilinear sampling of an RGBA uint8 volume [z,y,x,4].
    Ignores empty voxels so surface colors don't bleed toward black."""
    nz, ny, nx = vol.shape[:3]
    p = np.clip(pts_zyx_idx, 0.0, np.array([nz - 1.001, ny - 1.001, nx - 1.001]))
    base = np.floor(p).astype(np.int64)
    frac = p - base
    rgb = np.zeros((len(p), 3), dtype=np.float64)
    wsum = np.zeros(len(p), dtype=np.float64)
    volf = vol.astype(np.float64)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (frac[:, 0] if dz else 1 - frac[:, 0]) * (frac[:, 1] if dy else 1 - frac[:, 1]) * (frac[:, 2] if dx else 1 - frac[:, 2])
                corner = volf[base[:, 0] + dz, base[:, 1] + dy, base[:, 2] + dx]
                a = corner[:, 3] / 255.0
                wa = w * a
                rgb += wa[:, None] * corner[:, :3]
                wsum += wa
    ok = wsum > 1e-6
    rgb[ok] /= wsum[ok, None]
    rgb[~ok] = 128.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ------------------------------------------------------------- frame builder
def build_slime_frame(X, Y, Z, pose):
    d = spirit_sdf(X, Y, Z, pose)
    vol = sculpt_spirit_frame(X, Y, Z, pose)
    verts_idx, tris = extract_mesh(d)          # (x, y, z) grid-index coords
    # grid-index -> demo world coords (128 grid, coords were scaled x0.5)
    verts = np.stack([
        (verts_idx[:, 0] - (GRID - 1) / 2.0) * VOX,
        (verts_idx[:, 1] - (GRID - 1) / 2.0) * VOX,
        (verts_idx[:, 2] + 0.5) * VOX,
    ], axis=1)                                  # demo (X, Y, Z)
    verts = taubin_smooth(verts, tris)
    tris, vol_signed = fix_winding(verts, tris)
    nrm = vertex_normals(verts, tris)
    # sample colors slightly INSIDE the surface so the face-paint shell
    # (painted at d >= -2.5) is hit reliably; alpha weighting skips empties.
    sample_pts = verts - nrm * 0.5
    pts_zyx = np.stack([
        sample_pts[:, 2] / VOX - 0.5,
        sample_pts[:, 1] / VOX + (GRID - 1) / 2.0,
        sample_pts[:, 0] / VOX + (GRID - 1) / 2.0,
    ], axis=1)
    colors = sample_colors(vol, pts_zyx)
    # demo (X right, Y forward, Z up) -> three.js (x, y up, z toward viewer),
    # centered: ty = Z - CENTER_Z. (Mirror flip is invisible: the slime is
    # bilaterally symmetric; winding was fixed in this space below.)
    three = np.stack([verts[:, 0], verts[:, 2] - CENTER_Z, verts[:, 1]], axis=1)
    tris, _ = fix_winding(three, tris)
    return three, tris, colors


def build_star():
    X, Y, Z = make_grid(32, 32, 32)
    X, Y, Z = X.astype(np.float64), Y.astype(np.float64), Z.astype(np.float64)
    d = star_sdf(X, Y, Z, 0)
    vol = sculpt_star_companion(0)
    verts_idx, tris = extract_mesh(d.astype(np.float32))
    verts = np.stack([verts_idx[:, 0] - 15.5, verts_idx[:, 1] - 15.5, verts_idx[:, 2] + 0.5], axis=1)
    verts = taubin_smooth(verts, tris, iterations=2)
    tris, _ = fix_winding(verts, tris)
    nrm = vertex_normals(verts, tris)
    sample_pts = verts - nrm * 0.4
    pts_zyx = np.stack([sample_pts[:, 2] - 0.5, sample_pts[:, 1] + 15.5, sample_pts[:, 0] + 15.5], axis=1)
    colors = sample_colors(vol, pts_zyx)
    three = np.stack([verts[:, 0], verts[:, 2] - 16.0, verts[:, 1]], axis=1)
    tris, _ = fix_winding(three, tris)
    return three, tris, colors


# ----------------------------------------------------------------- encoding
def encode_mesh(verts, tris, colors):
    pos16 = np.clip(np.round(verts * 100.0), -32768, 32767).astype(np.int16)
    idx32 = len(verts) >= 65536
    idx = tris.astype(np.uint32 if idx32 else np.uint16)
    return {
        "nv": int(len(verts)),
        "nt": int(len(tris)),
        "idx32": bool(idx32),
        "pos": base64.b64encode(pos16.tobytes()).decode(),
        "col": base64.b64encode(colors.astype(np.uint8).tobytes()).decode(),
        "idx": base64.b64encode(idx.tobytes()).decode(),
        "raw": pos16.nbytes + colors.nbytes + idx.nbytes,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Variant G - three.js Mesh POC</title>
<style>
  html, body { margin:0; padding:0; height:100%; overflow:hidden;
    background: radial-gradient(120% 90% at 50% 30%, #22355e 0%, #101a30 55%, #070b16 100%);
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif; color:#dceaff; }
  #app { position:fixed; inset:0; touch-action:none; }
  canvas { display:block; }
  .hud { position:fixed; left:14px; top:14px; z-index:10; pointer-events:none;
    background: rgba(10,18,36,.55); border:1px solid rgba(140,190,255,.25);
    border-radius:12px; padding:10px 14px; backdrop-filter: blur(6px); }
  .hud h1 { margin:0; font-size:15px; letter-spacing:.14em; font-weight:700; }
  .hud p { margin:3px 0 0; font-size:11px; letter-spacing:.08em; color:#9db9dd; }
  .hint { position:fixed; left:50%; bottom:16px; transform:translateX(-50%); z-index:10;
    pointer-events:none; font-size:11px; letter-spacing:.1em; color:#8fb0da;
    background: rgba(10,18,36,.45); border:1px solid rgba(140,190,255,.18);
    border-radius:999px; padding:6px 14px; white-space:nowrap; }
</style>
</head>
<body>
<div id="app"></div>
<div class="hud">
  <h1>VARIANT G: THREE.JS MESH</h1>
  <p id="sub">REAL 3D ENGINE RE-RENDER - SURFACE NETS + SMOOTH MESH</p>
</div>
<div class="hint">DRAG: ROTATE&nbsp;&nbsp;/&nbsp;&nbsp;WHEEL OR PINCH: ZOOM</div>
<script>
__THREE_JS__
</script>
<script>
"use strict";
const DATA = __DATA_JSON__;

function b64ToBytes(b64) {
  const s = atob(b64);
  const b = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i);
  return b;
}
function buildGeometry(fr) {
  const pos16 = new Int16Array(b64ToBytes(fr.pos).buffer);
  const pos = new Float32Array(pos16.length);
  for (let i = 0; i < pos16.length; i++) pos[i] = pos16[i] / 100.0;
  const colAttr = new THREE.BufferAttribute(b64ToBytes(fr.col), 3);
  colAttr.normalized = true;
  const idxBytes = b64ToBytes(fr.idx);
  const idx = fr.idx32 ? new Uint32Array(idxBytes.buffer) : new Uint16Array(idxBytes.buffer);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", colAttr);
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  return g;
}

const container = document.getElementById("app");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 1000);

// Lighting matched to the demo's baked cel light dir (-0.22, 0.38 fwd, 0.90 up)
// mapped into three.js space (x, up, toward viewer).
const hemi = new THREE.HemisphereLight(0xeaf6ff, 0x33415e, 0.95);
scene.add(hemi);
const dir = new THREE.DirectionalLight(0xffffff, 0.45);
dir.position.set(-22, 90, 38);
scene.add(dir);

const mat = new THREE.MeshLambertMaterial({ vertexColors: true });
const geos = DATA.frames.map(buildGeometry);
const slime = new THREE.Mesh(geos[0], mat);
scene.add(slime);

const starGeo = buildGeometry(DATA.star);
const star = new THREE.Mesh(starGeo, mat);
star.scale.setScalar(0.9);
scene.add(star);

document.getElementById("sub").textContent =
  "REAL 3D ENGINE RE-RENDER - " + DATA.frames.length + " MESH FRAMES @ 24 FPS, " +
  DATA.frames[0].nv.toLocaleString() + " VERTS / " + DATA.frames[0].nt.toLocaleString() + " TRIS";

// Minimal orbit control: drag rotate, wheel / pinch zoom, idle auto-rotate.
let theta = 0.6, phi = 1.12, radius = 100;
let lastInteract = -1e9;
const pointers = new Map();
let pinchDist = 0;
const el = renderer.domElement;
el.addEventListener("pointerdown", (e) => {
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 2) {
    const p = [...pointers.values()];
    pinchDist = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
  }
  lastInteract = performance.now();
});
el.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId)) return;
  const prev = pointers.get(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  lastInteract = performance.now();
  if (pointers.size === 1) {
    theta -= (e.clientX - prev.x) * 0.008;
    phi -= (e.clientY - prev.y) * 0.008;
    phi = Math.max(0.15, Math.min(2.9, phi));
  } else if (pointers.size === 2) {
    const p = [...pointers.values()];
    const d = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
    if (pinchDist > 0) radius *= pinchDist / d;
    pinchDist = d;
    radius = Math.max(45, Math.min(220, radius));
  }
});
const endPointer = (e) => { pointers.delete(e.pointerId); lastInteract = performance.now(); };
el.addEventListener("pointerup", endPointer);
el.addEventListener("pointercancel", endPointer);
el.addEventListener("wheel", (e) => {
  e.preventDefault();
  radius *= (1 + Math.sign(e.deltaY) * 0.09);
  radius = Math.max(45, Math.min(220, radius));
  lastInteract = performance.now();
}, { passive: false });

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const frameInterval = DATA.frameStep / 24.0;
let last = performance.now(), acc = 0, frame = 0;
function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  acc += dt;
  while (acc >= frameInterval) {
    acc -= frameInterval;
    frame = (frame + 1) % geos.length;
  }
  slime.geometry = geos[frame];
  if (now - lastInteract > 2500) theta += dt * 0.25;
  const t = now / 1000;
  const a = t * (Math.PI * 2 / 4.0);
  star.position.set(Math.cos(a) * 30, 8 + Math.sin(a * 2) * 3.5, Math.sin(a) * 21);
  star.rotation.y = -a;
  star.rotation.z = Math.sin(t * 2.0) * 0.25;
  camera.position.set(
    radius * Math.sin(phi) * Math.sin(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.cos(theta)
  );
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);
</script>
</body>
</html>
"""


def main():
    t0 = time.time()
    with open(THREE_JS_PATH, "r", encoding="utf-8") as f:
        three_js = f.read()
    assert "</script" not in three_js.lower(), "three.js contains </script>"

    X, Y, Z = make_grid(GRID, GRID, GRID)
    X, Y, Z = X * VOX, Y * VOX, Z * VOX

    poses = [evaluate_demo_pose("bounce", f, 24) for f in range(24)]

    # Build frame 0 first to size the payload and pick the frame stride.
    v0, t0m, c0 = build_slime_frame(X, Y, Z, poses[0])
    enc0 = encode_mesh(v0, t0m, c0)
    per_frame_b64 = len(enc0["pos"]) + len(enc0["col"]) + len(enc0["idx"])
    print(f"frame 0: {enc0['nv']} verts, {enc0['nt']} tris, b64 payload {per_frame_b64/1024:.0f} KiB")

    fixed = len(three_js) + 20000  # template + star + manifest slack
    star_v, star_t, star_c = build_star()
    enc_star = encode_mesh(star_v, star_t, star_c)
    fixed += len(enc_star["pos"]) + len(enc_star["col"]) + len(enc_star["idx"])
    print(f"star: {enc_star['nv']} verts, {enc_star['nt']} tris")

    stride = 1
    if fixed + per_frame_b64 * 24 > SIZE_BUDGET:
        stride = 2
    if fixed + per_frame_b64 * (24 // stride) > SIZE_BUDGET:
        stride = 24  # frame 0 only
    n_frames = 24 // stride
    print(f"stride={stride} -> shipping {n_frames} mesh frames; "
          f"projected HTML {(fixed + per_frame_b64 * n_frames) / 1048576:.2f} MiB")

    frames = [enc0]
    for f in range(stride, 24, stride):
        v, tm, col = build_slime_frame(X, Y, Z, poses[f])
        frames.append(encode_mesh(v, tm, col))
        print(f"  frame {f}: {frames[-1]['nv']} verts, {frames[-1]['nt']} tris "
              f"({time.time() - t0:.1f}s elapsed)")

    data = {"frameStep": stride, "frames": frames, "star": enc_star}
    data_json = json.dumps(data, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__THREE_JS__", three_js).replace("__DATA_JSON__", data_json)

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    size = os.path.getsize(OUT_HTML)
    print(f"Wrote {OUT_HTML}: {size} bytes ({size/1048576:.2f} MiB) in {time.time() - t0:.1f}s")
    print(f"frames shipped: {len(frames)} (stride {stride}), "
          f"verts/frame ~{frames[0]['nv']}, tris/frame ~{frames[0]['nt']}")


if __name__ == "__main__":
    main()
