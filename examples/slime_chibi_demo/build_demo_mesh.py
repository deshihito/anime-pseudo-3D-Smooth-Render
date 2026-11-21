"""Mesh builder: the slime demo re-rendered as a three.js polygon mesh
inside the STOCK H player UI (the G7 build of the mesh line).

G6 proved the mesh content (surface-projected geometry, gradient
normals, fixed winding, face decal, wrap lighting) but shipped it in
a bare debug page: dark radial gradient, a floating title card, one
hint pill. The user asked for the pure stock H UI with the content
left as-is.

G7 therefore keeps every DATA payload field byte-identical to G6
(proved in main() against the shipped G6 file, field by field) and
rebuilds only the page chrome to the stock spec of
engine/anime_ui_kit.js + engine/anime_ui_theme.css -- the UI the H
standalone (variants/slime-H-light.html, and the repo port
deshihito/anime-pseudo-3D-Smooth-Render) ships:

  * app-shell layout: stage (left) + 376px studio panel (right);
    stage badge (title/subtitle), HUD chip, hint box, and the
    touch-rotate group (BOUNCE / <3D / 3D>) in the stage footer;
  * panel sections: MOTIONS shelf + TIMELINE scrubber, RENDER STYLE
    mode grid, VIEW ANGLES presets, ANIM & 3D CONTROLS sliders
    (FPS / PITCH / ZOOM), DISPLAY & EFFECTS toggles (SPIN / FX /
    OUTLINE / theme / SPRING / RESET);
  * the stock backdrop itself: the engine's background shader
    (vignette + 32px grid + twinkling star specks) with the three
    stock themes, defaulting to ICE LAB (the H pack's themeIdx 1);
  * stock control semantics and H pack defaults throughout: fps 24,
    pitch 20, zoom 1.45, angle 22, spin 0.42 rad/s, spring
    k=160/damp=16, wheel zoom in +-0.1 steps, drag stops the spin,
    setFrame pauses playback, FACE -> zoom 2.25 + pitch 20, TOP
    toggles pitch 90 <-> 20, RESET restores pack defaults.

Render-mode mapping for a polygon renderer (kit labels kept):
  PIXEL+AA  = pre-G6 hard 4-step Nearest ramp, wrap off, outline on
  RETRO PX  = hard ramp + renderer pixel ratio forced to 0.5
  SMOOTH AA = G6 wrap ramp, outline hull off
  SILKY 3D  = G6 wrap ramp + outline (default; exactly G6's shading)
The wrap remap is a shader uniform (uWrapAmt), so switching modes
never recompiles a program. Entering a mode also sets the outline
default per the table; the OUTLINE toggle remains a live master
switch afterwards.

Omitted rather than faked (no honest mesh equivalent): the DOT SIZE
and SPACING sliders (a polygon mesh has no dot grid and no z-slices),
the PACKAGE (.ap3d) load/merge/export section, the drag-drop
overlay, the empty-stage banner and the status toast (a baked mesh
payload cannot load sprite-stack packages). The FX toggle drives
the scene's only non-character animated element, the orbiting star
companion. Section-head/badge icons exist in the DOM exactly as the
kit builds them but stay hidden: the mesh payload carries no icon
sprite assets (the kit's hydrateIcons hides unresolved icons too).

Output: examples/slime_chibi_demo/build_mesh/slime-chibi-standalone.html
(single file, zero external requests)
"""
import base64
import io
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from build_demo import evaluate_demo_pose, sculpt_star_companion  # noqa: E402
from build_demo_threejs import (  # noqa: E402
    extract_mesh, taubin_smooth, fix_winding, spirit_sdf, star_sdf,
    vertex_normals,
)
from toolkit.gpu_sdf import make_grid  # noqa: E402

THREE_JS_PATH = os.path.join(REPO_ROOT, "build_threejs", "three.min.js")
THEME_CSS_PATH = os.path.join(REPO_ROOT, "engine", "anime_ui_theme.css")
OUT_HTML = os.path.join(HERE, "build_mesh", "slime-chibi-standalone.html")
G5_HTML = os.path.join(REPO_ROOT, "variants", "slime-G5-threejs.html")
G6_HTML = os.path.join(REPO_ROOT, "variants", "slime-G6-threejs.html")
G4_HTML = os.path.join(REPO_ROOT, "variants", "slime-G4-threejs.html")
SIZE_BUDGET = 5 * 1024 * 1024

PROJ_ROUNDS = 4      # Taubin + Newton-projection rounds after decimation
PROJ_NEWTON_STEPS = 2

GRID = 128
VOX = 0.5
CENTER_Z = 24.0

# Unshaded base albedo (the `base` argument of shade_cel_3step in build_demo).
AQUA = (86, 178, 250)
BELLY = (244, 250, 255)
STAR_YELLOW = (252, 214, 92)
SLIME_PALETTE = [AQUA, BELLY]
STAR_PALETTE = [STAR_YELLOW]

# Face projector window in POSE space (the space sculpt_spirit_frame paints
# in): PX in [FACE_X0, FACE_X0+FACE_W], PZ in [FACE_Z0, FACE_Z0+FACE_H].
FACE_X0, FACE_W = -12.5, 25.0
FACE_Z0, FACE_H = 17.0, 12.5
FACE_TEX_W, FACE_TEX_H = 2048, 1024

GRAD_H = 0.05  # central-difference step for field gradients (field units)

# ------------------------------------------------------------- G6 lighting
# Single source of truth for the viewer's lights and toon ramp: the same
# numbers are injected into the HTML template (via render_html) and used
# by the numpy strata audit, so the audit simulates exactly what ships.
DIR_POS = (-62.0, 30.0, 55.0)     # key light position (three.js space);
# diagonal front-left-top, tuned by the strata audit: of the candidates
# scanned it gave the best worst-camera eta^2 with a balanced tone
# occupancy (shadow .12 / transition .22 / lit .67 from the front)
DIR_INTENSITY = 1.0
AMB_COLOR = (234, 246, 255)       # 0xeaf6ff flat fill (no normal term)
AMB_INTENSITY = 0.35
WRAP = 0.4                        # wrap-lighting remap width
RAMP_TEXELS = 64
RAMP_SHADOW = 0.42                # ramp plateaus (irradiance multipliers)
RAMP_LIT = 1.0
RAMP_LO, RAMP_HI = 0.45, 0.68     # smoothstep span in wrapped coord

# G5 lighting, kept only so the strata audit can simulate the "before".
OLD_DIR_POS = (-22.0, 90.0, 38.0)
OLD_DIR_INTENSITY = 0.95
OLD_HEMI_SKY = (234, 246, 255)
OLD_HEMI_GROUND = (51, 65, 94)
OLD_HEMI_INTENSITY = 0.55
OLD_GRAD4 = (70, 135, 195, 255)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ramp_texel_values():
    """RAMP_TEXELS gradient-map values (float), one per texel center of
    the WRAPPED coordinate u in [0, 1]: shadow plateau, smoothstep, lit
    plateau."""
    u = (np.arange(RAMP_TEXELS) + 0.5) / RAMP_TEXELS
    return RAMP_SHADOW + (RAMP_LIT - RAMP_SHADOW) * _smoothstep(
        RAMP_LO, RAMP_HI, u)


def ramp_bytes():
    """RGBA byte list for the JS DataTexture (gray ramp, opaque alpha)."""
    v = np.clip(np.round(ramp_texel_values() * 255.0), 0, 255).astype(np.uint8)
    rgba = np.stack([v, v, v, np.full_like(v, 255)], axis=1)
    return rgba.ravel().tolist()


def sample_ramp(coord):
    """Mirror of the GPU's sampling of the shipped gradient texture:
    LinearFilter over texel centers, clamped to the edge texels."""
    vals = np.clip(np.round(ramp_texel_values() * 255.0), 0, 255) / 255.0
    x = np.clip(coord, 0.0, 1.0) * RAMP_TEXELS - 0.5
    i0 = np.floor(x).astype(int)
    frac = x - i0
    i0c = np.clip(i0, 0, RAMP_TEXELS - 1)
    i1c = np.clip(i0 + 1, 0, RAMP_TEXELS - 1)
    return vals[i0c] * (1.0 - frac) + vals[i1c] * frac


# ------------------------------------------------------- field-gradient normals
def _normalize_rows(v, fallback):
    ln = np.linalg.norm(v, axis=1)
    bad = ln < 1e-9
    out = np.empty_like(v)
    if np.any(~bad):
        out[~bad] = v[~bad] / ln[~bad, None]
    if np.any(bad):
        fb = np.asarray(fallback, dtype=np.float64)[bad]
        fl = np.linalg.norm(fb, axis=1)
        fl[fl == 0] = 1.0
        out[bad] = fb / fl[:, None]
    return out


def _field_gradient(field_fn, pts):
    grad = np.empty_like(pts)
    for ax in range(3):
        d = np.zeros(3)
        d[ax] = GRAD_H
        fp = field_fn(pts[:, 0] + d[0], pts[:, 1] + d[1], pts[:, 2] + d[2])
        fm = field_fn(pts[:, 0] - d[0], pts[:, 1] - d[1], pts[:, 2] - d[2])
        grad[:, ax] = (fp - fm) / (2.0 * GRAD_H)
    return grad


def spirit_gradient_normals(verts_demo, pose):
    """Outward unit normals from the analytic spirit_sdf gradient, evaluated
    at the final (decimated/smoothed) demo-space vertex positions. Field
    space IS demo space: make_grid returns centered coordinates
    (X = ix - 63.5, Y = iy - 63.5, Z = iz + 0.5, scaled by VOX), which is
    exactly the conversion extract_mesh's output goes through, so the field
    can be evaluated at the vertex positions directly. The SDF is negative
    inside, so the gradient points outward by construction."""
    grad = _field_gradient(lambda x, y, z: spirit_sdf(x, y, z, pose),
                           verts_demo)
    center = np.array([0.0, 0.0, CENTER_Z])  # body centre, fallback only
    return _normalize_rows(grad, verts_demo - center)


def star_gradient_normals(verts_demo):
    """Same treatment for the star companion: make_grid(32,32,32) coords are
    already the demo-space coords build_star_g5 produces, so evaluate
    star_sdf at the vertex positions directly."""
    grad = _field_gradient(lambda x, y, z: star_sdf(x, y, z, 0), verts_demo)
    center = np.array([0.0, 0.0, 16.0])
    return _normalize_rows(grad, verts_demo - center)


def neighbor_angle_metric(tris, normals):
    """Mean angle (deg) between each vertex normal and the renormalized
    average of its edge-neighbors' normals. Lower = smoother normal field."""
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    avg = np.zeros_like(normals)
    np.add.at(avg, edges[:, 0], normals[edges[:, 1]])
    np.add.at(avg, edges[:, 1], normals[edges[:, 0]])
    deg = np.zeros(len(normals))
    np.add.at(deg, edges[:, 0], 1.0)
    np.add.at(deg, edges[:, 1], 1.0)
    m = deg > 0
    a = avg[m]
    a = a / np.linalg.norm(a, axis=1)[:, None]
    dots = np.clip(np.einsum("ij,ij->i", normals[m], a), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).mean())


def repair_winding(verts, tris, vert_normals):
    """Flip every triangle whose winding normal disagrees with the average
    of its vertex (gradient) normals, by swapping two indices. Returns
    (tris, n_flipped). Surface-nets quads wind inconsistently (fixed cyclic
    order per edge direction), so ~50% of faces start out inward; the net
    signed-volume check in fix_winding cannot see that."""
    f = verts[tris].astype(np.float64)
    fn = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
    gn = vert_normals[tris].mean(axis=1)
    dots = np.einsum("ij,ij->i", fn, gn)
    flip = dots < 0
    out = tris.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    return out, int(flip.sum())


# ---------------------------------------------------------------- decimation
def decimate_cluster(verts, tris, cells=(0.7, 0.85, 1.0, 1.15, 1.3),
                     target_hi=20000):
    """Vertex-clustering decimation. Tries cell sizes in order and keeps the
    first result at or below target_hi triangles. Returns (verts, tris, cell).
    """
    best = None
    for cell in cells:
        keys = np.floor(verts / cell).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        nv = int(inverse.max()) + 1
        sums = np.zeros((nv, 3), dtype=np.float64)
        np.add.at(sums, inverse, verts)
        cnt = np.bincount(inverse, minlength=nv).astype(np.float64)
        new_v = sums / cnt[:, None]
        new_t = inverse[tris]
        keep = ((new_t[:, 0] != new_t[:, 1]) & (new_t[:, 1] != new_t[:, 2])
                & (new_t[:, 2] != new_t[:, 0]))
        new_t = new_t[keep]
        # drop exact duplicate triangles
        srt = np.sort(new_t, axis=1)
        _, uniq = np.unique(srt, axis=0, return_index=True)
        new_t = new_t[np.sort(uniq)]
        # drop unreferenced verts and reindex
        used, remap = np.unique(new_t, return_inverse=True)
        new_v = new_v[used]
        new_t = remap.reshape(new_t.shape).astype(np.int32)
        best = (new_v, new_t, cell)
        if len(new_t) <= target_hi:
            break
    return best


def signed_volume(verts, tris):
    f = verts[tris].astype(np.float64)
    return np.einsum("ij,ij->i", f[:, 0], np.cross(f[:, 1], f[:, 2])).sum() / 6.0


# ------------------------------------------- G5: projection + terracing
def project_to_surface(verts, field_fn, steps=PROJ_NEWTON_STEPS):
    """Newton-project every vertex onto the field's zero-set along the
    gradient: p -= d(p) * grad(d) / |grad(d)|^2. The field is negative
    inside, so this moves off-surface vertices straight back onto the
    analytic surface regardless of sign."""
    p = verts.copy()
    for _ in range(steps):
        d = field_fn(p[:, 0], p[:, 1], p[:, 2])
        g = _field_gradient(field_fn, p)
        g2 = np.einsum("ij,ij->i", g, g)
        g2[g2 < 1e-12] = 1e-12
        p = p - (d / g2)[:, None] * g
    return p


def smooth_project_rounds(verts, tris, field_fn, rounds=PROJ_ROUNDS):
    """rounds x [Taubin pass -> projection]. Taubin redistributes vertices
    tangentially (off the clustering grid); projection restores the exact
    surface radially. Neither changes topology or vertex order."""
    v = verts
    for _ in range(rounds):
        v = taubin_smooth(v, tris, iterations=1)
        v = project_to_surface(v, field_fn)
    return v


def field_abs_stats(verts, field_fn):
    d = np.abs(field_fn(verts[:, 0], verts[:, 1], verts[:, 2]))
    return float(d.mean()), float(d.max())


def _unit_rows(v):
    ln = np.linalg.norm(v, axis=1)
    out = np.zeros_like(v)
    ok = ln > 1e-12
    out[ok] = v[ok] / ln[ok, None]
    out[~ok] = np.array([0.0, 0.0, 1.0])
    return out


def _pair_face_angles(verts, tris):
    """Unsigned angle (deg) between edge-adjacent face normals. Unsigned
    because surface-nets winding is mixed: a signed version reads smooth
    coplanar opposite-wound neighbours as ~180 deg and measures winding,
    not roughness."""
    f = verts[tris].astype(np.float64)
    fn = _unit_rows(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]))
    fe = np.stack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1)
    fe = np.sort(fe.reshape(-1, 2), axis=1)
    _, inv, cnt = np.unique(fe, axis=0, return_inverse=True, return_counts=True)
    face_of_edge = np.repeat(np.arange(len(tris)), 3)
    pair = cnt[inv] == 2
    order = np.argsort(inv[pair], kind="stable")
    fs = face_of_edge[pair][order]
    a, b = fs[0::2], fs[1::2]
    dots = np.abs(np.einsum("ij,ij->i", fn[a], fn[b]))
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def face_vs_field_angle(verts, tris, field_fn):
    """Mean angle (deg) between each face's (unsigned) normal and the
    analytic gradient normal at its centroid: how far the piecewise-flat
    surface's facets sit from the smooth field's tangent planes."""
    f = verts[tris].astype(np.float64)
    fn = _unit_rows(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]))
    cen = f.mean(axis=1)
    g = _unit_rows(_field_gradient(field_fn, cen))
    dots = np.abs(np.einsum("ij,ij->i", fn, g))
    return float(np.degrees(np.arccos(np.clip(dots, -1.0, 1.0))).mean())


# ------------------------------------------------------------- frame builder
def build_slime_frame_g5(X, Y, Z, pose):
    d = spirit_sdf(X, Y, Z, pose)
    verts_idx, tris = extract_mesh(d)          # (x, y, z) grid-index coords
    verts = np.stack([
        (verts_idx[:, 0] - (GRID - 1) / 2.0) * VOX,
        (verts_idx[:, 1] - (GRID - 1) / 2.0) * VOX,
        (verts_idx[:, 2] + 0.5) * VOX,
    ], axis=1)                                  # demo (X, Y, Z)
    verts = taubin_smooth(verts, tris)
    vol_before = signed_volume(verts, tris)

    verts, tris, cell = decimate_cluster(verts, tris)
    # G5: replace G4's post-decimation Taubin(2) with Taubin+projection
    # rounds, still in demo/field space. Record the pre-round state for
    # the on-surface / displacement report.
    field_fn = lambda x, y, z: spirit_sdf(x, y, z, pose)  # noqa: E731
    pre_verts = verts.copy()
    pre_d = field_abs_stats(verts, field_fn)
    verts = smooth_project_rounds(verts, tris, field_fn)
    post_d = field_abs_stats(verts, field_fn)
    disp = np.linalg.norm(verts - pre_verts, axis=1)
    stats = {
        "d_pre": pre_d, "d_post": post_d,
        "disp_mean": float(disp.mean()), "disp_max": float(disp.max()),
        "pair_pre": float(_pair_face_angles(pre_verts, tris).mean()),
        "pair_post": float(_pair_face_angles(verts, tris).mean()),
        "fvf_pre": face_vs_field_angle(pre_verts, tris, field_fn),
        "fvf_post": face_vs_field_angle(verts, tris, field_fn),
        "zmax_pre": float(pre_verts[:, 2].max()),
        "zmax_post": float(verts[:, 2].max()),
    }
    if np.sign(signed_volume(verts, tris)) != np.sign(vol_before):
        tris = tris[:, [0, 2, 1]].copy()

    # Albedo region from the sculpt's own belly mask, evaluated at the final
    # (decimated, smoothed) vertex positions in pose space. 0=aqua, 1=belly.
    PX = verts[:, 0] / pose.sx
    PY = verts[:, 1] / pose.stretch_y
    PZ = (verts[:, 2] - pose.root_bob) / pose.squash_z
    belly = (PY > 3.5) & (PZ < 20.5) & (np.abs(PX) < 11.5)
    region = belly.astype(np.uint8)

    # Vertex normals from the analytic field gradient at the final
    # (decimated, smoothed) vertex positions, still in demo space here.
    nrm_demo = spirit_gradient_normals(verts, pose)

    # demo (X right, Y forward, Z up) -> three.js (x, y up, z toward viewer).
    # Positions AND normals take the same permutation [x, z, y]; only the
    # position's y component carries the -CENTER_Z translation.
    three = np.stack([verts[:, 0], verts[:, 2] - CENTER_Z, verts[:, 1]], axis=1)
    tris, _ = fix_winding(three, tris)
    normals = np.stack([nrm_demo[:, 0], nrm_demo[:, 2], nrm_demo[:, 1]], axis=1)
    # G4's repair, unchanged: per-face winding against the gradient
    # normals. fix_winding above only fixes the net volume sign; ~50% of
    # faces still wind inward. (Projection moves vertices smoothly and
    # should not flip faces; the shipped-payload audit proves it didn't.)
    tris, n_flip = repair_winding(three, tris, normals)
    # Mesh normals from the REPAIRED winding (metric only): in G3 these
    # disagreed with the field normals (mean dot -0.001); after the repair
    # they should agree, which cross-checks the fix.
    mesh_normals = vertex_normals(three, tris)
    pose4 = [round(float(pose.sx), 4), round(float(pose.stretch_y), 4),
             round(float(pose.squash_z), 4), round(float(pose.root_bob), 4)]
    return three, tris, region, normals, mesh_normals, pose4, cell, n_flip, stats


def build_star_g5():
    X, Y, Z = make_grid(32, 32, 32)
    X, Y, Z = X.astype(np.float64), Y.astype(np.float64), Z.astype(np.float64)
    d = star_sdf(X, Y, Z, 0)
    verts_idx, tris = extract_mesh(d.astype(np.float32))
    verts = np.stack([verts_idx[:, 0] - 15.5, verts_idx[:, 1] - 15.5,
                      verts_idx[:, 2] + 0.5], axis=1)
    # G5: same projection rounds as the slime, replacing G4's Taubin(2).
    # The star is extracted from a coarse 32^3 grid (1.0-unit voxels), so
    # its pre-round on-surface error is the largest in the build.
    field_fn = lambda x, y, z: star_sdf(x, y, z, 0)  # noqa: E731
    pre_verts = verts.copy()
    pre_d = field_abs_stats(verts, field_fn)
    verts = smooth_project_rounds(verts, tris, field_fn)
    post_d = field_abs_stats(verts, field_fn)
    disp = np.linalg.norm(verts - pre_verts, axis=1)
    stats = {
        "d_pre": pre_d, "d_post": post_d,
        "disp_mean": float(disp.mean()), "disp_max": float(disp.max()),
        "pair_pre": float(_pair_face_angles(pre_verts, tris).mean()),
        "pair_post": float(_pair_face_angles(verts, tris).mean()),
        "fvf_pre": face_vs_field_angle(pre_verts, tris, field_fn),
        "fvf_post": face_vs_field_angle(verts, tris, field_fn),
    }
    region = np.zeros(len(verts), dtype=np.uint8)
    nrm_demo = star_gradient_normals(verts)
    three = np.stack([verts[:, 0], verts[:, 2] - 16.0, verts[:, 1]], axis=1)
    tris, _ = fix_winding(three, tris)
    normals = np.stack([nrm_demo[:, 0], nrm_demo[:, 2], nrm_demo[:, 1]], axis=1)
    tris, n_flip = repair_winding(three, tris, normals)
    return three, tris, region, normals, n_flip, stats


# --------------------------------------------------------------- face texture
def build_face_texture():
    """Rasterize the demo's face-paint ellipses (pose space, from
    sculpt_spirit_frame) into a transparent 2048x1024 PNG data URI.
    Drawn at 2x supersample and downscaled for anti-aliased edges."""
    SS = 2
    W, H = FACE_TEX_W * SS, FACE_TEX_H * SS
    sxu = W / FACE_W          # px per pose-space unit (x)
    szu = H / FACE_H          # px per pose-space unit (z) -- equal to sxu

    def px(x):
        return (x - FACE_X0) * sxu

    def py(z):
        return (FACE_Z0 + FACE_H - z) * szu

    def ellipse(draw, cx, cz, rx, ry, fill):
        draw.ellipse([px(cx - rx), py(cz + ry), px(cx + rx), py(cz - ry)],
                     fill=fill)

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    # eyes + highlights (demo order: eye bg, highlight, then blush, then mouth)
    for cx in (-6.0, 6.0):
        ellipse(dr, cx, 24.5, 3.2, 3.8, (28, 36, 86, 255))
    for cx in (-6.0, 6.0):
        ellipse(dr, cx - 0.9, 26.2, 1.1, 1.1, (255, 255, 255, 255))
    # blush on its own layer, softened radially (blur), composited over
    blush = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    db = ImageDraw.Draw(blush)
    for cx in (-8.7, 8.7):
        ellipse(db, cx, 19.8, 2.4, 1.4, (255, 152, 178, 235))
    blush = blush.filter(ImageFilter.GaussianBlur(radius=0.30 * sxu))
    img.alpha_composite(blush)
    dr = ImageDraw.Draw(img)
    ellipse(dr, 0.0, 19.5, 2.2, 1.6, (238, 118, 140, 255))

    img = img.resize((FACE_TEX_W, FACE_TEX_H), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------- encoding
def encode_mesh(verts, tris, region, normals, pose4=None):
    pos16 = np.clip(np.round(verts * 100.0), -32768, 32767).astype(np.int16)
    nrm16 = np.clip(np.round(normals * 1000.0), -32768, 32767).astype(np.int16)
    idx32 = len(verts) >= 65536
    idx = tris.astype(np.uint32 if idx32 else np.uint16)
    enc = {
        "nv": int(len(verts)),
        "nt": int(len(tris)),
        "idx32": bool(idx32),
        "pos": base64.b64encode(pos16.tobytes()).decode(),
        "nrm": base64.b64encode(nrm16.tobytes()).decode(),
        "reg": base64.b64encode(region.astype(np.uint8).tobytes()).decode(),
        "idx": base64.b64encode(idx.tobytes()).decode(),
    }
    if pose4 is not None:
        enc["pose"] = pose4
    return enc


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover" />
<title>SLIME SPIRIT 3D DEMO</title>
<style>
__THEME_CSS__
</style>
</head>
<body>
<div id="appRoot"></div>
<script>
__THREE_JS__
</script>
<script>
"use strict";
const DATA = __DATA_JSON__;
const CENTER_Z = __CENTER_Z__;
const FACE_WIN = { x0: __FACE_X0__, w: __FACE_W__, z0: __FACE_Z0__, h: __FACE_H__ };
const WRAP_AMT = __WRAP__;

/* =====================================================================
 * Stock UI chrome: the DOM below mirrors, element for element, what
 * engine/anime_ui_kit.js mountStudioUI() builds for the sprite-stack
 * player (same classes, ids, labels and layout), styled by the stock
 * engine/anime_ui_theme.css inlined above. The stack engine is replaced
 * by an adapter over the three.js scene that implements the same
 * control semantics (stock defaults from the H motion pack:
 * fps 24, renderMode 3, pitch 20, zoom 1.45, theme ICE LAB, angle 22).
 * Controls with no honest mesh equivalent are omitted, not faked:
 * DOT SIZE + SPACING sliders (no dot grid / no z-slices in a polygon
 * mesh) and the PACKAGE (.ap3d) load/merge/export section plus the
 * drag-drop overlay / empty banner / toast (a baked mesh payload
 * cannot load stack packages).
 * ===================================================================== */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}
function icoImg(key, cls) {
  // The mesh payload carries no icon sprite assets; like the kit's
  // hydrateIcons(), unresolved icons stay present but hidden.
  const img = el("img", cls || "px-ico");
  img.setAttribute("data-ico", key);
  img.alt = "";
  img.style.visibility = "hidden";
  return img;
}

/* ------------------------- state (stock-shaped) ------------------------- */
const FC = 24;                       // logical frames per motion cycle
const PACK_DEFAULTS = { motion: "bounce", animFps: 24, renderMode: 3,
  pitchDeg: 20, zoom: 1.45, themeIdx: 1, angleDeg: 22.0 };
const DEFAULT_PITCH_DEG = 20;
const state = {
  motion: "bounce", animPlaying: true, animFps: 24, animPhase: 0,
  manualFrame: 0, renderMode: 3, angle: PACK_DEFAULTS.angleDeg * Math.PI / 180,
  autoSpin: true, spinSpeed: 0.42, pitchDeg: 20, zoom: 1.45,
  faceFocus: false, outlineOn: true, fxOn: true, springOn: true,
  themeIdx: 1, holdRotDir: 0, dragActive: false,
  sqX: 1.0, sqY: 1.0, sqVx: 0.0, sqVy: 0.0,
};
const keys = { left: false, right: false };
const MOTIONS = [{ id: "bounce", label: "BOUNCE", subtitle: "3D STAR PET",
  category: "COMPANION", accent: "companion", frames: FC, impulse: [0.92, 1.10] }];

const THEMES = [
  { name: "DEEP SEA", bg: [13, 22, 48], grid: [24, 39, 80], star: [100, 181, 246], starHi: [212, 240, 255], outline: [34, 38, 82] },
  { name: "ICE LAB", bg: [223, 233, 248], grid: [198, 214, 238], star: [107, 143, 212], starHi: [255, 255, 255], outline: [36, 42, 88] },
  { name: "MIDNIGHT", bg: [18, 15, 36], grid: [34, 28, 66], star: [129, 140, 248], starHi: [254, 240, 138], outline: [24, 20, 56] },
];

/* ------------------------------ DOM build ------------------------------ */
const rootEl = document.getElementById("appRoot");
rootEl.innerHTML = "";
const shell = el("div", "app-shell");

const stageWrap = el("div", "stage-wrap");
const canvas = el("canvas");
canvas.id = "stageCanvas";
stageWrap.appendChild(canvas);

const badge = el("div", "stage-badge");
const badgeIcon = icoImg("ico_brand", "px-ico px-ico-lg");
badgeIcon.id = "badgeIcon";
badgeIcon.alt = "brand";
const badgeText = el("div");
const badgeTitle = el("div", "stage-badge-title", "SLIME SPIRIT 3D DEMO");
badgeTitle.id = "badgeTitle";
const badgeSub = el("div", "stage-badge-sub",
  "VARIANT G7: THREE.JS MESH - TOON + FACE DECAL + WRAP LIGHTING");
badgeSub.id = "badgeSub";
badgeText.appendChild(badgeTitle);
badgeText.appendChild(badgeSub);
badge.appendChild(badgeIcon);
badge.appendChild(badgeText);
stageWrap.appendChild(badge);

const hudChip = el("div", "hud-chip", "BOUNCE F01/24 | 24 FPS | 022 DEG");
hudChip.id = "hudChip";
stageWrap.appendChild(hudChip);

const footer = el("div", "stage-footer");
const hintBox = el("div", "hint-box",
  "1-9 / [ ]: MOTION | DRAG: 360 | F: FACE | B: BOUNCE | SPACE: PLAY");
hintBox.id = "hintBox";
const rotGrp = el("div", "touch-rot-group");
const btnBounce = el("button", "px-btn pink", "BOUNCE");
btnBounce.type = "button"; btnBounce.id = "btnBounce";
const btnRotL = el("button", "px-btn", "<3D");
btnRotL.type = "button"; btnRotL.id = "btnRotL";
const btnRotR = el("button", "px-btn", "3D>");
btnRotR.type = "button"; btnRotR.id = "btnRotR";
rotGrp.appendChild(btnBounce);
rotGrp.appendChild(btnRotL);
rotGrp.appendChild(btnRotR);
footer.appendChild(hintBox);
footer.appendChild(rotGrp);
stageWrap.appendChild(footer);
shell.appendChild(stageWrap);

const panel = el("aside", "panel");

/* ---- panel section 1: MOTIONS shelf + TIMELINE (kit structure) ---- */
const motionSection = el("div", "panel-section");
{
  const head = el("div", "sec-head");
  const headLeft = el("div", "sec-head-left");
  headLeft.appendChild(icoImg("ico_anim"));
  const headTitle = el("span", "", "1 MOTIONS (24F)");
  headTitle.id = "motionHead";
  headLeft.appendChild(headTitle);
  const nav = el("div", "motion-nav-inline");
  nav.style.display = "none";              // kit hides nav for <= 1 motion
  const btnPrev = el("button", "px-btn motion-nav-btn", "<");
  btnPrev.type = "button"; btnPrev.id = "btnMotionPrev";
  btnPrev.title = "Previous Motion ([)";
  const countBadge = el("span", "motion-count-badge", "1/1");
  countBadge.id = "motionCounter";
  const btnNext = el("button", "px-btn motion-nav-btn", ">");
  btnNext.type = "button"; btnNext.id = "btnMotionNext";
  btnNext.title = "Next Motion (])";
  btnPrev.addEventListener("click", () => stepMotion(-1));
  btnNext.addEventListener("click", () => stepMotion(1));
  nav.appendChild(btnPrev); nav.appendChild(countBadge); nav.appendChild(btnNext);
  head.appendChild(headLeft); head.appendChild(nav);
  motionSection.appendChild(head);
  const catBar = el("div", "motion-cat-bar");
  catBar.id = "motionCatBar";
  catBar.style.display = "none";           // kit hides for < 2 categories
  motionSection.appendChild(catBar);
  const shelf = el("div", "motion-shelf");
  const grid = el("div", "motion-grid cols-1");
  grid.id = "motionGrid";
  MOTIONS.forEach((m, i) => {
    const btn = el("button", "px-btn motion-btn accent-" + m.accent +
      (m.id === state.motion ? " active" : ""));
    btn.type = "button";
    btn.setAttribute("data-motion", m.id);
    const top = el("span", "motion-btn-top");
    const ico = icoImg(m.icon || "ico_bounce");
    top.appendChild(ico);
    top.appendChild(el("span", "", (i + 1) + "." + m.label));
    btn.appendChild(top);
    btn.appendChild(el("span", "motion-btn-sub", m.subtitle));
    btn.addEventListener("click", () => selectMotion(m.id));
    grid.appendChild(btn);
  });
  shelf.appendChild(grid);
  motionSection.appendChild(shelf);
}
/* timeline scrubber (kit structure) */
const tlWrap = el("div", "timeline-wrap");
const tlHead = el("div", "timeline-head");
tlHead.appendChild(el("span", "", "TIMELINE (SCRUB)"));
const tlVal = el("span", "val", "F01 / 24");
tlVal.id = "tlVal";
tlHead.appendChild(tlVal);
const tlBar = el("div", "timeline-bar");
tlBar.id = "timelineBar";
tlBar.setAttribute("role", "slider");
tlBar.setAttribute("tabindex", "0");
tlBar.setAttribute("aria-label", "Animation Timeline");
tlBar.setAttribute("aria-valuemin", "1");
tlBar.setAttribute("aria-valuemax", String(FC));
tlBar.style.gridTemplateColumns = "repeat(" + FC + ", 1fr)";
const tlCells = [];
for (let i = 0; i < FC; i++) {
  const c = el("div", "tl-cell" + (i === 0 ? " cur" : ""));
  tlBar.appendChild(c); tlCells.push(c);
}
tlWrap.appendChild(tlHead);
tlWrap.appendChild(tlBar);
motionSection.appendChild(tlWrap);
panel.appendChild(motionSection);

function timelineUpdate(fIdx) {
  const f = Math.max(0, Math.min(FC - 1, fIdx | 0));
  for (let i = 0; i < FC; i++) {
    tlCells[i].className = i === f ? "tl-cell cur" : (i < f ? "tl-cell past" : "tl-cell");
  }
  tlVal.textContent = "F" + String(f + 1).padStart(2, "0") + " / " + FC;
  tlBar.setAttribute("aria-valuenow", String(f + 1));
}
{
  let scrubbing = false;
  const scrubAt = (clientX) => {
    const rect = tlBar.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(0.9999, (clientX - rect.left) / Math.max(1, rect.width)));
    setFrame(Math.floor(ratio * FC));
  };
  tlBar.addEventListener("pointerdown", (e) => {
    e.preventDefault(); scrubbing = true;
    tlBar.setPointerCapture(e.pointerId); scrubAt(e.clientX);
  });
  tlBar.addEventListener("pointermove", (e) => { if (scrubbing) scrubAt(e.clientX); });
  tlBar.addEventListener("pointerup", () => { scrubbing = false; });
  tlBar.addEventListener("pointercancel", () => { scrubbing = false; });
}

/* ---- pixel slider (kit structure + behavior) ---- */
function createPixelSlider(cfg) {
  const row = el("div", "slider-row");
  const lbl = el("span", "slider-label", cfg.label);
  const slider = el("div", "px-slider");
  if (cfg.id) slider.id = cfg.id;
  slider.setAttribute("role", "slider");
  slider.setAttribute("tabindex", "0");
  slider.setAttribute("aria-label", cfg.ariaLabel || cfg.label);
  slider.setAttribute("aria-valuemin", String(cfg.min));
  slider.setAttribute("aria-valuemax", String(cfg.max));
  const track = el("div", "px-slider-track");
  const fill = el("div", "px-slider-fill");
  const ticks = el("div", "px-slider-ticks");
  track.appendChild(fill); track.appendChild(ticks);
  const thumb = el("div", "px-slider-thumb");
  slider.appendChild(track); slider.appendChild(thumb);
  const valSpan = el("span", "slider-val");
  if (cfg.id) valSpan.id = cfg.id.replace(/^sl/, "val");
  const actBtn = el("button", "px-btn slider-action", cfg.actionText || "");
  actBtn.type = "button";
  if (cfg.actionId) actBtn.id = cfg.actionId;
  if (cfg.onAction) actBtn.addEventListener("click", cfg.onAction);
  row.appendChild(lbl); row.appendChild(slider);
  row.appendChild(valSpan); row.appendChild(actBtn);
  function sync() {
    const v = cfg.getVal();
    if (typeof v !== "number" || Number.isNaN(v)) return;
    const pct = Math.max(0, Math.min(1, (v - cfg.min) / Math.max(1e-6, cfg.max - cfg.min))) * 100;
    fill.style.width = pct.toFixed(2) + "%";
    thumb.style.left = pct.toFixed(2) + "%";
    slider.setAttribute("aria-valuenow", String(v));
    valSpan.textContent = cfg.formatVal ? cfg.formatVal(v) : String(v);
  }
  function setFromClientX(clientX) {
    const rect = slider.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
    let v = cfg.min + ratio * (cfg.max - cfg.min);
    v = Math.round(v / cfg.step) * cfg.step;
    v = Math.max(cfg.min, Math.min(cfg.max, v));
    cfg.setVal(v); sync();
  }
  slider.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    slider.setPointerCapture(e.pointerId);
    slider.classList.add("dragging");
    setFromClientX(e.clientX);
  });
  slider.addEventListener("pointermove", (e) => {
    if (!slider.classList.contains("dragging")) return;
    setFromClientX(e.clientX);
  });
  const endDrag = () => slider.classList.remove("dragging");
  slider.addEventListener("pointerup", endDrag);
  slider.addEventListener("pointercancel", endDrag);
  slider.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      e.preventDefault(); cfg.setVal(Math.max(cfg.min, cfg.getVal() - cfg.step)); sync();
    } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      e.preventDefault(); cfg.setVal(Math.min(cfg.max, cfg.getVal() + cfg.step)); sync();
    }
  });
  sync();
  return { el: row, slider, valSpan, actionBtn: actBtn, sync };
}
function sectionHead(iconKey, title) {
  const sh = el("div", "sec-head");
  const sl = el("div", "sec-head-left");
  sl.appendChild(icoImg(iconKey));
  sl.appendChild(el("span", "", title));
  sh.appendChild(sl);
  return sh;
}

/* ---- panel section 2: RENDER STYLE & PIXEL FEEL (mode grid) ---- */
const modeBtns = [];
{
  const sec = el("div", "panel-section");
  sec.appendChild(sectionHead("ico_outline", "RENDER STYLE & PIXEL FEEL"));
  const mgrid = el("div", "mode-grid");
  [["0", "PIXEL+AA"], ["1", "RETRO PX"], ["2", "SMOOTH AA"], ["3", "SILKY 3D"]]
    .forEach(([rm, txt]) => {
      const b = el("button", "px-btn", txt);
      b.type = "button";
      b.setAttribute("data-rmode", rm);
      b.addEventListener("click", () => setRenderMode(Number(rm)));
      mgrid.appendChild(b); modeBtns.push(b);
    });
  sec.appendChild(mgrid);
  panel.appendChild(sec);
}

/* ---- panel section 3: VIEW ANGLES ---- */
{
  const sec = el("div", "panel-section");
  sec.appendChild(sectionHead("ico_cam", "VIEW ANGLES"));
  const pgrid = el("div", "preset-grid");
  [["0", "FRONT"], ["24", "3/4 L"], ["90", "L SIDE"],
   ["180", "BACK"], ["270", "R SIDE"], ["336", "3/4 R"]]
    .forEach(([deg, txt]) => {
      const b = el("button", "px-btn", txt);
      b.type = "button";
      b.setAttribute("data-angle", deg);
      b.addEventListener("click", () => setAngleDeg(Number(deg)));
      pgrid.appendChild(b);
    });
  sec.appendChild(pgrid);
  panel.appendChild(sec);
}

/* ---- panel section 4: ANIM & 3D CONTROLS (FPS / PITCH / ZOOM) ---- */
let fpsSlider, pitchSlider, zoomSlider;
{
  const sec = el("div", "panel-section");
  sec.appendChild(sectionHead("ico_layers", "ANIM & 3D CONTROLS"));
  fpsSlider = createPixelSlider({
    id: "slFps", label: "ANIM FPS", min: 6, max: 36, step: 1,
    getVal: () => state.animFps,
    setVal: (v) => { state.animFps = Math.round(v); },
    formatVal: (v) => String(Math.round(v)),
    actionText: "PLAY", actionId: "btnPlayAnim",
    onAction: () => togglePlay(),
  });
  pitchSlider = createPixelSlider({
    id: "slSquash", label: "PITCH", ariaLabel: "Camera Pitch Angle",
    min: 15, max: 90, step: 1,
    getVal: () => state.pitchDeg,
    setVal: (v) => setPitchDeg(v),
    formatVal: (v) => String(Math.round(v)) + "°",
    actionText: "TOP", actionId: "btnTopView",
    onAction: () => toggleTopView(),
  });
  zoomSlider = createPixelSlider({
    id: "slZoom", label: "ZOOM", min: 0.8, max: 3.2, step: 0.05,
    getVal: () => state.zoom,
    setVal: (v) => { state.zoom = v; },
    formatVal: (v) => v.toFixed(2) + "x",
    actionText: "FACE", actionId: "btnFaceZoom",
    onAction: () => toggleFaceFocus(),
  });
  sec.appendChild(fpsSlider.el);
  sec.appendChild(pitchSlider.el);
  sec.appendChild(zoomSlider.el);
  panel.appendChild(sec);
}

/* ---- panel section 5: DISPLAY & EFFECTS ---- */
let btnAutoSpin, btnFx, btnOutline, btnTheme, txtTheme, btnSpring, btnReset;
{
  const sec = el("div", "panel-section");
  sec.appendChild(sectionHead("ico_theme", "DISPLAY & EFFECTS"));
  const trow = el("div", "toggle-row");
  btnAutoSpin = el("button", "px-btn active"); btnAutoSpin.type = "button"; btnAutoSpin.id = "btnAutoSpin";
  btnAutoSpin.appendChild(el("span", "", "SPIN: ON"));
  btnAutoSpin.addEventListener("click", () => { state.autoSpin = !state.autoSpin; syncAll(); });
  btnFx = el("button", "px-btn active"); btnFx.type = "button"; btnFx.id = "btnFx";
  btnFx.appendChild(el("span", "", "FX: ON"));
  btnFx.addEventListener("click", () => { state.fxOn = !state.fxOn; applyFx(); syncAll(); });
  btnOutline = el("button", "px-btn active"); btnOutline.type = "button"; btnOutline.id = "btnOutline";
  btnOutline.appendChild(icoImg("ico_outline"));
  btnOutline.appendChild(el("span", "", "OUTLINE: ON"));
  btnOutline.addEventListener("click", () => { state.outlineOn = !state.outlineOn; applyOutline(); syncAll(); });
  btnTheme = el("button", "px-btn"); btnTheme.type = "button"; btnTheme.id = "btnTheme";
  btnTheme.appendChild(icoImg("ico_theme"));
  txtTheme = el("span", "", "ICE LAB"); txtTheme.id = "txtTheme";
  btnTheme.appendChild(txtTheme);
  btnTheme.addEventListener("click", () => cycleTheme());
  btnSpring = el("button", "px-btn active"); btnSpring.type = "button"; btnSpring.id = "btnSpring";
  btnSpring.appendChild(el("span", "", "SPRING: ON"));
  btnSpring.addEventListener("click", () => { state.springOn = !state.springOn; syncAll(); });
  btnReset = el("button", "px-btn gold"); btnReset.type = "button"; btnReset.id = "btnReset";
  btnReset.appendChild(icoImg("ico_reset"));
  btnReset.appendChild(el("span", "", "RESET"));
  btnReset.addEventListener("click", () => resetAll());
  [btnAutoSpin, btnFx, btnOutline, btnTheme, btnSpring, btnReset]
    .forEach((b) => trow.appendChild(b));
  sec.appendChild(trow);
  panel.appendChild(sec);
}
shell.appendChild(panel);
rootEl.appendChild(shell);

/* ============================ three.js scene ============================ */
function b64ToBytes(b64) {
  const s = atob(b64);
  const b = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i);
  return b;
}
function buildGeometry(fr, palette) {
  const pos16 = new Int16Array(b64ToBytes(fr.pos).buffer);
  const pos = new Float32Array(pos16.length);
  for (let i = 0; i < pos16.length; i++) pos[i] = pos16[i] / 100.0;
  const reg = b64ToBytes(fr.reg);
  const col = new Uint8Array(fr.nv * 3);
  for (let i = 0; i < fr.nv; i++) {
    const c = palette[reg[i]] || palette[0];
    col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2];
  }
  const colAttr = new THREE.BufferAttribute(col, 3);
  colAttr.normalized = true;
  const nrm16 = new Int16Array(b64ToBytes(fr.nrm).buffer);
  const nrm = new Float32Array(nrm16.length);
  for (let i = 0; i < nrm16.length; i++) nrm[i] = nrm16[i] / 1000.0;
  const idxBytes = b64ToBytes(fr.idx);
  const idx = fr.idx32 ? new Uint32Array(idxBytes.buffer) : new Uint16Array(idxBytes.buffer);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", colAttr);
  g.setAttribute("normal", new THREE.BufferAttribute(nrm, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
renderer.autoClear = false;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1000);

/* Stock backdrop: the stack engine's background shader (vignette +
 * 32px grid + twinkling star specks, themed colors) ported verbatim
 * to a fullscreen pass rendered behind the mesh scene. */
const bgUniforms = {
  uRes: { value: new THREE.Vector2(1, 1) },
  uBg: { value: new THREE.Vector3(223 / 255, 233 / 255, 248 / 255) },
  uGrid: { value: new THREE.Vector3(198 / 255, 214 / 255, 238 / 255) },
  uStar: { value: new THREE.Vector3(107 / 255, 143 / 255, 212 / 255) },
  uStarHi: { value: new THREE.Vector3(1, 1, 1) },
  uTime: { value: 0 },
};
const bgScene = new THREE.Scene();
const bgCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const bgMat = new THREE.ShaderMaterial({
  uniforms: bgUniforms,
  depthTest: false, depthWrite: false,
  vertexShader: [
    "varying vec2 vUv;",
    "void main() {",
    "  vUv = position.xy * 0.5 + 0.5;",
    "  gl_Position = vec4(position.xy, 0.0, 1.0);",
    "}",
  ].join("\n"),
  fragmentShader: [
    "precision mediump float;",
    "varying vec2 vUv;",
    "uniform vec2 uRes;",
    "uniform vec3 uBg;",
    "uniform vec3 uGrid;",
    "uniform vec3 uStar;",
    "uniform vec3 uStarHi;",
    "uniform float uTime;",
    "float hash21(vec2 p) {",
    "  p = fract(p * vec2(123.34, 456.21));",
    "  p += dot(p, p + 45.32);",
    "  return fract(p.x * p.y);",
    "}",
    "void main() {",
    "  vec2 px = vUv * uRes;",
    "  vec2 dv = vUv - vec2(0.5, 0.5);",
    "  float vig = 1.0 - dot(dv, dv) * 0.28;",
    "  vec3 col = uBg * clamp(vig, 0.80, 1.05);",
    "  float stepPx = 32.0;",
    "  vec2 g = mod(px - 0.5 * mod(uRes, stepPx), stepPx);",
    "  if (g.x < 1.2 || g.y < 1.2) {",
    "    col = mix(col, uGrid, 0.65);",
    "  }",
    "  vec2 cell = floor(px / 48.0);",
    "  vec2 loc = fract(px / 48.0) * 48.0;",
    "  float h = hash21(cell);",
    "  if (h > 0.82) {",
    "    vec2 spos = vec2(8.0 + hash21(cell + 7.1) * 32.0, 8.0 + hash21(cell + 19.3) * 32.0);",
    "    float sz = (h > 0.94) ? 2.6 : 1.6;",
    "    if (abs(loc.x - spos.x) < sz * 0.5 && abs(loc.y - spos.y) < sz * 0.5) {",
    "      float tw = 0.65 + 0.35 * sin(uTime * 2.4 + h * 40.0);",
    "      col = mix(col, (h > 0.92) ? uStarHi : uStar, tw);",
    "    }",
    "  }",
    "  gl_FragColor = vec4(col, 1.0);",
    "}",
  ].join("\n"),
});
const bgMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bgMat);
bgMesh.frustumCulled = false;
bgScene.add(bgMesh);

/* G6 lighting: diagonal key + flat ambient fill (unchanged). */
const amb = new THREE.AmbientLight(0xeaf6ff, __AMB_INT__);
scene.add(amb);
const dir = new THREE.DirectionalLight(0xffffff, __DIR_INT__);
dir.position.set(__DIR_POS__);
scene.add(dir);

/* Toon ramps: SILKY/SMOOTH use the G6 wrap ramp (Linear); PIXEL/RETRO
 * use the pre-G6 hard 4-step ramp (Nearest). Wrap itself is a uniform
 * (uWrapAmt), so mode switches never recompile: 0.4 = G6 wrap, 0 = off. */
const wrapTexData = new Uint8Array([__GRAD_RGBA__]);
const wrapTex = new THREE.DataTexture(wrapTexData, __RAMP_N__, 1, THREE.RGBAFormat);
wrapTex.minFilter = THREE.LinearFilter;
wrapTex.magFilter = THREE.LinearFilter;
wrapTex.generateMipmaps = false;
wrapTex.needsUpdate = true;
const hardTexData = new Uint8Array([__HARD_RGBA__]);
const hardTex = new THREE.DataTexture(hardTexData, 4, 1, THREE.RGBAFormat);
hardTex.minFilter = THREE.NearestFilter;
hardTex.magFilter = THREE.NearestFilter;
hardTex.generateMipmaps = false;
hardTex.needsUpdate = true;

function injectWrap(shader, uWrapObj) {
  const anchor = "float dotNL = dot( normal, lightDirection );";
  const chunk = THREE.ShaderChunk.gradientmap_pars_fragment;
  if (!chunk.includes(anchor)) throw new Error("gradientmap anchor missing");
  const modified = "uniform float uWrapAmt;\n" + chunk.replace(anchor, anchor +
    "\n\tdotNL = clamp( ( dotNL + uWrapAmt ) / ( 1.0 + uWrapAmt ), -1.0, 1.0 );");
  shader.uniforms.uWrapAmt = uWrapObj;
  shader.fragmentShader = shader.fragmentShader.replace(
    "#include <gradientmap_pars_fragment>", modified);
}

const faceImg = new Image();
const faceTex = new THREE.Texture(faceImg);
faceImg.onload = () => { faceTex.needsUpdate = true; };
faceImg.src = DATA.faceTex;

const uPose = { value: new THREE.Vector4(1, 1, 1, 0) };
const uWrapSlime = { value: WRAP_AMT };
const uWrapStar = { value: WRAP_AMT };

const slimeMat = new THREE.MeshToonMaterial({ vertexColors: true, gradientMap: wrapTex });
slimeMat.onBeforeCompile = (shader) => {
  injectWrap(shader, uWrapSlime);
  shader.uniforms.uFaceTex = { value: faceTex };
  shader.uniforms.uPose = uPose;
  shader.vertexShader = shader.vertexShader
    .replace("#include <common>",
      "#include <common>\nvarying vec3 vObjPos;\nvarying vec3 vObjNorm;")
    .replace("#include <begin_vertex>",
      "#include <begin_vertex>\nvObjPos = position;\nvObjNorm = normal;");
  shader.fragmentShader = shader.fragmentShader
    .replace("#include <common>",
      "#include <common>\nvarying vec3 vObjPos;\nvarying vec3 vObjNorm;\n" +
      "uniform sampler2D uFaceTex;\nuniform vec4 uPose;")
    .replace("#include <color_fragment>", `#include <color_fragment>
      {
        vec3 pp = vec3(vObjPos.x / uPose.x,
                       vObjPos.z / uPose.y,
                       (vObjPos.y + CENTER_Z_F - uPose.w) / uPose.z);
        vec2 fuv = vec2((pp.x - FACE_X0_F) / FACE_W_F, (pp.z - FACE_Z0_F) / FACE_H_F);
        float inWin = smoothstep(0.0, 0.03, fuv.x) * (1.0 - smoothstep(0.97, 1.0, fuv.x))
                    * smoothstep(0.0, 0.03, fuv.y) * (1.0 - smoothstep(0.97, 1.0, fuv.y));
        float facing = smoothstep(0.10, 0.50, normalize(vObjNorm).z);
        float front = smoothstep(5.0, 8.0, pp.y);
        float m = inWin * facing * front;
        if (m > 0.001) {
          vec4 face = texture2D(uFaceTex, fuv);
          diffuseColor.rgb = mix(diffuseColor.rgb, face.rgb, face.a * m);
        }
      }`);
};
slimeMat.customProgramCacheKey = () => "slime-decal-wrap-g7";

const starMat = new THREE.MeshToonMaterial({ vertexColors: true, gradientMap: wrapTex });
starMat.onBeforeCompile = (shader) => {
  injectWrap(shader, uWrapStar);
};
starMat.customProgramCacheKey = () => "star-wrap-g7";

const outlineMat = new THREE.MeshBasicMaterial({ color: 0x242a58, side: THREE.BackSide });
outlineMat.onBeforeCompile = (shader) => {
  shader.vertexShader = shader.vertexShader.replace("#include <begin_vertex>",
    "#include <begin_vertex>\ntransformed += normal * 0.38;");
};
outlineMat.customProgramCacheKey = () => "outline-g7";

const geos = DATA.frames.map((fr) => buildGeometry(fr, DATA.slimePalette));
const slimeGroup = new THREE.Group();
const slime = new THREE.Mesh(geos[0], slimeMat);
const outline = new THREE.Mesh(geos[0], outlineMat);
slimeGroup.add(slime);
slimeGroup.add(outline);
scene.add(slimeGroup);

const starGeo = buildGeometry(DATA.star, DATA.starPalette);
const star = new THREE.Mesh(starGeo, starMat);
star.scale.setScalar(0.9);
scene.add(star);

let curMesh = -1;
function applyFrame(f) {
  slime.geometry = geos[f];
  outline.geometry = geos[f];
  const p = DATA.frames[f].pose;
  uPose.value.set(p[0], p[1], p[2], p[3]);
  curMesh = f;
}
applyFrame(0);

/* ============================ control logic ============================ */
function impulseSquash(sx, sy) {
  if (!state.springOn) return;
  state.sqX = sx; state.sqY = sy;
  state.sqVx = 0; state.sqVy = 0;
}
function stepSpring(dt) {
  if (!state.springOn) {
    state.sqX = 1.0; state.sqY = 1.0; state.sqVx = 0; state.sqVy = 0;
    return;
  }
  const kSpring = 160, kDamp = 16;
  state.sqVx += ((1 - state.sqX) * kSpring - state.sqVx * kDamp) * dt;
  state.sqVy += ((1 - state.sqY) * kSpring - state.sqVy * kDamp) * dt;
  state.sqX = Math.max(0.76, Math.min(1.26, state.sqX + state.sqVx * dt));
  state.sqY = Math.max(0.76, Math.min(1.26, state.sqY + state.sqVy * dt));
}
function selectMotion(id) {
  const m = MOTIONS.find((x) => x.id === id);
  if (!m) return;
  state.motion = id;
  state.animPhase = 0;
  state.manualFrame = 0;
  state.animPlaying = true;
  const imp = m.impulse || [1.04, 0.96];
  impulseSquash(imp[0], imp[1]);
  timelineUpdate(0);
  syncAll();
}
function stepMotion(d) {
  if (MOTIONS.length <= 1) return;         // stock: no-op for 1 motion
  const idx = Math.max(0, MOTIONS.findIndex((m) => m.id === state.motion));
  selectMotion(MOTIONS[(idx + d + MOTIONS.length) % MOTIONS.length].id);
}
function setFrame(f) {
  const safe = Math.max(0, Math.min(FC - 1, f | 0));
  state.animPlaying = false;
  state.manualFrame = safe;
  state.animPhase = safe;
  timelineUpdate(safe);
  syncAll();
}
function togglePlay() {
  if (state.animPlaying) {
    state.manualFrame = Math.floor(state.animPhase) % FC;
    state.animPlaying = false;
    timelineUpdate(state.manualFrame);
  } else {
    state.animPhase = state.manualFrame;
    state.animPlaying = true;
  }
  syncAll();
}
function setPitchDeg(deg) {
  state.pitchDeg = Math.max(15, Math.min(90, Math.round(deg)));
}
function toggleTopView() {
  state.faceFocus = false;
  state.pitchDeg = state.pitchDeg >= 89 ? DEFAULT_PITCH_DEG : 90;
  syncAll();
}
function toggleFaceFocus() {
  state.faceFocus = !state.faceFocus;
  if (state.faceFocus) {
    state.zoom = 2.25;
    state.pitchDeg = DEFAULT_PITCH_DEG;
  } else {
    state.zoom = PACK_DEFAULTS.zoom;
  }
  syncAll();
}
function setAngleDeg(deg) {
  const d = ((Number(deg) || 0) % 360 + 360) % 360;
  state.angle = (d > 180 ? d - 360 : d) * Math.PI / 180;
  state.autoSpin = false;
  syncAll();
}
function setHoldRotDir(d) {
  state.holdRotDir = d;
  if (d !== 0) { state.autoSpin = false; syncAll(); }
}
function applyOutline() {
  outline.visible = state.outlineOn;
}
function applyFx() {
  // FX layer in this scene = the orbiting star companion (the only
  // non-character animated element); toggling FX shows/hides it.
  star.visible = state.fxOn;
}
function applyRenderMode() {
  const m = state.renderMode;
  const hard = (m === 0 || m === 1);
  slimeMat.gradientMap = hard ? hardTex : wrapTex;
  starMat.gradientMap = hard ? hardTex : wrapTex;
  uWrapSlime.value = hard ? 0.0 : WRAP_AMT;
  uWrapStar.value = hard ? 0.0 : WRAP_AMT;
  // Mode definitions set the outline default (SMOOTH AA ships without
  // the hull); the OUTLINE toggle stays a live master switch after.
  state.outlineOn = (m !== 2);
  applyOutline();
  renderer.setPixelRatio(m === 1 ? 0.5
    : Math.min(window.devicePixelRatio || 1, 2));
  resizeRenderer();
}
function setRenderMode(m) {
  state.renderMode = m | 0;
  applyRenderMode();
  syncAll();
}
function applyTheme() {
  const t = THEMES[state.themeIdx];
  const set = (u, rgb) => u.value.set(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
  set(bgUniforms.uBg, t.bg);
  set(bgUniforms.uGrid, t.grid);
  set(bgUniforms.uStar, t.star);
  set(bgUniforms.uStarHi, t.starHi);
  outlineMat.color.setRGB(t.outline[0] / 255, t.outline[1] / 255, t.outline[2] / 255);
}
function cycleTheme() {
  state.themeIdx = (state.themeIdx + 1) % THEMES.length;
  applyTheme();
  syncAll();
}
function resetAll() {
  state.motion = PACK_DEFAULTS.motion;
  state.animPlaying = true;
  state.animFps = PACK_DEFAULTS.animFps;
  state.animPhase = 0;
  state.manualFrame = 0;
  state.renderMode = PACK_DEFAULTS.renderMode;
  state.angle = PACK_DEFAULTS.angleDeg * Math.PI / 180;
  state.autoSpin = true;
  state.springOn = true;
  state.fxOn = true;
  state.pitchDeg = DEFAULT_PITCH_DEG;
  state.zoom = PACK_DEFAULTS.zoom;
  state.faceFocus = false;
  state.outlineOn = true;
  state.themeIdx = PACK_DEFAULTS.themeIdx;
  applyRenderMode();
  applyTheme();
  applyFx();
  impulseSquash(1.10, 0.90);
  timelineUpdate(0);
  syncAll();
}
function syncAll() {
  panel.querySelectorAll("[data-motion]").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-motion") === state.motion);
  });
  modeBtns.forEach((btn) => {
    btn.classList.toggle("active", Number(btn.getAttribute("data-rmode")) === state.renderMode);
  });
  if (fpsSlider) {
    fpsSlider.sync();
    fpsSlider.actionBtn.textContent = state.animPlaying ? "PLAY" : "STOP";
    fpsSlider.actionBtn.classList.toggle("active", state.animPlaying);
  }
  if (pitchSlider) {
    pitchSlider.sync();
    pitchSlider.actionBtn.classList.toggle("active", state.pitchDeg >= 89);
  }
  if (zoomSlider) {
    zoomSlider.sync();
    zoomSlider.actionBtn.classList.toggle("active", state.faceFocus);
  }
  if (btnAutoSpin) {
    btnAutoSpin.classList.toggle("active", state.autoSpin);
    btnAutoSpin.querySelector("span").textContent = state.autoSpin ? "SPIN: ON" : "SPIN: OFF";
  }
  if (btnFx) {
    btnFx.classList.toggle("active", state.fxOn);
    btnFx.querySelector("span").textContent = state.fxOn ? "FX: ON" : "FX: OFF";
  }
  if (btnOutline) {
    btnOutline.classList.toggle("active", state.outlineOn);
    btnOutline.querySelectorAll("span").forEach((s) => {
      s.textContent = state.outlineOn ? "OUTLINE: ON" : "OUTLINE: OFF";
    });
  }
  if (btnSpring) {
    btnSpring.classList.toggle("active", state.springOn);
    btnSpring.querySelector("span").textContent = state.springOn ? "SPRING: ON" : "SPRING: OFF";
  }
  if (txtTheme) txtTheme.textContent = THEMES[state.themeIdx].name;
}

/* ------------------------- stage interactions ------------------------- */
btnBounce.addEventListener("click", () => impulseSquash(0.84, 1.20));
function bindHold(btn, d) {
  const start = (e) => { e.preventDefault(); setHoldRotDir(d); };
  const stop = () => { if (state.holdRotDir === d) state.holdRotDir = 0; };
  btn.addEventListener("pointerdown", start);
  window.addEventListener("pointerup", stop);
  window.addEventListener("pointercancel", stop);
}
bindHold(btnRotL, -1);
bindHold(btnRotR, 1);

const pointers = new Map();
let pinchDist = 0;
canvas.addEventListener("pointerdown", (e) => {
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 1) {
    state.dragActive = true;
    state.autoSpin = false;               // stock: drag stops the spin
    canvas.classList.add("dragging");
    canvas.setPointerCapture(e.pointerId);
    syncAll();
  } else if (pointers.size === 2) {
    const p = [...pointers.values()];
    pinchDist = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
  }
});
canvas.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId)) return;
  const prev = pointers.get(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 1 && state.dragActive) {
    // Camera azimuth mirrors the stock model angle (angle += dx*0.011
    // on the model == theta -= dx*0.011 for the camera); vertical drag
    // drives the pitch, clamped to the stock 15..90 deg range.
    state.angle -= (e.clientX - prev.x) * 0.011;
    setPitchDeg(state.pitchDeg + (e.clientY - prev.y) * 0.11);
    pitchSlider.sync();
  } else if (pointers.size === 2) {
    const p = [...pointers.values()];
    const d = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
    if (pinchDist > 0 && d > 0) {
      state.zoom = Math.max(0.8, Math.min(3.2, state.zoom * (d / pinchDist)));
      zoomSlider.sync();
    }
    pinchDist = d;
  }
});
const endPointer = (e) => {
  pointers.delete(e.pointerId);
  if (pointers.size === 0) {
    state.dragActive = false;
    canvas.classList.remove("dragging");
  }
};
canvas.addEventListener("pointerup", endPointer);
canvas.addEventListener("pointercancel", endPointer);
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const delta = e.deltaY < 0 ? 0.1 : -0.1;
  state.zoom = Math.max(0.8, Math.min(3.2, Math.round((state.zoom + delta) * 20) / 20));
  zoomSlider.sync();
}, { passive: false });

window.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
  if (e.code === "KeyQ" || e.code === "ArrowLeft") keys.left = true;
  if (e.code === "KeyE" || e.code === "ArrowRight") keys.right = true;
  if (e.key === "[") stepMotion(-1);
  if (e.key === "]") stepMotion(1);
  if (e.key >= "1" && e.key <= "9") {
    const idx = Number(e.key) - 1;
    if (idx < MOTIONS.length) selectMotion(MOTIONS[idx].id);
  }
  if (e.code === "KeyF") toggleFaceFocus();
  if (e.code === "KeyB") impulseSquash(0.84, 1.20);
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  if (e.code === "KeyR") resetAll();
});
window.addEventListener("keyup", (e) => {
  if (e.code === "KeyQ" || e.code === "ArrowLeft") keys.left = false;
  if (e.code === "KeyE" || e.code === "ArrowRight") keys.right = false;
});

function resizeRenderer() {
  const w = Math.max(1, stageWrap.clientWidth);
  const h = Math.max(1, stageWrap.clientHeight);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const db = new THREE.Vector2();
  renderer.getDrawingBufferSize(db);
  bgUniforms.uRes.value.set(db.x, db.y);
}
window.addEventListener("resize", resizeRenderer);

/* ================================ loop ================================ */
let last = performance.now();
let lastF = -1;
function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;

  if (state.holdRotDir !== 0) state.angle -= state.holdRotDir * 1.9 * dt;
  else if (keys.left) state.angle += 1.9 * dt;
  else if (keys.right) state.angle -= 1.9 * dt;
  else if (state.autoSpin && !state.dragActive) state.angle -= state.spinSpeed * dt;

  if (state.animPlaying) state.animPhase += state.animFps * dt;
  const fIdx = Math.floor(state.animPlaying ? state.animPhase : state.manualFrame) % FC;
  const meshF = Math.floor(fIdx / DATA.frameStep) % geos.length;
  if (meshF !== curMesh) applyFrame(meshF);
  if (fIdx !== lastF) { timelineUpdate(fIdx); lastF = fIdx; }

  stepSpring(dt);
  slimeGroup.scale.set(state.sqX, state.sqY, state.sqX);

  const t = now / 1000;
  const a = t * (Math.PI * 2 / 4.0);
  star.position.set(Math.cos(a) * 30, 8 + Math.sin(a * 2) * 3.5, Math.sin(a) * 21);
  star.rotation.y = -a;
  star.rotation.z = Math.sin(t * 2.0) * 0.25;

  const theta = state.angle;
  const phi = (90 - state.pitchDeg) * Math.PI / 180;
  const radius = 145 / state.zoom;
  camera.position.set(
    radius * Math.sin(phi) * Math.sin(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.cos(theta)
  );
  camera.lookAt(0, 0, 0);

  const deg = Math.round((((theta % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2)) * 180 / Math.PI) % 360;
  const fStr = String(fIdx + 1).padStart(2, "0");
  const degStr = String(deg).padStart(3, "0");
  hudChip.textContent = (window.innerWidth <= 860)
    ? "BOUNCE F" + fStr + " " + degStr + "D"
    : "BOUNCE F" + fStr + "/" + FC + " | " + state.animFps + " FPS | " + degStr + " DEG";

  bgUniforms.uTime.value = t;
  renderer.clear();
  renderer.render(bgScene, bgCam);
  renderer.render(scene, camera);
}

applyTheme();
applyRenderMode();
applyFx();
resizeRenderer();
syncAll();
requestAnimationFrame(animate);
</script>
</body>
</html>
"""


def render_html(three_js, data_json):
    with open(THEME_CSS_PATH, "r", encoding="utf-8") as f:
        theme_css = f.read()
    assert "</style" not in theme_css.lower(), "theme CSS contains </style>"
    html = HTML_TEMPLATE.replace("__THEME_CSS__", theme_css)
    html = html.replace("__THREE_JS__", three_js)
    html = html.replace("__DATA_JSON__", data_json)
    html = html.replace("const CENTER_Z = __CENTER_Z__;",
                        f"const CENTER_Z = {CENTER_Z};")
    html = html.replace("__FACE_X0__", str(FACE_X0)).replace("__FACE_W__", str(FACE_W))
    html = html.replace("__FACE_Z0__", str(FACE_Z0)).replace("__FACE_H__", str(FACE_H))
    # GLSL float literals inside the injected shader chunk
    html = html.replace("CENTER_Z_F", f"{CENTER_Z:.1f}")
    html = html.replace("FACE_X0_F", f"{FACE_X0:.1f}").replace("FACE_W_F", f"{FACE_W:.1f}")
    html = html.replace("FACE_Z0_F", f"{FACE_Z0:.1f}").replace("FACE_H_F", f"{FACE_H:.1f}")
    # G6 lighting numbers (single source of truth = the constants above;
    # the numpy strata audit uses the same values).
    html = html.replace("__DIR_POS__", ", ".join(str(int(v)) for v in DIR_POS))
    html = html.replace("__DIR_INT__", repr(DIR_INTENSITY))
    html = html.replace("__AMB_INT__", repr(AMB_INTENSITY))
    html = html.replace("__WRAP__", repr(WRAP))
    html = html.replace("__RAMP_N__", str(RAMP_TEXELS))
    html = html.replace("__GRAD_RGBA__", ", ".join(str(v) for v in ramp_bytes()))
    # Pre-G6 hard 4-step ramp (PIXEL+AA / RETRO PX render modes).
    hard = []
    for v in OLD_GRAD4:
        hard += [v, v, v, 255]
    html = html.replace("__HARD_RGBA__", ", ".join(str(v) for v in hard))
    return html


# ------------------------------------------------- shipped-space verification
def _dec(b64, dtype, scale=1.0):
    a = np.frombuffer(base64.b64decode(b64), dtype=dtype)
    return a.reshape(-1, 3).astype(np.float64) / scale


def _unit(v):
    ln = np.linalg.norm(v, axis=1)
    ln[ln == 0] = 1.0
    return v / ln[:, None]


def _load_payload(html_text):
    i = html_text.index("const DATA = ") + len("const DATA = ")
    j = html_text.index(";\n", i)
    return json.loads(html_text[i:j])


# Cameras for the audits (shipped space, looking at the origin): the
# viewer's initial view (theta=0.6, phi=1.12, radius=100) and a back-top
# view approximating the user's G5 screenshot angle (theta=2.8,
# phi=0.75: behind the slime, elevated).
_AUDIT_CAM = np.array([100.0 * math.sin(1.12) * math.sin(0.6),
                       100.0 * math.cos(1.12),
                       100.0 * math.sin(1.12) * math.cos(0.6)])
_AUDIT_CAM_BACKTOP = np.array([100.0 * math.sin(0.75) * math.sin(2.8),
                               100.0 * math.cos(0.75),
                               100.0 * math.sin(0.75) * math.cos(2.8)])

_LUM = np.array([0.2126, 0.7152, 0.0722])


def _dir_unit(pos3):
    v = np.asarray(pos3, dtype=np.float64)
    return v / np.linalg.norm(v)


def irradiance_old(n):
    """G5 lighting as a scalar irradiance factor per vertex (normals in
    shipped space): near-vertical key through the 4-step Nearest gradient
    map [70, 135, 195, 255], plus the hemisphere term -- a smooth
    function of the normal's y component, i.e. shading by elevation."""
    n = _unit(n)
    dotnl = n @ _dir_unit(OLD_DIR_POS)
    band = np.minimum((np.clip(dotnl * 0.5 + 0.5, 0.0, 1.0) * 4).astype(int), 3)
    grad = np.array(OLD_GRAD4, dtype=np.float64) / 255.0
    sky = np.array(OLD_HEMI_SKY, dtype=np.float64) / 255.0
    gnd = np.array(OLD_HEMI_GROUND, dtype=np.float64) / 255.0
    hemi = (gnd[None, :] + (sky - gnd)[None, :]
            * (0.5 * n[:, 1] + 0.5)[:, None]) * OLD_HEMI_INTENSITY
    return OLD_DIR_INTENSITY * grad[band] + hemi @ _LUM


def irradiance_new(n):
    """G6 lighting: diagonal key through the wrap remap + soft ramp
    (sampled exactly like the shipped LinearFilter texture), plus the
    flat ambient fill (a constant -- no normal dependence)."""
    n = _unit(n)
    dotnl = n @ _dir_unit(DIR_POS)
    wrapped = np.clip((dotnl + WRAP) / (1.0 + WRAP), -1.0, 1.0)
    amb = AMB_INTENSITY * float(np.array(AMB_COLOR) / 255.0 @ _LUM)
    return DIR_INTENSITY * sample_ramp(wrapped * 0.5 + 0.5) + amb


def _strata_metrics(y, irr):
    """How strongly the shading level is a function of world height y
    alone: R^2 of the linear fit irr ~ y, and eta^2 over 10 quantile
    height bins (between-bin share of the total variance). Strata =>
    both near 1 (shade = f(height)); form-following shading => clearly
    lower, because azimuth/facing vary the shade at fixed height."""
    A = np.stack([y, np.ones_like(y)], axis=1)
    coef, *_ = np.linalg.lstsq(A, irr, rcond=None)
    ss_res = float(((irr - A @ coef) ** 2).sum())
    ss_tot = float(((irr - irr.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 0.0, 0.0
    r2 = 1.0 - ss_res / ss_tot
    qs = np.quantile(y, np.linspace(0.0, 1.0, 11))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    b = np.clip(np.searchsorted(qs, y, side="right") - 1, 0, 9)
    ssb = 0.0
    for k in range(10):
        m = b == k
        if m.any():
            ssb += float(m.sum()) * float((irr[m].mean() - irr.mean()) ** 2)
    return r2, ssb / ss_tot


def strata_audit(frames_enc):
    """Old vs new lighting simulated on the shipped positions+normals
    of every frame, from the front and back-top cameras. Returns
    (rows, occ): rows maps (camera, lighting) -> (mean R^2, mean eta^2)
    over frames; occ is the frame-0/front-camera tone occupancy."""
    cams = {"front": _AUDIT_CAM, "back-top": _AUDIT_CAM_BACKTOP}
    acc = {(c, l): [] for c in cams for l in ("old", "new")}
    occ = None
    for enc in frames_enc:
        pos = _dec(enc["pos"], np.int16, 100.0)
        nrm = _unit(_dec(enc["nrm"], np.int16, 1000.0))
        irr = {"old": irradiance_old(nrm), "new": irradiance_new(nrm)}
        for cname, cam in cams.items():
            view = _unit(cam[None, :] - pos)
            front = np.einsum("ij,ij->i", nrm, view) > 0
            for lname in ("old", "new"):
                acc[(cname, lname)].append(
                    _strata_metrics(pos[front, 1], irr[lname][front]))
        if occ is None:
            view = _unit(_AUDIT_CAM[None, :] - pos)
            front = np.einsum("ij,ij->i", nrm, view) > 0
            dotnl = nrm @ _dir_unit(OLD_DIR_POS)
            band = np.minimum(
                (np.clip(dotnl * 0.5 + 0.5, 0.0, 1.0) * 4).astype(int), 3)
            old_dist = np.bincount(band[front], minlength=4) / front.sum()
            wrapped = np.clip(
                (nrm @ _dir_unit(DIR_POS) + WRAP) / (1.0 + WRAP), -1.0, 1.0)
            rv = sample_ramp(wrapped * 0.5 + 0.5)[front]
            occ = {
                "old_band_dist": old_dist,
                "new_shadow": float((rv <= RAMP_SHADOW + 0.03).mean()),
                "new_lit": float((rv >= RAMP_LIT - 0.03).mean()),
            }
            occ["new_trans"] = 1.0 - occ["new_shadow"] - occ["new_lit"]
    rows = {k: (float(np.mean([m[0] for m in v])),
                float(np.mean([m[1] for m in v]))) for k, v in acc.items()}
    return rows, occ


def winding_audit(pos, idx, nrm):
    """Fraction of faces winding inward (face normal vs vertex normals),
    signed volume, and visible-surface completeness from the audit camera:
    of the area whose TRUE normal faces the camera, how much is not culled
    (winding also faces the camera). 1.0 = solid surface; G3 was ~0.5."""
    f = pos[idx]
    fn = _unit(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]))
    gn = _unit(nrm[idx].mean(axis=1))
    inw = np.einsum("ij,ij->i", fn, gn) < 0
    inward = float(inw.mean())
    vol = float(np.einsum("ij,ij->i", f[:, 0],
                          np.cross(f[:, 1], f[:, 2])).sum() / 6.0)
    area = 0.5 * np.linalg.norm(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]),
                                axis=1)
    # Degenerate sliver faces from vertex clustering can disagree with the
    # smooth vertex normals under position quantization; what matters is
    # their AREA share (sub-pixel), so report it alongside the count share.
    inward_area = float(area[inw].sum() / area.sum()) if area.sum() else 0.0
    cen = f.mean(axis=1)
    view = _unit(_AUDIT_CAM[None, :] - cen)
    true_front = np.einsum("ij,ij->i", gn, view) > 0
    wind_front = np.einsum("ij,ij->i", fn, view) > 0
    denom = area[true_front].sum()
    completeness = float(area[true_front & wind_front].sum() / denom) if denom else 0.0
    return inward, vol, completeness, inward_area


def main():
    t0 = time.time()
    with open(THREE_JS_PATH, "r", encoding="utf-8") as f:
        three_js = f.read()
    assert "</script" not in three_js.lower(), "three.js contains </script>"
    # The wrap injection extends this exact line inside the shipped
    # library's gradientmap_pars_fragment chunk (occurs exactly once).
    assert three_js.count(
        "float dotNL = dot( normal, lightDirection );") == 1, \
        "wrap anchor missing/ambiguous in shipped three.min.js"

    face_uri = build_face_texture()
    print(f"face texture: {FACE_TEX_W}x{FACE_TEX_H} PNG data URI "
          f"({len(face_uri)/1024:.0f} KiB)")

    X, Y, Z = make_grid(GRID, GRID, GRID)
    X, Y, Z = X * VOX, Y * VOX, Z * VOX
    poses = [evaluate_demo_pose("bounce", f, 24) for f in range(24)]

    frames = []
    cells = []
    metric_mesh, metric_field = [], []
    frame_stats = []
    total_flips = 0
    for f in range(0, 24, 2):
        v, tm, reg, nrm, mnrm, pose4, cell, nflip, stats = \
            build_slime_frame_g5(X, Y, Z, poses[f])
        frames.append(encode_mesh(v, tm, reg, nrm, pose4))
        metric_field.append(neighbor_angle_metric(tm, nrm))
        metric_mesh.append(neighbor_angle_metric(tm, mnrm))
        cells.append(cell)
        frame_stats.append(stats)
        total_flips += nflip
        print(f"  frame {f:2d}: {frames[-1]['nv']} verts, {frames[-1]['nt']} tris "
              f"(cluster cell {cell}, winding flips {nflip}) "
              f"({time.time() - t0:.1f}s)")
    print(f"normal smoothness, mean angle to neighbor-average normal (deg): "
          f"field normals {float(np.mean(metric_field)):.3f}; mesh normals "
          f"from REPAIRED winding {float(np.mean(metric_mesh)):.3f} "
          f"(G3's unrepaired mesh normals measured 19.558)")
    s0 = frame_stats[0]
    print(f"G5 projection (frame 0, full precision, demo space): |d| "
          f"mean {s0['d_pre'][0]:.5f} -> {s0['d_post'][0]:.5f}, max "
          f"{s0['d_pre'][1]:.4f} -> {s0['d_post'][1]:.5f}; displacement "
          f"from decimated state mean {s0['disp_mean']:.4f} max "
          f"{s0['disp_max']:.4f}; pair-angle {s0['pair_pre']:.3f} -> "
          f"{s0['pair_post']:.3f} deg; face-vs-field {s0['fvf_pre']:.3f} "
          f"-> {s0['fvf_post']:.3f} deg; zmax {s0['zmax_pre']:.2f} -> "
          f"{s0['zmax_post']:.2f} (ahoge tip height, pre-round vs post)")
    print(f"G5 projection (12-frame means): |d| mean "
          f"{float(np.mean([s['d_pre'][0] for s in frame_stats])):.5f} -> "
          f"{float(np.mean([s['d_post'][0] for s in frame_stats])):.5f}; "
          f"pair-angle {float(np.mean([s['pair_pre'] for s in frame_stats])):.3f} "
          f"-> {float(np.mean([s['pair_post'] for s in frame_stats])):.3f} deg; "
          f"face-vs-field {float(np.mean([s['fvf_pre'] for s in frame_stats])):.3f} "
          f"-> {float(np.mean([s['fvf_post'] for s in frame_stats])):.3f} deg")

    sv, st, sreg, snrm, sflip, sstats = build_star_g5()
    enc_star = encode_mesh(sv, st, sreg, snrm)
    print(f"star: {enc_star['nv']} verts, {enc_star['nt']} tris "
          f"(winding flips {sflip})")
    print(f"G5 projection (star, full precision): |d| mean "
          f"{sstats['d_pre'][0]:.5f} -> {sstats['d_post'][0]:.5f}, max "
          f"{sstats['d_pre'][1]:.4f} -> {sstats['d_post'][1]:.5f}; "
          f"displacement mean {sstats['disp_mean']:.4f} max "
          f"{sstats['disp_max']:.4f}; pair-angle {sstats['pair_pre']:.3f} "
          f"-> {sstats['pair_post']:.3f} deg; face-vs-field "
          f"{sstats['fvf_pre']:.3f} -> {sstats['fvf_post']:.3f} deg")
    total_flips += sflip

    data = {
        "frameStep": 2,
        "slimePalette": SLIME_PALETTE,
        "starPalette": STAR_PALETTE,
        "faceTex": face_uri,
        "frames": frames,
        "star": enc_star,
    }
    html = render_html(three_js, json.dumps(data, separators=(",", ":")))

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    size = os.path.getsize(OUT_HTML)
    print(f"Wrote {OUT_HTML}: {size} bytes ({size/1048576:.2f} MiB) "
          f"in {time.time() - t0:.1f}s (budget {SIZE_BUDGET/1048576:.0f} MiB)")

    # ------------------------------------------------------- verification
    rest = html.replace(three_js, "")
    assert "http://" not in rest and "https://" not in rest, "external URL found"
    assert 'src="http' not in rest and "src='http" not in rest
    png = base64.b64decode(face_uri.split(",", 1)[1])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pw = int.from_bytes(png[16:20], "big")
    ph = int.from_bytes(png[20:24], "big")
    print(f"verify: face PNG is {pw}x{ph}; no external URLs outside three.js blob")

    # body vertex-color distribution (frame 0, palette-expanded)
    reg0 = np.frombuffer(base64.b64decode(frames[0]["reg"]), dtype=np.uint8)
    pal = np.array(SLIME_PALETTE, dtype=np.float64)
    cols = pal[reg0]
    uniq = np.unique(cols, axis=0)
    print(f"verify: frame0 unique vertex colors = {uniq.tolist()}")
    assert len(uniq) == 2 and tuple(uniq[0]) == AQUA and tuple(uniq[1]) == BELLY
    body = cols[reg0 == 0]
    print(f"verify: body (aqua) verts {len(body)}, mean color {body.mean(axis=0)}, "
          f"belly verts {(reg0 == 1).sum()}")
    assert "uFaceTex" in html and "uPose" in html and '"pose"' in html
    print("verify: decal uniforms + per-frame pose data embedded")
    assert "app-shell" in rest and "stageCanvas" in rest \
        and "RENDER STYLE" in rest and "SILKY 3D" in rest
    print("verify: stock UI chrome embedded (app-shell, stageCanvas, mode grid)")
    assert ".computeVertexNormals(" not in rest, "viewer must not recompute normals"

    # ---- all remaining checks run on the DECODED shipped payload, in
    # shipped three.js space (what the viewer actually consumes).
    ship = _load_payload(html)

    # (c1) unit lengths
    len_lo, len_hi = float("inf"), 0.0
    for enc in ship["frames"] + [ship["star"]]:
        n = _dec(enc["nrm"], np.int16, 1000.0)
        ln = np.linalg.norm(n, axis=1)
        len_lo = min(len_lo, float(ln.min()))
        len_hi = max(len_hi, float(ln.max()))
    print(f"verify: decoded normal lengths in [{len_lo:.4f}, {len_hi:.4f}]")
    assert 0.99 < len_lo and len_hi < 1.01

    # (a) shipped normals vs the SDF gradient recomputed in shipped
    # coordinates: decode positions, invert the position map
    # (demo = [tx, tz, ty + CENTER_Z]), evaluate the field gradient there,
    # permute it with the SAME [x, z, y] map the positions took, compare.
    g_min, g_mean, g_n = float("inf"), 0.0, 0
    for k, enc in enumerate(ship["frames"]):
        p = _dec(enc["pos"], np.int16, 100.0)
        n = _unit(_dec(enc["nrm"], np.int16, 1000.0))
        demo = np.stack([p[:, 0], p[:, 2], p[:, 1] + CENTER_Z], axis=1)
        gd = spirit_gradient_normals(demo, poses[2 * k])
        gt = gd[:, [0, 2, 1]]
        dots = np.einsum("ij,ij->i", n, gt)
        g_min = min(g_min, float(dots.min()))
        g_mean += float(dots.sum())
        g_n += len(dots)
    sp = _dec(ship["star"]["pos"], np.int16, 100.0)
    sn = _unit(_dec(ship["star"]["nrm"], np.int16, 1000.0))
    sdemo = np.stack([sp[:, 0], sp[:, 2], sp[:, 1] + 16.0], axis=1)
    sgt = star_gradient_normals(sdemo)[:, [0, 2, 1]]
    sdots = np.einsum("ij,ij->i", sn, sgt)
    g_min = min(g_min, float(sdots.min()))
    g_mean += float(sdots.sum())
    g_n += len(sdots)
    print(f"verify: shipped normals vs gradient-in-shipped-space: "
          f"mean dot {g_mean / g_n:.4f}, min dot {g_min:.4f} over {g_n} verts")
    assert g_min > 0.99

    # (c2) winding audit on the shipped payload: no inward faces, positive
    # volume, complete visible surface from the canonical camera.
    for k, enc in enumerate(ship["frames"]):
        p = _dec(enc["pos"], np.int16, 100.0)
        n = _dec(enc["nrm"], np.int16, 1000.0)
        idx = np.frombuffer(base64.b64decode(enc["idx"]),
                            dtype=np.uint32 if enc["idx32"] else np.uint16)
        idx = idx.reshape(-1, 3).astype(np.int64)
        inward, vol, comp, inw_area = winding_audit(p, idx, n)
        # completeness < 1.0 only by silhouette-edge faces, whose flat
        # winding normal and smooth vertex normals straddle the view plane;
        # residual "inward" faces are degenerate slivers (area ~0).
        assert inw_area < 1e-4 and vol > 0 and comp > 0.98, \
            (k, inward, inw_area, vol, comp)
        if k == 0:
            print(f"verify: frame0 winding: inward faces {inward:.5f} "
                  f"(area share {inw_area:.2e}), signed volume {vol:+.1f}, "
                  f"visible-surface completeness {comp:.3f}")
    sp_idx = np.frombuffer(base64.b64decode(ship["star"]["idx"]),
                           dtype=np.uint16).reshape(-1, 3).astype(np.int64)
    s_inward, s_vol, s_comp, s_inw_area = winding_audit(sp, sp_idx, sn)
    print(f"verify: star winding: inward faces {s_inward:.5f} "
          f"(area share {s_inw_area:.2e}), signed volume {s_vol:+.1f}, "
          f"completeness {s_comp:.3f}")
    # star completeness runs lower than the slime's: it is a coarse 32^3
    # cross with large flat faces, so more limb-edge area straddles the
    # view plane between flat and smooth normals. Inward = 0 is the fix.
    assert s_inw_area < 1e-4 and s_vol > 0 and s_comp > 0.95

    # (b) G6 strata audit: simulate BOTH lightings on the shipped
    # positions+normals and measure how much of the shading world
    # height alone explains, from the front and back-top cameras.
    assert "HemisphereLight" not in rest and "AmbientLight" in rest
    assert "injectWrap" in rest and "gradientmap_pars_fragment" in rest
    print("verify: viewer has AmbientLight + injectWrap, no HemisphereLight")
    rows, occ = strata_audit(ship["frames"])
    for cname in ("front", "back-top"):
        for lname in ("old", "new"):
            r2, eta2 = rows[(cname, lname)]
            print(f"verify: strata audit [{cname}] {lname} lighting: "
                  f"R^2(shade~height) {r2:.3f}, eta^2 {eta2:.3f}")
    print(f"verify: occupancy (frame0, front camera): G5 4-band dist "
          f"{np.round(occ['old_band_dist'], 3).tolist()}; G6 ramp zones "
          f"shadow {occ['new_shadow']:.3f} / transition "
          f"{occ['new_trans']:.3f} / lit {occ['new_lit']:.3f}")
    for cname in ("front", "back-top"):
        assert rows[(cname, "new")][1] < rows[(cname, "old")][1] - 0.05, \
            (cname, rows)
    assert 0.02 < occ["new_shadow"] and occ["new_lit"] > 0.10

    # (b2) byte-identity of every DATA payload field vs G6: G7 changes
    # only the page chrome (stock UI), so geometry/color/normal/decal
    # data must match G6 exactly, field by field. Cross-check against a
    # previous build's file: runs only when that reference is present
    # (the variants/ files are not part of this repository).
    if not os.path.exists(G6_HTML):
        print("SKIP G6 byte-identity check: reference "
              "variants/slime-G6-threejs.html not present")
    else:
        with open(G6_HTML, encoding="utf-8") as fh:
            d6 = _load_payload(fh.read())
        same = len(d6["frames"]) == len(ship["frames"])
        for a, b in zip(d6["frames"], ship["frames"]):
            for k in ("pos", "nrm", "reg", "idx", "pose", "nv", "nt", "idx32"):
                same &= (a[k] == b[k])
        for k in ("pos", "nrm", "reg", "idx", "nv", "nt", "idx32"):
            same &= (d6["star"][k] == ship["star"][k])
        same &= (d6["faceTex"] == ship["faceTex"]
                 and d6["slimePalette"] == ship["slimePalette"]
                 and d6["starPalette"] == ship["starPalette"]
                 and d6["frameStep"] == ship["frameStep"])
        print(f"verify: DATA payload byte-identical to G6 (frames "
              f"pos/nrm/reg/idx/pose, star, faceTex, palettes): {bool(same)}")
        assert same

    # ---- G4 vs G5 on the DECODED shipped payloads: same topology
    # (vertex counts and triangle multisets must match -- projection
    # moves vertices, it never reorders or retriangulates), then the
    # terracing comparison: on-surface error |d| (positions inverted
    # back to demo space, field re-evaluated), unsigned adjacent-face
    # angle, face-vs-field angle, and per-vertex displacement.
    # Cross-check against a previous build's file: runs only when that
    # reference is present (the variants/ files are not part of this
    # repository).
    if not os.path.exists(G4_HTML):
        print("SKIP G4 comparison checks: reference "
              "variants/slime-G4-threejs.html not present")
    else:
        with open(G4_HTML, encoding="utf-8") as fh:
            d4 = _load_payload(fh.read())
        assert len(d4["frames"]) == len(frames)

        def _idx_of(enc):
            idx = np.frombuffer(base64.b64decode(enc["idx"]),
                                dtype=np.uint32 if enc["idx32"] else np.uint16)
            return idx.reshape(-1, 3).astype(np.int64)

        def _tri_multiset(enc):
            return sorted(map(tuple, np.sort(_idx_of(enc), axis=1).tolist()))

        topo_frames = all(
            a["nv"] == b["nv"] and a["nt"] == b["nt"]
            and _tri_multiset(a) == _tri_multiset(b)
            for a, b in zip(d4["frames"], frames))
        topo_star = (d4["star"]["nv"] == enc_star["nv"]
                     and d4["star"]["nt"] == enc_star["nt"]
                     and _tri_multiset(d4["star"]) == _tri_multiset(enc_star))
        print(f"verify: topology identical to G4 (frames: {topo_frames}, "
              f"star: {topo_star}); triangles re-wound by the repair: "
              f"{total_flips}")
        assert topo_frames and topo_star

        rows = []
        for k in range(len(frames)):
            pose_k = poses[2 * k]
            field_k = lambda x, y, z: spirit_sdf(x, y, z, pose_k)  # noqa: E731
            row = {}
            for tag, enc in (("g4", d4["frames"][k]), ("g5", frames[k])):
                p = _dec(enc["pos"], np.int16, 100.0)
                demo = np.stack([p[:, 0], p[:, 2], p[:, 1] + CENTER_Z], axis=1)
                idx = _idx_of(enc)
                dm, dx = field_abs_stats(demo, field_k)
                row[tag] = {
                    "d_mean": dm, "d_max": dx,
                    "pair": float(_pair_face_angles(demo, idx).mean()),
                    "fvf": face_vs_field_angle(demo, idx, field_k),
                    "demo": demo,
                }
            disp = np.linalg.norm(row["g5"]["demo"] - row["g4"]["demo"], axis=1)
            row["disp_mean"] = float(disp.mean())
            row["disp_max"] = float(disp.max())
            rows.append(row)
        r0 = rows[0]
        print(f"compare frame0 (shipped payloads): |d| G4 mean "
              f"{r0['g4']['d_mean']:.5f} max {r0['g4']['d_max']:.4f} -> G5 mean "
              f"{r0['g5']['d_mean']:.5f} max {r0['g5']['d_max']:.4f}; "
              f"pair-angle {r0['g4']['pair']:.3f} -> {r0['g5']['pair']:.3f} deg; "
              f"face-vs-field {r0['g4']['fvf']:.3f} -> {r0['g5']['fvf']:.3f} "
              f"deg; displacement G4->G5 mean {r0['disp_mean']:.4f} max "
              f"{r0['disp_max']:.4f}")
        for key in ("d_mean", "d_max", "pair", "fvf"):
            m4 = float(np.mean([r["g4"][key] for r in rows]))
            m5 = float(np.mean([r["g5"][key] for r in rows]))
            print(f"compare 12-frame mean {key}: G4 {m4:.5f} -> G5 {m5:.5f}")
        print(f"compare displacement G4->G5 (12 frames): mean of means "
              f"{float(np.mean([r['disp_mean'] for r in rows])):.4f}, worst max "
              f"{float(np.max([r['disp_max'] for r in rows])):.4f}")
        # The G5 gate: shipped vertices sit on the analytic surface.
        assert all(r["g5"]["d_mean"] < 0.02 for r in rows)
        assert float(np.mean([r["g5"]["d_max"] for r in rows])) < \
            float(np.mean([r["g4"]["d_max"] for r in rows]))

        sfield = lambda x, y, z: star_sdf(x, y, z, 0)  # noqa: E731
        star_rows = {}
        for tag, enc in (("g4", d4["star"]), ("g5", enc_star)):
            p = _dec(enc["pos"], np.int16, 100.0)
            demo = np.stack([p[:, 0], p[:, 2], p[:, 1] + 16.0], axis=1)
            idx = _idx_of(enc)
            dm, dx = field_abs_stats(demo, sfield)
            star_rows[tag] = {"d_mean": dm, "d_max": dx,
                              "pair": float(_pair_face_angles(demo, idx).mean()),
                              "fvf": face_vs_field_angle(demo, idx, sfield),
                              "demo": demo}
        sdisp = np.linalg.norm(star_rows["g5"]["demo"] - star_rows["g4"]["demo"],
                               axis=1)
        print(f"compare star (shipped payloads): |d| G4 mean "
              f"{star_rows['g4']['d_mean']:.5f} max {star_rows['g4']['d_max']:.4f} "
              f"-> G5 mean {star_rows['g5']['d_mean']:.5f} max "
              f"{star_rows['g5']['d_max']:.4f}; pair-angle "
              f"{star_rows['g4']['pair']:.3f} -> {star_rows['g5']['pair']:.3f} "
              f"deg; face-vs-field {star_rows['g4']['fvf']:.3f} -> "
              f"{star_rows['g5']['fvf']:.3f} deg; displacement mean "
              f"{float(sdisp.mean()):.4f} max {float(sdisp.max()):.4f}")
        assert star_rows["g5"]["d_mean"] < 0.02
        print(f"compare: G4 HTML {os.path.getsize(G4_HTML)} bytes -> "
              f"G7 HTML {size} bytes; frame0 {frames[0]['nv']} verts / "
              f"{frames[0]['nt']} tris (unchanged)")


if __name__ == "__main__":
    main()
