"""Variant H: F's smoothed pipeline + de-banded colors + lighter package.

Two changes vs variant F (build_demo_smooth256.py, untouched):

1) DE-BANDING. F shades with shade_cel_3step: ndotl is hard-thresholded at
   0.52 / 0.06 / -0.38 into 4 flat colors, so the curved body shows discrete
   strata rings wherever ndotl crosses a threshold. Here the SAME anchor
   colors and the SAME threshold centers are kept, but each hard step is
   replaced by a smoothstep ramp of half-width SOFT_W (0.22) in ndotl space,
   evaluated in float32 and quantized to uint8 only at the end. The cel
   character (broad near-flat plateaus) survives; the strata edges become a
   continuous gradient. Region safety: aqua and belly are shaded as two
   separate full-volume fields and composited with F's sharp belly mask
   AFTER shading, so no color bleeds across the body-region boundary, and
   the face paint (eyes / highlights / blush / mouth) is applied last with
   F's sharp surface masks, exactly as in F -- it never enters any smoothing.

2) LIGHTER PACKAGE. F bakes 256 slices at spacing 1.0. The engine places
   slice i at screen offset i * spacing * cos(pitch) (stack vertex shader),
   so baking only the EVEN layers of the same 256^3 field (128 slices of
   256x256) with pack spacing = 2.0 lands every kept slice at exactly the
   position F's even slices had; total stack height and framing are
   unchanged (engine needH = sliceCount * max(1, spacing) is identical).
   Honest note: at spacing > 1.32 the engine selects its 1x VBO, i.e. the
   2x sub-slice doubling is OFF -- the gaps are covered instead by the huge
   per-slice screen overlap (a slice image projects ~87 FBO px tall at the
   default 20 deg pitch vs a 2 px z-step). What can thin out is only
   z-features < 0.5 world units (ear/ahoge tips lose at most one layer).
   All z-space pack numbers are halved so world placement matches F:
   faceFocusZ 144 -> 72, companion orbit z 128/-32/20 -> 64/-16/10,
   orbit-trail z 144/-32/20 -> 72/-16/10, splash slashZ 112 -> 56,
   originZ 104 -> 52. XY / FBO-pixel numbers (fboVirt 560x880, pivot 224,
   orbit rx/ry 136/112) are unchanged because the FBO size is unchanged.
   The star companion is subsampled the same way (32 -> 16 slices): the
   engine applies the character's spacing to companions too, so halving
   its slice count keeps its projected height identical to F.

   WEIGHT REALITY CHECK (measured on the first H build): the slice halving
   works geometrically (atlas pixels exactly halved), but a continuous
   gradient is far more expensive to store than F's flat cel colors --
   one H atlas was 197 KB lossless vs F's 46 KB for twice the pixels.
   The character atlases are therefore saved LOSSY WebP (quality 90,
   method 4; alpha stays bit-exact, opaque-RGB mean error ~1.0 LSB,
   p99 ~12 at hard paint edges). Net result (measured): HTML 5.35 MB /
   ap3d 3.93 MB vs F's 3.17 MB / 2.29 MB -- i.e. the de-banding costs
   ~+70% package weight even after the slice halving; vs the lossless
   de-banded build (13.05 MB HTML) the lossy pass removes ~59%.
   De-banding and flat-cel file sizes cannot both be had; this variant
   chooses de-banding.

Field pipeline (geometry) is F's, unchanged: analytic SDF on the 128^3
grid, Gaussian sigma=1.25, trilinear upsample x2 to 256^3, re-threshold,
normals re-derived on the fine grid. renderMode 3 (SILKY 3D),
assetLinear=true, dotSize 1.0 -- same as F.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "engine"))
sys.path.insert(0, HERE)

from build_demo import sculpt_star_companion, evaluate_demo_pose
from toolkit.gpu_sdf import (
    xp, USE_GPU, to_numpy, make_grid,
    smin, sd_ellipsoid, sd_capsule,
    compute_normals,
)
from toolkit.pixel_fx_kit import generate_standard_fx_and_ui_pack, add_pixel_outline, draw_pattern, save_rgba
from toolkit.qa_validator import validate_motion_poses, assert_no_border_clipping, save_tiled_webp
from bundle_standalone import bundle_standalone

BLUR_SIGMA = 1.25  # in 128-grid voxels (= 0.625 world units) -- same as F
SOFT_W = 0.22      # smoothstep half-width around each cel threshold (ndotl units)
LIGHT = (-0.22, 0.38, 0.90)  # same light dir as shade_cel_3step's default
WEBP_Q = 90        # lossy quality for character atlases (see docstring)


def save_tiled_webp_lossy(vol, path, cols=8, quality=WEBP_Q):
    """save_tiled_webp with lossy WebP (alpha remains bit-exact in PIL's
    encoder); needed because continuous gradients explode lossless sizes."""
    import math
    from PIL import Image
    H, L, W, _ = vol.shape
    rows = int(math.ceil(H / cols))
    atlas = np.zeros((rows * L, cols * W, 4), dtype=np.uint8)
    for z in range(H):
        r, c = z // cols, z % cols
        atlas[r * L:(r + 1) * L, c * W:(c + 1) * W] = vol[z]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(atlas, "RGBA").save(path, format="WEBP", quality=quality, method=4)


def _smoothstep(edge0, edge1, x):
    t = xp.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def shade_cel_soft(base_rgb, hi_rgb, sh_rgb, deep_rgb, nx, ny, nz):
    """Continuous version of shade_cel_3step.

    Same anchors and threshold centers (hi > 0.52, sh < 0.06, deep < -0.38);
    each step becomes a smoothstep ramp of half-width SOFT_W, blended in
    float32 in deep -> shadow -> base -> highlight order, quantized last.
    """
    lx, ly, lz = LIGHT
    t = (nx * lx + ny * ly + nz * lz).astype(xp.float32)
    deep = xp.asarray(deep_rgb, dtype=xp.float32)
    sh = xp.asarray(sh_rgb, dtype=xp.float32)
    base = xp.asarray(base_rgb, dtype=xp.float32)
    hi = xp.asarray(hi_rgb, dtype=xp.float32)
    out = xp.broadcast_to(deep, (*t.shape, 3)).copy()
    for thr, col in ((-0.38, sh), (0.06, base), (0.52, hi)):
        w = _smoothstep(thr - SOFT_W, thr + SOFT_W, t)[..., None]
        out = out * (1.0 - w) + col * w
    return xp.clip(xp.rint(out), 0, 255).astype(xp.uint8)


def sculpt_spirit_frame_light(Xf, Yf, Zf, pose):
    """Xf/Yf/Zf: fine-grid (256^3) coordinates in world units (make_grid(256) * 0.25).

    Returns the full 256^3 RGBA volume (caller subsamples even layers).
    """
    sx, sy, sz = pose.sx, pose.stretch_y, pose.squash_z
    rb = pose.root_bob

    # --- 1) Analytic SDF field natively on the 128^3 grid (F's sampling) ---
    X, Y, Z = make_grid(128, 128, 128)
    X, Y, Z = X * 0.5, Y * 0.5, Z * 0.5
    PX = X / sx
    PY = Y / sy
    PZ = (Z - rb) / sz
    d_body = sd_ellipsoid(PX, PY, PZ, 0.0, 0.0, 22.0, 16.5, 15.5, 18.0)
    d_ear = sd_capsule(xp.abs(PX), PY, PZ, 12.0, -2.0, 32.0, 22.5, -5.0, 37.0 + pose.ear_flap_l, 4.2, 2.2)
    d_ahoge = sd_capsule(PX, PY, PZ, 0.0, 0.0, 39.0, -6.0, 4.0, 46.0 + pose.ear_flap_l * 0.4, 2.2, 1.0)
    d_all = smin(smin(d_body, d_ear, 2.2), d_ahoge, 1.4)
    del PX, PY, PZ, d_body, d_ear, d_ahoge

    # --- 2) Post-process: Gaussian-smooth the finished 128^3 field, upsample x2 ---
    d_smooth = gaussian_filter(to_numpy(d_all), sigma=BLUR_SIGMA, mode="nearest")
    del d_all
    d_up = zoom(d_smooth, 2.0, order=1)  # (256,256,256), values stay in world units
    del d_smooth
    assert d_up.shape == (256, 256, 256), d_up.shape
    d_up = xp.asarray(d_up)

    # --- 3) Re-derive occupancy / normals / shading / face paint on the 256^3 grid ---
    PXf = Xf / sx
    PYf = Yf / sy
    PZf = (Zf - rb) / sz
    occ = (d_up <= 0.0) & (Zf >= 0.45)

    nx, ny, nz = compute_normals(d_up)
    # De-banded shading: continuous smoothstep ramps instead of 3 hard steps.
    col_aqua = shade_cel_soft([86, 178, 250], [162, 228, 255], [52, 126, 214], [32, 82, 168], nx, ny, nz)
    col_belly = shade_cel_soft([244, 250, 255], [255, 255, 255], [210, 224, 246], [176, 196, 228], nx, ny, nz)
    del nx, ny, nz

    # Sharp region composite AFTER shading (no bleed across the belly boundary).
    colors = col_aqua.copy()
    del col_aqua
    is_belly = occ & (PYf > 3.5) & (PZf < 20.5) & (xp.abs(PXf) < 11.5)
    colors[is_belly] = col_belly[is_belly]
    del col_belly, is_belly

    # Face paint LAST, sharp masks, never smoothed (identical to F).
    surf = occ & (d_up >= -2.5) & (PYf > 8.0)
    del d_up
    for cx in (-6.0, 6.0):
        eye_bg = surf & (((PXf - cx) / 3.2) ** 2 + ((PZf - 24.5) / 3.8) ** 2 <= 1.0)
        eye_hi = surf & (((PXf - (cx - 0.9)) / 1.1) ** 2 + ((PZf - 26.2) / 1.1) ** 2 <= 1.0)
        blush = surf & (((PXf - cx * 1.45) / 2.4) ** 2 + ((PZf - 19.8) / 1.4) ** 2 <= 1.0)
        colors[eye_bg] = xp.asarray([28, 36, 86], dtype=xp.uint8)
        colors[eye_hi] = xp.asarray([255, 255, 255], dtype=xp.uint8)
        colors[blush] = xp.asarray([255, 152, 178], dtype=xp.uint8)
    mouth = surf & ((PXf / 2.2) ** 2 + ((PZf - 19.5) / 1.6) ** 2 <= 1.0)
    colors[mouth] = xp.asarray([238, 118, 140], dtype=xp.uint8)

    vol = xp.zeros((*occ.shape, 4), dtype=xp.uint8)
    vol[occ, :3] = colors[occ]
    vol[occ, 3] = 255
    return to_numpy(vol)


def main():
    t0 = time.time()
    build_dir = os.path.join(HERE, "build_light")
    assets_dir = os.path.join(build_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    print(f"Building variant H (F pipeline + soft cel ramp w={SOFT_W}, 128 slices @ spacing 2.0, backend={'CuPy GPU' if USE_GPU else 'NumPy CPU'})...", flush=True)

    # Fine grid in world units (same world as the 64-grid, quarter-voxel spacing).
    Xf, Yf, Zf = make_grid(256, 256, 256)
    Xf, Yf, Zf = Xf * 0.25, Yf * 0.25, Zf * 0.25
    assets = generate_standard_fx_and_ui_pack(assets_dir)

    pal = {"C": (118, 224, 255, 255), "W": (248, 253, 255, 255), "Y": (252, 214, 92, 255), "P": (255, 142, 178, 255)}
    for name, rows in [
        ("ico_bounce", ["................","......YY........",".....YYYY.......","....CCCCCCC.....","..CCWWWWWWWCC...","..CWWCWWWCWWC...","..CWWWWWWWWWC...","..CWWPWWWPWWC...","...CCWWWWWCC....",".....CCCCC......","................","................","................","................","................","................"]),
        ("ico_splash", ["................",".....CCCCC......","...CCWWWWWCC....","..CWWCCCCCWWC...",".CWC.......CWC..",".CC.........CC..","....CC...CC.....","...CWWC.CWWC....","....CCWWWWWCC....","................","................","................","................","................","................"]),
    ]:
        save_rgba(add_pixel_outline(draw_pattern(rows, pal), (14, 18, 38, 255)), os.path.join(assets_dir, f"{name}.png"))
        assets[name] = {"file": f"assets/{name}.png"}

    # Bake 2 motions x 24 frames: full 256^3 field per frame, keep even layers
    # -> 128 slices of 256x256 (8x16 atlas), pack spacing 2.0 restores height.
    for m in ("bounce", "splash"):
        tm = time.time()
        poses = [evaluate_demo_pose(m, f, 24) for f in range(24)]
        warns = validate_motion_poses(poses)
        if warns:
            print("  QA warnings:", warns)
        for p in poses:
            vol = sculpt_spirit_frame_light(Xf, Yf, Zf, p)
            vol = vol[::2]  # even layers of the 256^3 field -> (128,256,256,4)
            assert vol.shape == (128, 256, 256, 4), vol.shape
            assert_no_border_clipping(vol, label=f"{m}_{p.frame_idx}", check_bottom=False)
            rel = f"assets/anim_{m}_{p.frame_idx:02d}.webp"
            save_tiled_webp_lossy(vol, os.path.join(build_dir, rel), cols=8)
            assets[f"anim_{m}_{p.frame_idx:02d}"] = {"file": rel}
            print(f"    {m} frame {p.frame_idx:02d} done ({time.time() - t0:.0f}s elapsed)", flush=True)
        print(f"  baked {m} in {time.time() - tm:.1f}s ({time.time() - t0:.1f}s elapsed)", flush=True)

    # Star companion: same treatment, 32 -> 16 slices (spacing is global).
    for f in range(24):
        vol = sculpt_star_companion(f, 24)[::2]
        assert vol.shape == (16, 32, 32, 4), vol.shape
        assert_no_border_clipping(vol, label=f"star_{f}", check_bottom=True)
        rel = f"assets/star_{f:02d}.webp"
        save_tiled_webp(vol, os.path.join(build_dir, rel), cols=8)
        assets[f"star_{f:02d}"] = {"file": rel}
    print(f"  baked star companion ({time.time() - t0:.1f}s elapsed)", flush=True)

    pack = {
        "schema_version": "1.1",
        "title": "SLIME SPIRIT 3D DEMO",
        "subtitle": "VARIANT H: DE-BANDED GRADIENT + 128 SLICES x2 SPACING (LIGHT)",
        "badge_icon": "ico_brand",
        "defaults": {
            "motion": "bounce",
            "animFps": 24,
            "renderMode": 3,
            "dotSize": 1.0,
            "pitchDeg": 20,
            "squash": 0.35,
            "spacing": 2.0,
            "zoom": 1.45,
            "themeIdx": 1,
            "angleDeg": 22.0,
        },
        "character": {
            "sliceW": 256, "sliceH": 256, "sliceCount": 128,
            "atlasCols": 8, "atlasRows": 16, "yOffset": 0.0,
            "fboVirtW": 560, "fboVirtH": 880, "fboPivotYOffset": 224, "faceFocusZ": 72.0,
        },
        "companions": {
            "mini_star": {
                "sliceW": 32, "sliceH": 32, "sliceCount": 16,
                "atlasCols": 8, "atlasRows": 2, "frames": 24,
                "keyPrefix": "star_", "fboVirtW": 72, "fboVirtH": 96,
                "scale": 0.85, "outlineRgb": [0.42, 0.28, 0.08],
            }
        },
        "motions": [
            {
                "id": "bounce",
                "label": "BOUNCE",
                "subtitle": "3D STAR PET",
                "category": "COMPANION",
                "accent": "companion",
                "icon": "ico_bounce",
                "frames": 24,
                "keyPrefix": "anim_bounce_",
                "impulse": [0.92, 1.10],
                "companion": {"id": "mini_star", "rx": 136, "ry": 112, "zBase": 64, "zSin1": -16, "zSin2": 10, "faceTangent": False},
                "fx": [
                    {"type": "orbit_trail", "count": 14, "rx": 136, "ry": 112, "zBase": 72, "zSin1": -16, "zSin2": 10, "primarySprite": "fxHeart", "secondarySprite": "fxBubble"},
                ],
            },
            {
                "id": "splash",
                "label": "SPLASH",
                "subtitle": "AQUA SLASH",
                "category": "ATTACK",
                "accent": "attack",
                "icon": "ico_splash",
                "frames": 24,
                "keyPrefix": "anim_splash_",
                "impulse": [0.88, 1.14],
                "springBeats": [{"frame": 6, "sx": 0.88, "sy": 1.14}],
                "fx": [
                    {"type": "forward_slash_fan", "startFrame": 5, "durationFrames": 14, "count": 48, "slashZ": 56, "originZ": 52, "slashW": 224, "slashH": 120},
                    {"type": "floor_wave_rings", "count": 1, "startFrame": 5.5, "durationFrames": 11, "forwardOffset": 56, "startScale": 0.45, "expandScale": 0.85},
                ],
            },
        ],
        "assets": assets,
        "assetLinear": {k: True for k in assets},
    }

    pack_path = os.path.join(build_dir, "motion_pack.json")
    with open(pack_path, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2)

    out_html = os.path.join(build_dir, "slime-chibi-standalone.html")
    out_ap3d = os.path.join(build_dir, "slime-chibi.ap3d")
    bundle_standalone(
        html_path=os.path.join(REPO_ROOT, "engine", "viewer_shell.html"),
        js_path=os.path.join(REPO_ROOT, "engine", "anime_stack_engine.js"),
        pack_path=pack_path,
        assets_root=build_dir,
        out_path=out_html,
        ap3d_out=out_ap3d,
    )
    print(f"Variant H build complete in {time.time() - t0:.2f}s -> {out_html} & {out_ap3d}", flush=True)


if __name__ == "__main__":
    main()
