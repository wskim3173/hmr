from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import flags
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel


# -----------------------------
# Flags
# -----------------------------
flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string('json_path', None, 'If specified, uses OpenPose output to crop the image.')
flags.DEFINE_string('out_dir', 'ppt_outputs', 'Directory to save PPT images.')
flags.DEFINE_integer('render_size', 1200, 'Target output resolution for saved single-panel images.')
flags.DEFINE_string('mesh_view', 'front', 'front, side, or top')

# Shape / pose demos.
flags.DEFINE_integer('shape_coeff_idx', 0, 'Which beta coefficient to vary.')
flags.DEFINE_string('shape_deltas', '-3,0,3', 'Comma-separated beta deltas.')
flags.DEFINE_float('pose_strength', 1.0, 'Strength multiplier for pose demo rotations.')

# Camera demos.
flags.DEFINE_float('camera_t_delta', 0.30, 'Translation delta for tx.')
flags.DEFINE_float('camera_s_small', 0.75, 'Scale multiplier for smaller demo.')
flags.DEFINE_float('camera_s_large', 1.25, 'Scale multiplier for larger demo.')
flags.DEFINE_string('camera_R_angles', '-60,0,60', 'Comma-separated view rotation angles.')

# Visualization look.
flags.DEFINE_float('mesh_elev', 18.0, 'Elevation angle for matplotlib 3D mesh view.')
flags.DEFINE_float('mesh_azim', -65.0, 'Azimuth angle for matplotlib 3D mesh view.')
flags.DEFINE_float('crop_margin_ratio', 0.10, 'Margin ratio for auto-cropping rendered mesh images.')


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.0
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center, config.img_size)
    crop = 2 * ((crop / 255.0) - 0.5)
    return crop, proc_param, img


def split_theta(theta_1d):
    theta_1d = np.asarray(theta_1d).reshape(-1)
    cam = theta_1d[:3].copy()
    pose = theta_1d[3:75].copy()
    shape = theta_1d[75:85].copy()
    return cam, pose, shape


def save_theta_info(out_dir, cam, pose, shape):
    path = os.path.join(out_dir, '00_theta_values.txt')
    with open(path, 'w') as f:
        f.write('theta layout = [camera(3), pose(72), shape(10)]\n\n')
        f.write('camera [s, tx, ty]\n')
        f.write(np.array2string(cam, precision=5, suppress_small=True))
        f.write('\n\nshape beta, 10D\n')
        f.write(np.array2string(shape, precision=5, suppress_small=True))
        f.write('\n\npose theta, 72D axis-angle, 24 joints x 3\n')
        f.write(np.array2string(pose.reshape(24, 3), precision=5, suppress_small=True))
        f.write('\n')
    print('Saved theta values to %s' % path)


def get_original_render_data(img, proc_param, joints, verts, cam):
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])
    return cam_for_render, vert_shifted, joints_orig


def load_faces(face_path):
    faces = np.load(face_path)
    if isinstance(faces, np.ndarray) and faces.dtype == object:
        faces = faces.item()
    if isinstance(faces, dict):
        for key in ['f', 'faces', 'faces_generic']:
            if key in faces:
                faces = faces[key]
                break
    return np.asarray(faces, dtype=np.int32)


def orient_points(points):
    p = np.asarray(points)
    view = getattr(config, 'mesh_view', 'front')
    if view == 'front':
        p = np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1)
    elif view == 'side':
        p = np.stack([p[:, 2], p[:, 0], -p[:, 1]], axis=1)
    elif view == 'top':
        p = np.stack([p[:, 0], p[:, 1], p[:, 2]], axis=1)
    else:
        raise ValueError('--mesh_view must be one of: front, side, top')
    return p


def normalize_points(points):
    p = orient_points(points)
    p = p - p.mean(axis=0, keepdims=True)
    ext = p.max(axis=0) - p.min(axis=0)
    scale = np.max(ext)
    if scale > 0:
        p = p / scale
    return p


def normalize_points_like_reference(points, reference_points):
    """Normalize points using the same center/scale as a reference mesh.

    This makes 3D keypoints and the wireframe use the same coordinate frame,
    scale, and viewing direction.
    """
    p = orient_points(points)
    ref = orient_points(reference_points)

    ref_center = ref.mean(axis=0, keepdims=True)
    ref_ext = ref.max(axis=0) - ref.min(axis=0)
    ref_scale = np.max(ref_ext)

    p = p - ref_center
    if ref_scale > 0:
        p = p / ref_scale
    return p


def apply_common_3d_view(ax):
    """Use the same camera/view settings for wireframe and 3D keypoints."""
    lim = 0.72
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    try:
        ax.set_box_aspect((1.0, 1.0, 1.35))
        ax.set_proj_type('persp')
    except Exception:
        pass

    ax.view_init(elev=0, azim=-90)
    ax.set_axis_off()


def add_surface_mesh_to_axis(ax, verts, faces, title=None):
    verts_n = normalize_points(verts)

    # plot_trisurf gives clearer 3D shading than a flat Poly3DCollection.
    ax.plot_trisurf(
        verts_n[:, 0], verts_n[:, 1], verts_n[:, 2],
        triangles=faces,
        color=(0.74, 0.80, 0.88),
        shade=True,
        linewidth=0.03,
        edgecolor=(0.63, 0.69, 0.77),
        antialiased=True,
    )

    xyz_min = verts_n.min(axis=0)
    xyz_max = verts_n.max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    radius = 0.6 * np.max(xyz_max - xyz_min)
    if radius <= 0:
        radius = 0.5

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1.4))
    except Exception:
        pass
    try:
        ax.set_proj_type('persp')
    except Exception:
        pass

    ax.view_init(elev=float(config.mesh_elev), azim=float(config.mesh_azim))
    if title is not None:
        ax.set_title(title, fontsize=12, pad=10)
    ax.set_axis_off()


def add_wireframe_to_axis(ax, verts, faces, title=None):
    verts_n = normalize_points(verts)

    edges = []
    for tri in faces:
        edges.append([verts_n[tri[0]], verts_n[tri[1]]])
        edges.append([verts_n[tri[1]], verts_n[tri[2]]])
        edges.append([verts_n[tri[2]], verts_n[tri[0]]])

    lc = Line3DCollection(edges, linewidths=0.08, colors=(0.45, 0.52, 0.63, 1.0))
    ax.add_collection3d(lc)

    lim = 0.72
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    try:
        ax.set_box_aspect((1, 1, 1.35))
    except Exception:
        pass
    try:
        ax.set_proj_type('persp')
    except Exception:
        pass
    ax.view_init(elev=float(config.mesh_elev), azim=float(config.mesh_azim))
    if title is not None:
        ax.set_title(title, fontsize=14, pad=10)
    ax.set_axis_off()


def save_big_wireframe(verts, faces, out_dir):
    """Save a full-body wireframe from a front-facing 3D view.

    The previous 2D-projection version could be over-cropped and show only the
    middle of the mesh. This version uses normalized 3D coordinates and fixed
    axis limits, so the full body always stays in frame.
    """
    out_path = os.path.join(out_dir, '03_big_mesh_wireframe.png')

    verts_n = normalize_points(verts)

    # Build unique edges from triangular faces.
    edge_set = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edge_set.add(tuple(sorted((a, b))))
        edge_set.add(tuple(sorted((b, c))))
        edge_set.add(tuple(sorted((c, a))))

    edges = [[verts_n[i], verts_n[j]] for i, j in edge_set]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    lc = Line3DCollection(
        edges,
        linewidths=0.075,
        colors=(0.45, 0.52, 0.63, 0.55),
    )
    ax.add_collection3d(lc)

    apply_common_3d_view(ax)
    ax.set_title('Predicted SMPL wireframe mesh', fontsize=15, pad=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=320, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print('Saved %s' % out_path)


# ---------- image cropping / cleanup for renderer outputs ----------
def _foreground_mask(img, threshold=245):
    if img.ndim == 2:
        gray = img
    else:
        gray = img[..., :3].astype(np.float32).mean(axis=2)
    return gray < threshold


def crop_images_to_union_bbox(images, margin_ratio=0.10, threshold=245):
    ys = []
    xs = []
    h = images[0].shape[0]
    w = images[0].shape[1]
    for im in images:
        mask = _foreground_mask(im, threshold=threshold)
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        ys.extend(coords[:, 0].tolist())
        xs.extend(coords[:, 1].tolist())

    if not xs or not ys:
        return images

    y0, y1 = min(ys), max(ys)
    x0, x1 = min(xs), max(xs)
    margin_y = int((y1 - y0 + 1) * margin_ratio)
    margin_x = int((x1 - x0 + 1) * margin_ratio)
    y0 = max(0, y0 - margin_y)
    y1 = min(h, y1 + margin_y + 1)
    x0 = max(0, x0 - margin_x)
    x1 = min(w, x1 + margin_x + 1)

    return [im[y0:y1, x0:x1] for im in images]


def save_image_grid(images, titles, out_path, suptitle=None, ncols=None, figsize=None):
    images = crop_images_to_union_bbox(images, margin_ratio=float(config.crop_margin_ratio))

    n = len(images)
    if ncols is None:
        ncols = n
    nrows = int(np.ceil(float(n) / float(ncols)))
    if figsize is None:
        figsize = (4.2 * ncols, 4.2 * nrows)

    plt.figure(figsize=figsize)
    for i, (im, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(nrows, ncols, i)
        plt.imshow(im)
        plt.title(title, fontsize=12)
        plt.axis('off')
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print('Saved %s' % out_path)


def save_camera_summary_tight_grid(images, titles, out_path, suptitle=None):
    """Save the 3x3 camera summary with smaller gaps between panels."""
    images = crop_images_to_union_bbox(images, margin_ratio=float(config.crop_margin_ratio))

    ncols = 3
    nrows = 3

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6, 6),
        gridspec_kw={
            'wspace': -0.24,
            'hspace': 0.24,
            'left': 0.02,
            'right': 0.98,
            'top': 0.94,
            'bottom': 0.02,
        }
    )

    for ax, im, title in zip(axes.ravel(), images, titles):
        ax.imshow(im)
        ax.set_title(title, fontsize=11, pad=3)
        ax.axis('off')

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14, y=0.985)

    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print('Saved %s' % out_path)


def scale_camera_for_highres(cam, base_img_shape, target_long_side=None):
    """Scale cam [f, cx, cy]-style values when rendering the same mesh at higher resolution."""
    if target_long_side is None:
        target_long_side = int(config.render_size)

    base_h, base_w = int(base_img_shape[0]), int(base_img_shape[1])
    long_side = max(base_h, base_w)
    scale = float(target_long_side) / float(long_side)

    target_h = max(1, int(round(base_h * scale)))
    target_w = max(1, int(round(base_w * scale)))

    cam_scaled = np.copy(cam)
    cam_scaled[0] *= scale
    cam_scaled[1] *= scale
    cam_scaled[2] *= scale
    return cam_scaled, (target_h, target_w)


def render_hmr_highres(renderer, verts, cam, base_img_shape, crop=True):
    """Render a mesh with the HMR renderer at high resolution, then crop around the body.

    This keeps the same visual style as the original HMR renderer, while avoiding
    the pixelated look that happens when a small native render is enlarged.
    """
    cam_scaled, img_size = scale_camera_for_highres(
        cam, base_img_shape, target_long_side=int(config.render_size)
    )
    im = renderer(verts, cam=cam_scaled, img_size=img_size)
    if crop:
        im = crop_images_to_union_bbox([im], margin_ratio=float(config.crop_margin_ratio))[0]
    return im


def render_shifted_mesh_highres(renderer, vert_shifted, cam_for_render, base_img_shape, crop=False):
    """Render vertices already transformed to original-image coordinates at high resolution."""
    cam_scaled, img_size = scale_camera_for_highres(
        cam_for_render, base_img_shape, target_long_side=int(config.render_size)
    )
    im = renderer(vert_shifted, cam=cam_scaled, img_size=img_size)
    if crop:
        im = crop_images_to_union_bbox([im], margin_ratio=float(config.crop_margin_ratio))[0]
    return im


def lock_mesh_scale_to_reference(variant_verts, reference_verts):
    """Rescale a variant mesh so its overall body size matches the reference mesh.

    This is useful for shape comparison figures: changing beta should mainly show
    body proportions / thickness, not a bigger or smaller person in the frame.
    """
    v = np.asarray(variant_verts).copy()
    r = np.asarray(reference_verts)

    # Use the SMPL root area as an anchor. In practice the first vertex/joint-space
    # origin is stable enough, but centering by the bbox center is more robust for visuals.
    v_center = 0.5 * (v.min(axis=0) + v.max(axis=0))
    r_center = 0.5 * (r.min(axis=0) + r.max(axis=0))

    v_extent = np.max(v.max(axis=0) - v.min(axis=0))
    r_extent = np.max(r.max(axis=0) - r.min(axis=0))
    if v_extent > 1e-8 and r_extent > 1e-8:
        scale = float(r_extent) / float(v_extent)
        v = (v - v_center) * scale + r_center
    return v


def transform_variant_to_original_frame(img, proc_param, variant_verts, base_joints_2d, base_cam):
    """Use HMR's original-image transform for a modified SMPL mesh.

    The modified shape/pose vertices come from SMPL in crop/model coordinates.
    This converts them to the same original-image coordinate frame used for
    the normal HMR result, so the mesh is not lost or over-zoomed.
    """
    cam_for_render, vert_shifted, _ = vis_util.get_original(
        proc_param, variant_verts, base_cam, base_joints_2d, img_size=img.shape[:2]
    )
    return cam_for_render, vert_shifted


def save_rendered_mesh_grid(renderer, img, proc_param, verts_list, base_joints_2d, base_cam,
                            titles, out_path, suptitle):
    """Save shape/pose variants using SMPLRenderer in the original-image coordinate frame."""
    shifted_items = []
    for v in verts_list:
        cam_for_render, vert_shifted = transform_variant_to_original_frame(
            img, proc_param, v, base_joints_2d, base_cam
        )
        shifted_items.append((cam_for_render, vert_shifted))

    images = [
        render_shifted_mesh_highres(
            renderer, vert_shifted, cam_for_render, img.shape, crop=False
        )
        for cam_for_render, vert_shifted in shifted_items
    ]

    # Shared crop keeps all variants on a comparable scale while removing empty background.
    images = crop_images_to_union_bbox(images, margin_ratio=max(float(config.crop_margin_ratio), 0.20))

    plt.figure(figsize=(4.4 * len(images), 5.0))
    for i, (im, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, len(images), i)
        plt.imshow(im)
        plt.title(title, fontsize=12)
        plt.axis('off')
    plt.suptitle(suptitle, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print('Saved %s' % out_path)


# ---------- standard HMR renderer outputs ----------
def save_input_keypoint_overlay(img, joints_orig, vert_shifted, cam_for_render, renderer, out_dir):
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(vert_shifted, cam=cam_for_render, img=img, do_alpha=True)

    out_path = os.path.join(out_dir, '01_input_keypoint_mesh_overlay.png')
    plt.figure(figsize=(15, 5.3))
    for i, (im, title) in enumerate(zip([img, skel_img, rend_img_overlay],
                                         ['Input image', '2D keypoints', '3D mesh overlay']), start=1):
        plt.subplot(1, 3, i)
        plt.imshow(im)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close()
    print('Saved %s' % out_path)


def save_big_mesh_only(vert_shifted, cam_for_render, renderer, out_dir, base_img_shape):
    """Save a high-resolution non-overlay mesh with the same HMR renderer style."""
    mesh_img = render_hmr_highres(
        renderer, vert_shifted, cam_for_render, base_img_shape, crop=True
    )

    out_path = os.path.join(out_dir, '02_big_mesh_only.png')
    plt.figure(figsize=(8, 8))
    plt.imshow(mesh_img)
    plt.title('Predicted 3D mesh', fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=320, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print('Saved %s' % out_path)


def save_3d_keypoints(joints3d, reference_verts, out_dir):
    out_path = os.path.join(out_dir, '04_3d_keypoints_points_only.png')

    # Important: normalize keypoints with the same reference mesh used by wireframe.
    # If keypoints are independently centered/scaled, the side view will not match.
    pts = normalize_points_like_reference(joints3d, reference_verts)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=48, depthshade=True)

    apply_common_3d_view(ax)
    ax.set_title('Predicted 3D keypoints only')
    plt.tight_layout()
    plt.savefig(out_path, dpi=320, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print('Saved %s' % out_path)


# ---------- SMPL forward for custom shape/pose ----------
def make_smpl_forward_ops(model):
    if not hasattr(model, 'smpl'):
        raise RuntimeError('RunModel does not expose model.smpl. Check src/RunModel.py.')

    shape_ph = tf.placeholder(tf.float32, shape=[1, 10], name='ppt_shape_beta')
    pose_ph = tf.placeholder(tf.float32, shape=[1, 72], name='ppt_pose_theta')

    smpl_out = model.smpl(shape_ph, pose_ph, get_skin=True)
    verts_op = smpl_out[0]
    joints_op = smpl_out[1]
    return shape_ph, pose_ph, verts_op, joints_op


def smpl_forward(sess, shape_ph, pose_ph, verts_op, joints_op, shape, pose):
    shape_batch = np.asarray(shape, dtype=np.float32).reshape(1, 10)
    pose_batch = np.asarray(pose, dtype=np.float32).reshape(1, 72)
    verts, joints = sess.run([verts_op, joints_op], feed_dict={shape_ph: shape_batch, pose_ph: pose_batch})
    return verts[0], joints[0]


def save_surface_mesh_comparison_grid(verts_list, faces, titles, out_path, suptitle):
    n = len(verts_list)
    fig = plt.figure(figsize=(4.2 * n, 5.0))
    for i, (verts, title) in enumerate(zip(verts_list, titles), start=1):
        ax = fig.add_subplot(1, n, i, projection='3d')
        add_surface_mesh_to_axis(ax, verts, faces, title=title)
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print('Saved %s' % out_path)


def save_shape_comparison(sess, smpl_ops, pred_shape, pred_pose, pred_verts, base_joints_2d, base_cam, img, proc_param, faces, renderer, out_dir):
    shape_ph, pose_ph, verts_op, joints_op = smpl_ops
    deltas = parse_float_list(config.shape_deltas)
    base_shape = pred_shape.copy()
    base_pose = pred_pose.copy()

    idx = int(config.shape_coeff_idx)
    if idx < 0 or idx >= 10:
        raise ValueError('--shape_coeff_idx must be between 0 and 9')

    verts_list = []
    titles = []
    base_val = base_shape[idx]
    for delta in deltas:
        shape = base_shape.copy()
        shape[idx] = base_val + delta
        verts, _ = smpl_forward(sess, shape_ph, pose_ph, verts_op, joints_op, shape, base_pose)
        verts = lock_mesh_scale_to_reference(verts, pred_verts)
        verts_list.append(verts)
        titles.append('shape beta[%d] %+0.1f\nactual %.2f' % (idx, delta, shape[idx]))

    out_path = os.path.join(out_dir, '05_shape_mesh_variation.png')
    save_rendered_mesh_grid(
        renderer, img, proc_param, verts_list, base_joints_2d, base_cam,
        titles, out_path, 'Shape variation: same pose, different body shape'
    )


def build_pose_variants(pred_pose):
    strength = float(config.pose_strength)
    base = pred_pose.copy()
    variants = [('predicted pose', base.copy())]

    p = base.copy()
    p[16 * 3 + 2] += 0.9 * strength
    p[17 * 3 + 2] -= 0.9 * strength
    p[18 * 3 + 2] += 0.5 * strength
    p[19 * 3 + 2] -= 0.5 * strength
    variants.append(('arm pose changed', p))

    p = base.copy()
    p[1 * 3 + 0] += 0.40 * strength
    p[2 * 3 + 0] -= 0.40 * strength
    p[4 * 3 + 0] += 0.70 * strength
    p[5 * 3 + 0] += 0.70 * strength
    variants.append(('leg pose changed', p))
    return variants


def save_pose_comparison(sess, smpl_ops, pred_shape, pred_pose, base_joints_2d, base_cam, img, proc_param, faces, renderer, out_dir):
    shape_ph, pose_ph, verts_op, joints_op = smpl_ops
    shape = pred_shape.copy()

    verts_list = []
    titles = []
    for title, pose in build_pose_variants(pred_pose):
        verts, _ = smpl_forward(sess, shape_ph, pose_ph, verts_op, joints_op, shape, pose)
        verts_list.append(verts)
        titles.append(title)

    out_path = os.path.join(out_dir, '06_pose_mesh_variation.png')
    save_rendered_mesh_grid(
        renderer, img, proc_param, verts_list, base_joints_2d, base_cam,
        titles, out_path, 'Pose variation: same body shape, different joint rotations'
    )


# ---------- Camera R / t / s visualization ----------
def render_mesh_with_camera(renderer, vert_shifted, cam, img_size=None):
    if img_size is None:
        img_size = tuple(vert_shifted.shape[:2]) if hasattr(vert_shifted, 'shape') and len(vert_shifted.shape) >= 2 else None
    # Native-ish size helps keep the mesh large. Cropping is applied later.
    return renderer(vert_shifted, cam=cam, img_size=img_size)


def save_camera_R_demo(vert_shifted, cam_for_render, renderer, out_dir, base_img_shape):
    angles = parse_float_list(config.camera_R_angles)
    cam_scaled, img_size = scale_camera_for_highres(
        cam_for_render, base_img_shape, target_long_side=int(config.render_size)
    )

    images = []
    titles = []
    for angle in angles:
        if abs(angle) < 1e-6:
            im = renderer(vert_shifted, cam=cam_scaled, img_size=img_size)
            title = 'R: original view'
        else:
            im = renderer.rotated(vert_shifted, angle, cam=cam_scaled, img_size=img_size)
            title = 'R: rotate %+g deg' % angle
        images.append(im)
        titles.append(title)

    out_path = os.path.join(out_dir, '07_camera_R_view_rotation.png')
    save_image_grid(images, titles, out_path, suptitle='Camera R: changes the viewing direction')


def save_camera_t_demo(vert_shifted, cam_for_render, renderer, out_dir, base_img_shape):
    d = float(config.camera_t_delta)
    cam_scaled, img_size = scale_camera_for_highres(
        cam_for_render, base_img_shape, target_long_side=int(config.render_size)
    )

    cam_left = np.copy(cam_scaled); cam_left[1] -= d * img_size[1]
    cam_orig = np.copy(cam_scaled)
    cam_right = np.copy(cam_scaled); cam_right[1] += d * img_size[1]

    images = [renderer(vert_shifted, cam=c, img_size=img_size) for c in [cam_left, cam_orig, cam_right]]
    titles = ['t: move left', 't: original', 't: move right']

    out_path = os.path.join(out_dir, '08_camera_t_translation.png')
    save_image_grid(images, titles, out_path, suptitle='Camera t: shifts the 2D projection')


def save_camera_s_demo(vert_shifted, cam_for_render, renderer, out_dir, base_img_shape):
    small = float(config.camera_s_small)
    large = float(config.camera_s_large)
    cam_scaled, img_size = scale_camera_for_highres(
        cam_for_render, base_img_shape, target_long_side=int(config.render_size)
    )

    cam_small = np.copy(cam_scaled); cam_small[0] *= small
    cam_orig = np.copy(cam_scaled)
    cam_large = np.copy(cam_scaled); cam_large[0] *= large

    images = [renderer(vert_shifted, cam=c, img_size=img_size) for c in [cam_small, cam_orig, cam_large]]
    titles = ['s: %.2fx smaller' % small, 's: original', 's: %.2fx larger' % large]

    out_path = os.path.join(out_dir, '09_camera_s_scale.png')
    save_image_grid(images, titles, out_path, suptitle='Camera s: changes apparent size')


def save_camera_summary(vert_shifted, cam_for_render, renderer, out_dir, base_img_shape):
    d = float(config.camera_t_delta)
    small = float(config.camera_s_small)
    large = float(config.camera_s_large)

    cam_scaled, img_size = scale_camera_for_highres(
        cam_for_render, base_img_shape, target_long_side=int(config.render_size)
    )

    cam_t_left = np.copy(cam_scaled); cam_t_left[1] -= d * img_size[1]
    cam_t_right = np.copy(cam_scaled); cam_t_right[1] += d * img_size[1]
    cam_s_small = np.copy(cam_scaled); cam_s_small[0] *= small
    cam_s_large = np.copy(cam_scaled); cam_s_large[0] *= large

    images = [
        renderer.rotated(vert_shifted, -60, cam=cam_scaled, img_size=img_size),
        renderer(vert_shifted, cam=cam_scaled, img_size=img_size),
        renderer.rotated(vert_shifted, 60, cam=cam_scaled, img_size=img_size),
        renderer(vert_shifted, cam=cam_t_left, img_size=img_size),
        renderer(vert_shifted, cam=cam_scaled, img_size=img_size),
        renderer(vert_shifted, cam=cam_t_right, img_size=img_size),
        renderer(vert_shifted, cam=cam_s_small, img_size=img_size),
        renderer(vert_shifted, cam=cam_scaled, img_size=img_size),
        renderer(vert_shifted, cam=cam_s_large, img_size=img_size),
    ]
    titles = [
        'R: left view', 'R: original', 'R: right view',
        't: left', 't: original', 't: right',
        's: smaller', 's: original', 's: larger',
    ]

    out_path = os.path.join(out_dir, '10_camera_R_t_s_summary.png')
    save_camera_summary_tight_grid(
        images, titles, out_path, suptitle='Camera parameters: R, t, s'
    )


def save_camera_demos(vert_shifted, cam_for_render, renderer, out_dir, base_img_shape):
    save_camera_R_demo(vert_shifted, cam_for_render, renderer, out_dir, base_img_shape)
    save_camera_t_demo(vert_shifted, cam_for_render, renderer, out_dir, base_img_shape)
    save_camera_s_demo(vert_shifted, cam_for_render, renderer, out_dir, base_img_shape)
    save_camera_summary(vert_shifted, cam_for_render, renderer, out_dir, base_img_shape)


def main(img_path, json_path=None):
    ensure_dir(config.out_dir)

    sess = tf.Session()
    model = RunModel(config, sess=sess)

    input_img, proc_param, img = preprocess_image(img_path, json_path)
    input_img = np.expand_dims(input_img, 0)

    joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)

    cam, pred_pose, pred_shape = split_theta(theta[0])
    save_theta_info(config.out_dir, cam, pred_pose, pred_shape)

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
    faces = load_faces(config.smpl_face_path)

    cam_for_render, vert_shifted, joints_orig = get_original_render_data(
        img, proc_param, joints[0], verts[0], cams[0])

    save_input_keypoint_overlay(img, joints_orig, vert_shifted, cam_for_render, renderer, config.out_dir)
    save_big_mesh_only(vert_shifted, cam_for_render, renderer, config.out_dir, img.shape)
    save_big_wireframe(verts[0], faces, config.out_dir)
    save_3d_keypoints(joints3d[0], verts[0], config.out_dir)

    smpl_ops = make_smpl_forward_ops(model)
    save_shape_comparison(sess, smpl_ops, pred_shape, pred_pose, verts[0], joints[0], cams[0], img, proc_param, faces, renderer, config.out_dir)
    save_pose_comparison(sess, smpl_ops, pred_shape, pred_pose, joints[0], cams[0], img, proc_param, faces, renderer, config.out_dir)

    save_camera_demos(vert_shifted, cam_for_render, renderer, config.out_dir, img.shape)

    print('\nDone. PPT images saved under: %s' % os.path.abspath(config.out_dir))


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1
    main(config.img_path, config.json_path)
