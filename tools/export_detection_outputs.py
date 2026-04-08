#!/usr/bin/env python
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from utils.util_point_cloud import Object3D

Z_OFFSET = 0.7
LINES_3D = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]
BEV_LINE_ORDER = [(0, 1), (0, 2), (1, 3), (2, 3)]


def trim_white(img, thr=245, pad=10):
    if img is None:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = gray < thr
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x1 = max(0, xs.min() - pad)
    x2 = min(img.shape[1], xs.max() + pad + 1)
    y1 = max(0, ys.min() - pad)
    y2 = min(img.shape[0], ys.max() + pad + 1)
    return img[y1:y2, x1:x2]


def parse_label_header(path_label: Path):
    line = path_label.read_text().splitlines()[0]
    rdr_idx, ldr_idx, camf_idx, _, _ = line.split(',')[0].split('=')[1].split('_')
    return rdr_idx, ldr_idx, camf_idx


def load_calib_xy(path_calib: Path):
    lines = path_calib.read_text().splitlines()
    vals = list(map(float, lines[1].split(',')))
    return vals[1], vals[2], Z_OFFSET


def load_lidar_pcd(path_lidar: Path, calib_xyz):
    with open(path_lidar, 'r') as f:
        lines = [line.rstrip('\n') for line in f][13:]
    pc = np.array([point.split() for point in lines], dtype=float).reshape(-1, 9)[:, :4]
    pc = pc[pc[:, 0] > 0.01]
    pc[:, 0] += calib_xyz[0]
    pc[:, 1] += calib_xyz[1]
    pc[:, 2] += calib_xyz[2]
    return pc


def render_sector(points_xyv, mode: str, r_max=80.0, az_max_deg=52.0, canvas=(560, 560)):
    h, w = canvas
    margin = 16
    radius = min(w // 2 - margin, h - 2 * margin)
    cx = w // 2
    cy = h - margin
    az_max = np.deg2rad(az_max_deg)

    x = points_xyv[:, 0]
    y = points_xyv[:, 1]
    v = points_xyv[:, 2] if points_xyv.shape[1] > 2 else np.ones_like(x)

    r = np.sqrt(x ** 2 + y ** 2)
    az = np.arctan2(-y, x)
    mask = (x > 0) & (r > 0.5) & (r < r_max) & (np.abs(az) < az_max)
    x, y, r, az, v = x[mask], y[mask], r[mask], az[mask], v[mask]

    rr = (r / r_max) * radius
    px = np.clip(np.round(cx + rr * np.sin(az)).astype(np.int32), 0, w - 1)
    py = np.clip(np.round(cy - rr * np.cos(az)).astype(np.int32), 0, h - 1)

    if mode == 'lidar':
        canvas_img = np.full((h, w, 3), 255, dtype=np.uint8)
        canvas_img[py, px] = 0
        kernel = np.ones((2, 2), np.uint8)
        canvas_img = cv2.erode(canvas_img, kernel, iterations=1)
    else:
        acc = np.zeros((h, w), dtype=np.float32)
        np.maximum.at(acc, (py, px), v.astype(np.float32))
        acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=2.0, sigmaY=2.0)
        if np.max(acc) > 0:
            acc = np.log1p(acc)
            acc = acc / (np.max(acc) + 1e-6)
        cmap = cm.get_cmap('turbo')
        canvas_img = (cmap(acc)[..., :3] * 255).astype(np.uint8)
        canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)

    sector_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(sector_mask, (cx, cy), (radius, radius), 0, 270 - az_max_deg, 270 + az_max_deg, 255, thickness=-1)
    canvas_img[sector_mask == 0] = 255
    return canvas_img, (cx, cy, radius, r_max, az_max)


def xy_to_sector_px(x, y, meta):
    cx, cy, radius, r_max, az_max = meta
    r = np.sqrt(x * x + y * y)
    az = np.arctan2(-y, x)
    rr = (r / r_max) * radius
    px = int(np.round(cx + rr * np.sin(az)))
    py = int(np.round(cy - rr * np.cos(az)))
    return px, py


def draw_bev_boxes(img, boxes, color, meta, thickness=2):
    out = img.copy()
    for box in boxes:
        obj = Object3D(box['x'], box['y'], box['z'], box['l'], box['w'], box['h'], box['theta'])
        pts = [obj.corners[0, :], obj.corners[2, :], obj.corners[4, :], obj.corners[6, :]]
        pt_list = [xy_to_sector_px(pt[0], pt[1], meta) for pt in pts]
        for i, j in BEV_LINE_ORDER:
            cv2.line(out, pt_list[i], pt_list[j], color, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def render_lidar_bev(lidar_pc, x_range=(0.0, 80.0), y_range=(-40.0, 40.0), canvas=(560, 560)):
    h, w = canvas
    x_min, x_max = x_range
    y_min, y_max = y_range

    pts = lidar_pc[:, :3]
    mask = (
        (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
        (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max) &
        (pts[:, 2] >= -3.0) & (pts[:, 2] <= 6.0)
    )
    pts = pts[mask]

    canvas_img = np.full((h, w, 3), 255, dtype=np.uint8)
    if len(pts) == 0:
        return canvas_img, (x_min, x_max, y_min, y_max, h, w)

    x = pts[:, 0]
    y = pts[:, 1]
    px = np.clip(np.round((y - y_min) / (y_max - y_min) * (w - 1)).astype(np.int32), 0, w - 1)
    py = np.clip(np.round((1.0 - (x - x_min) / (x_max - x_min)) * (h - 1)).astype(np.int32), 0, h - 1)

    occ = np.zeros((h, w), dtype=np.uint8)
    occ[py, px] = 255
    occ = cv2.dilate(occ, np.ones((2, 2), np.uint8), iterations=1)
    canvas_img[occ > 0] = (70, 70, 70)
    return canvas_img, (x_min, x_max, y_min, y_max, h, w)


def xy_to_bev_px(x, y, meta):
    x_min, x_max, y_min, y_max, h, w = meta
    px = int(np.round((y - y_min) / (y_max - y_min) * (w - 1)))
    py = int(np.round((1.0 - (x - x_min) / (x_max - x_min)) * (h - 1)))
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    return px, py


def draw_bev_boxes_cartesian(img, boxes, color, meta, thickness=2):
    out = img.copy()
    for box in boxes:
        obj = Object3D(box['x'], box['y'], box['z'], box['l'], box['w'], box['h'], box['theta'])
        pts = [obj.corners[0, :], obj.corners[2, :], obj.corners[4, :], obj.corners[6, :]]
        pt_list = [xy_to_bev_px(pt[0], pt[1], meta) for pt in pts]
        for i, j in BEV_LINE_ORDER:
            cv2.line(out, pt_list[i], pt_list[j], color, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def parse_result_boxes(path_txt: Path, has_score: bool):
    boxes = []
    if not path_txt.exists():
        return boxes
    for line in path_txt.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls_name = parts[0]
        nums = list(map(float, parts[1:]))
        if has_score:
            h, w, l, x_cam, y_cam, z_cam, theta, score = nums[-8:]
        else:
            h, w, l, x_cam, y_cam, z_cam, theta = nums[-7:]
            score = None
        # Stored in KITTI-style order: (h, w, l, xcam, ycam, zcam, ry),
        # where xcam=yc_orig, ycam=zc_orig, zcam=xc_orig. Recover the original
        # radar/lidar box center used by the detector.
        x = z_cam
        y = x_cam
        z = y_cam
        boxes.append({
            'cls': cls_name,
            'h': h, 'w': w, 'l': l,
            'x': x, 'y': y, 'z': z,
            'theta': theta,
            'score': score,
        })
    return boxes


def render_3d(lidar_pc, gt_boxes, pred_boxes, out_path: Path):
    pts = lidar_pc[:, :3]
    mask = (
        (pts[:, 0] > 0) & (pts[:, 0] < 80) &
        (pts[:, 1] > -40) & (pts[:, 1] < 40) &
        (pts[:, 2] > -5) & (pts[:, 2] < 8)
    )
    pts = pts[mask]
    if len(pts) > 12000:
        idx = np.linspace(0, len(pts) - 1, 12000).astype(int)
        pts = pts[idx]

    fig = plt.figure(figsize=(8, 4.5), dpi=160)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.55, c='#D94B4B', alpha=0.5, depthshade=False)

    def draw_boxes(boxes, color):
        for box in boxes:
            obj = Object3D(box['x'], box['y'], box['z'], box['l'], box['w'], box['h'], box['theta'])
            corners = obj.corners
            for i, j in LINES_3D:
                ax.plot(
                    [corners[i, 0], corners[j, 0]],
                    [corners[i, 1], corners[j, 1]],
                    [corners[i, 2], corners[j, 2]],
                    color=color, linewidth=1.6, alpha=0.98,
                )

    draw_boxes(gt_boxes, '#E53935')
    draw_boxes(pred_boxes, '#F2C94C')

    ax.view_init(elev=18, azim=-98)
    ax.set_xlim(0, 80)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-4, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('white')
        axis.line.set_color((1, 1, 1, 0))
    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def compose_final(path_bev: Path, path_3d: Path, out_path: Path):
    img_bev = trim_white(cv2.imread(str(path_bev)), pad=14)
    img_3d = trim_white(cv2.imread(str(path_3d)), pad=14)
    width = 420

    def resize_keep(img):
        h, w = img.shape[:2]
        nh = int(h * width / w)
        return cv2.resize(img, (width, nh), interpolation=cv2.INTER_AREA)

    bev = resize_keep(img_bev)
    img3d = resize_keep(img_3d)
    gap = 18
    pad = 14
    title_h = 34
    total_h = pad + title_h + bev.shape[0] + gap + title_h + img3d.shape[0] + pad
    canvas = np.full((total_h, width + 2 * pad, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(canvas, '2D', (pad + 8, pad + 24), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    canvas[pad + title_h:pad + title_h + bev.shape[0], pad:pad + width] = bev

    y2 = pad + title_h + bev.shape[0] + gap
    cv2.putText(canvas, '3D', (pad + 8, y2 + 24), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    canvas[y2 + title_h:y2 + title_h + img3d.shape[0], pad:pad + width] = img3d

    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    default_root = ROOT
    parser.add_argument('--exp', required=True)
    parser.add_argument('--epoch', type=int, default=18)
    parser.add_argument('--sample_idx', type=int, default=102)
    parser.add_argument('--result_root', default=str(default_root / 'logs'))
    parser.add_argument('--dataset_root', default=str(default_root / 'data' / 'k_radar_dataset'))
    parser.add_argument('--split_path', default=str(default_root / 'resources' / 'split' / 'test.txt'))
    parser.add_argument('--outdir', default=str(default_root / 'vis_outputs'))
    args = parser.parse_args()

    split_lines = Path(args.split_path).read_text().splitlines()
    sample_ref = split_lines[args.sample_idx].strip()
    seq, fname = [x.strip() for x in sample_ref.split(',')]
    sample_stem = fname.replace('.txt', '')

    root = Path(args.dataset_root) / seq
    path_label = root / 'info_label' / fname
    rdr_idx, ldr_idx, cam_idx = parse_label_header(path_label)
    path_img = root / 'cam-front' / f'cam-front_{cam_idx}.png'
    path_lidar = root / 'os2-64' / f'os2-64_{ldr_idx}.pcd'
    path_radar = root / 'sparse_cube' / f'cube_{rdr_idx}.npy'
    path_calib = root / 'info_calib' / 'calib_radar_lidar.txt'

    result_dir = Path(args.result_root) / args.exp / 'test_kitti' / f'epoch_{args.epoch}_subset' / '0.3'
    path_pred = result_dir / 'pred' / f'{args.sample_idx:06d}.txt'
    path_gt = result_dir / 'gt' / f'{args.sample_idx:06d}.txt'
    path_desc = result_dir / 'desc' / f'{args.sample_idx:06d}.txt'

    outdir = Path(args.outdir) / f'{args.exp}_e{args.epoch}_idx{args.sample_idx:06d}'
    outdir.mkdir(parents=True, exist_ok=True)

    calib_xyz = load_calib_xy(path_calib)
    lidar_pc = load_lidar_pcd(path_lidar, calib_xyz)
    gt_boxes = parse_result_boxes(path_gt, has_score=False)
    pred_boxes = parse_result_boxes(path_pred, has_score=True)

    bev, bev_meta = render_lidar_bev(lidar_pc)
    bev = draw_bev_boxes_cartesian(bev, gt_boxes, (0, 0, 255), bev_meta, thickness=2)
    bev = draw_bev_boxes_cartesian(bev, pred_boxes, (0, 215, 255), bev_meta, thickness=2)
    cv2.putText(bev, 'GT: red   Pred: yellow', (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    bev_path = outdir / 'bev_2d.png'
    cv2.imwrite(str(bev_path), bev)

    render_3d(lidar_pc, gt_boxes, pred_boxes, outdir / 'boxes_3d.png')
    compose_final(bev_path, outdir / 'boxes_3d.png', outdir / 'final_output.png')

    with open(outdir / 'meta.txt', 'w') as f:
        f.write(f'sample_idx={args.sample_idx}\n')
        f.write(f'seq={seq}\n')
        f.write(f'raw_sample={sample_ref}\n')
        if path_desc.exists():
            f.write('desc=' + ' | '.join(path_desc.read_text().splitlines()) + '\n')
        f.write(f'cam={path_img}\n')
        f.write(f'num_gt={len(gt_boxes)}\n')
        f.write(f'num_pred={len(pred_boxes)}\n')

    print('Saved to:', outdir)
    print(bev_path)
    print(outdir / 'boxes_3d.png')
    print(outdir / 'final_output.png')
    print(outdir / 'meta.txt')


if __name__ == '__main__':
    main()
