import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_event_scalars(event_file):
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        points = ea.Scalars(tag)
        data[tag] = {p.step: float(p.value) for p in points}
    return data


def load_root_scalar(exp_dir, subdir, tag):
    event_files = sorted((exp_dir / subdir).glob("events.out.tfevents.*"))
    if not event_files:
        return {}
    scalars = load_event_scalars(event_files[0])
    return scalars.get(tag, {})


def scan_test_metrics(exp_dir):
    test_dir = exp_dir / "test"
    metrics = {}
    if not test_dir.exists():
        return metrics
    for sub in sorted(test_dir.iterdir()):
        if not sub.is_dir():
            continue
        event_files = sorted(sub.glob("events.out.tfevents.*"))
        if not event_files:
            continue
        m = re.match(r"^(minival|val_tot)_(BEV|3D)_conf_thr_([0-9.]+)_iou_([0-9.]+)_(.+)$", sub.name)
        if not m:
            continue
        split, metric_type, conf_thr, iou_thr, cls_key = m.groups()
        scalars = load_event_scalars(event_files[0])
        if not scalars:
            continue
        tag_name = f"{split}/{metric_type}_conf_thr_{conf_thr}"
        if tag_name not in scalars:
            tag_name = list(scalars.keys())[0]
        points = scalars.get(tag_name, {})
        key = {
            "split": split,
            "metric_type": metric_type,
            "conf_thr": conf_thr,
            "iou_thr": iou_thr,
            "cls_key": cls_key,
        }
        metrics[json.dumps(key, sort_keys=True)] = points
    return metrics


def parse_scores_from_pred_file(path):
    scores = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                scores.append(float(parts[-1]))
            except Exception:
                continue
    return scores


def scan_score_distribution(exp_dir, conf_thr):
    root = exp_dir / "test_kitti"
    out = {}
    if not root.exists():
        return out
    conf_name = str(conf_thr)
    for epoch_dir in sorted(root.glob("epoch_*_*")):
        m = re.match(r"^epoch_(\d+)_(subset|total)$", epoch_dir.name)
        if not m:
            continue
        epoch = int(m.group(1))
        split = m.group(2)
        pred_dir = epoch_dir / conf_name / "pred"
        if not pred_dir.exists():
            continue
        score_list = []
        for txt in pred_dir.glob("*.txt"):
            score_list.extend(parse_scores_from_pred_file(txt))
        if not score_list:
            continue
        arr = np.array(score_list, dtype=np.float64)
        out[(split, epoch)] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max()),
        }
    return out


def to_sorted_items(step_to_value):
    return sorted(step_to_value.items(), key=lambda x: x[0])


def first_last_mean(values, n=10):
    if not values:
        return (math.nan, math.nan)
    arr = np.array(values, dtype=np.float64)
    n = min(n, len(arr))
    return float(arr[:n].mean()), float(arr[-n:].mean())


def select_metric(metrics, split, metric_type, conf_thr, iou_thr, cls_key):
    target = {
        "split": split,
        "metric_type": metric_type,
        "conf_thr": str(conf_thr),
        "iou_thr": str(iou_thr),
        "cls_key": cls_key,
    }
    key = json.dumps(target, sort_keys=True)
    if key in metrics:
        return target, metrics[key]
    fallback = None
    for raw_key, step_map in metrics.items():
        desc = json.loads(raw_key)
        if desc["split"] == split and desc["metric_type"] == metric_type and desc["cls_key"] == cls_key:
            fallback = (desc, step_map)
            if desc["conf_thr"] == str(conf_thr):
                return desc, step_map
    if fallback is not None:
        return fallback
    return None, {}


def summarize(exp_dir, args):
    exp_dir = Path(exp_dir).resolve()
    train_avg_loss = load_root_scalar(exp_dir, "train_epoch", "train/avg_loss")
    train_ce = load_root_scalar(exp_dir, "train_iter", "train/loss_ce")
    train_bbox = load_root_scalar(exp_dir, "train_iter", "train/loss_bbox")
    train_con = load_root_scalar(exp_dir, "train_iter", "train/loss_contrastive")
    train_lr = load_root_scalar(exp_dir, "train_iter", "train/learning_rate")
    test_metrics = scan_test_metrics(exp_dir)
    score_dist = scan_score_distribution(exp_dir, args.score_conf)

    metric_desc, target_series = select_metric(
        test_metrics,
        args.metric_split,
        args.metric_type,
        args.metric_conf,
        args.metric_iou,
        args.metric_cls,
    )

    summary = {
        "experiment_dir": str(exp_dir),
        "target_metric": metric_desc,
        "target_series": to_sorted_items(target_series),
        "best_epoch_by_target": None,
        "topk_epochs_by_target": [],
        "last_epoch_by_target": None,
        "target_drop_from_best_to_last": None,
        "train_avg_loss": to_sorted_items(train_avg_loss),
        "train_iter_stats": {},
        "score_distribution": [],
        "recommendation": {},
    }

    if target_series:
        sorted_pairs = to_sorted_items(target_series)
        best_epoch, best_val = max(sorted_pairs, key=lambda x: x[1])
        last_epoch, last_val = sorted_pairs[-1]
        summary["best_epoch_by_target"] = {"epoch": best_epoch, "value": best_val}
        summary["last_epoch_by_target"] = {"epoch": last_epoch, "value": last_val}
        summary["target_drop_from_best_to_last"] = best_val - last_val
        topk = sorted(sorted_pairs, key=lambda x: x[1], reverse=True)[: args.topk]
        summary["topk_epochs_by_target"] = [{"epoch": e, "value": v} for e, v in topk]

    iter_map = {
        "loss_ce": train_ce,
        "loss_bbox": train_bbox,
        "loss_contrastive": train_con,
        "learning_rate": train_lr,
    }
    for k, v in iter_map.items():
        arr = [x[1] for x in to_sorted_items(v)]
        if not arr:
            continue
        first_mean, last_mean = first_last_mean(arr, args.window)
        summary["train_iter_stats"][k] = {
            "count": len(arr),
            "first": float(arr[0]),
            "last": float(arr[-1]),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "first_window_mean": first_mean,
            "last_window_mean": last_mean,
        }

    for (split, epoch), stats in sorted(score_dist.items(), key=lambda x: (x[0][0], x[0][1])):
        summary["score_distribution"].append(
            {
                "split": split,
                "epoch": epoch,
                "conf_thr": str(args.score_conf),
                **stats,
            }
        )

    rec = {}
    if summary["best_epoch_by_target"] is not None:
        rec["recommended_epoch"] = summary["best_epoch_by_target"]["epoch"]
        rec["reason"] = "best_target_metric"
        rec["recommended_model_path"] = str(exp_dir / "models" / f"model_{rec['recommended_epoch']}.pt")
    summary["recommendation"] = rec
    return summary


def print_summary(summary):
    print(f"Experiment: {summary['experiment_dir']}")
    if summary["target_metric"] is not None:
        tm = summary["target_metric"]
        print(
            f"Target metric: {tm['split']} {tm['metric_type']} conf={tm['conf_thr']} iou={tm['iou_thr']} cls={tm['cls_key']}"
        )
    else:
        print("Target metric: not found")

    if summary["best_epoch_by_target"] is not None:
        b = summary["best_epoch_by_target"]
        l = summary["last_epoch_by_target"]
        print(f"Best epoch: {b['epoch']} value={b['value']:.6f}")
        print(f"Last epoch: {l['epoch']} value={l['value']:.6f}")
        print(f"Drop best->last: {summary['target_drop_from_best_to_last']:.6f}")
        print("Top epochs by target:")
        for item in summary["topk_epochs_by_target"]:
            print(f"  epoch {item['epoch']}: {item['value']:.6f}")

    if summary["train_avg_loss"]:
        first_epoch, first_loss = summary["train_avg_loss"][0]
        last_epoch, last_loss = summary["train_avg_loss"][-1]
        print(f"Epoch avg_loss: first={first_epoch}:{first_loss:.6f}, last={last_epoch}:{last_loss:.6f}")

    if summary["train_iter_stats"]:
        print("Train iter stats:")
        for k, v in summary["train_iter_stats"].items():
            print(
                f"  {k}: first={v['first']:.6f}, last={v['last']:.6f}, "
                f"first_w={v['first_window_mean']:.6f}, last_w={v['last_window_mean']:.6f}, "
                f"min={v['min']:.6f}, max={v['max']:.6f}, count={v['count']}"
            )

    if summary["score_distribution"]:
        print("Score distribution per epoch:")
        for item in summary["score_distribution"]:
            print(
                f"  {item['split']} epoch {item['epoch']} conf={item['conf_thr']} "
                f"count={item['count']} mean={item['mean']:.6f} std={item['std']:.6f} "
                f"p10={item['p10']:.6f} p50={item['p50']:.6f} p90={item['p90']:.6f}"
            )

    if summary["recommendation"]:
        r = summary["recommendation"]
        print(
            f"Recommended checkpoint: epoch {r['recommended_epoch']} -> {r['recommended_model_path']}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--metric-split", type=str, default="val_tot", choices=["val_tot", "minival"])
    parser.add_argument("--metric-type", type=str, default="3D", choices=["3D", "BEV"])
    parser.add_argument("--metric-conf", type=str, default="0.3")
    parser.add_argument("--metric-iou", type=str, default="0.3")
    parser.add_argument("--metric-cls", type=str, default="sed")
    parser.add_argument("--score-conf", type=str, default="0.3")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize(args.exp, args)
    print_summary(summary)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary json: {args.json_out}")


if __name__ == "__main__":
    main()
