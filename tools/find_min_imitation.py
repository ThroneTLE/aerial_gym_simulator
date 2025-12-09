import argparse
import glob
import os
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_events(run_dir: str):
    event_files = glob.glob(os.path.join(run_dir, "summaries", "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"未找到 TensorBoard 事件文件，检查目录是否正确: {run_dir}")
    return event_files[0]


def find_min_tag(event_file: str, tag: str):
    ea = EventAccumulator(event_file)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        raise RuntimeError(f"事件文件中没有标量 {tag}")
    scalars = ea.Scalars(tag)
    best = min(scalars, key=lambda e: e.value)
    return best


def list_checkpoints(run_dir: str):
    ckpts = glob.glob(os.path.join(run_dir, "nn", "*.pth"))
    ckpts.sort()
    return [
        (os.path.basename(p), datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds"))
        for p in ckpts
    ]


def main():
    parser = argparse.ArgumentParser(description="Find min imitation/err step from TB logs.")
    parser.add_argument("run_dir", help="训练目录，例如 runs/teacher_residual_stage1_xx-xx-xx")
    parser.add_argument(
        "--tag", default="imitation/err", help="要搜索的标量名（默认 imitation/err）"
    )
    args = parser.parse_args()

    event_file = load_events(args.run_dir)
    best = find_min_tag(event_file, args.tag)
    print(f"最小 {args.tag}: {best.value:.6f} @ step {best.step} (wall_time={best.wall_time})")

    ckpts = list_checkpoints(args.run_dir)
    if ckpts:
        print("可用 checkpoint（按文件名排序）:")
        for name, mtime in ckpts:
            print(f"  {name} (mtime={mtime})")
    else:
        print("未找到任何 .pth checkpoint，请检查 save_frequency 是否开启。")


if __name__ == "__main__":
    main()
