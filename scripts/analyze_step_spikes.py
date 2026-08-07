#!/usr/bin/env python3
"""Correlate per-step training time spikes with the data samples in each batch.

Run on the training machine where the run artifacts and the pkl live. The step
-> sample-index mapping assumes the alignment config: --shuffle_train_data false
(SequentialSampler), per_device_train_batch_size=1, gradient_accumulation_steps=1,
no distributed sampler sharding -> global_step g loads dataset index g on every rank.

Usage:
  python scripts/analyze_step_spikes.py \
      --trainer_state logs/alignment_*/tierX_*/trainer_state.json \
      --pkl /data/models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl \
      [--hash_log logs/alignment_*/tierX_*/data_hashes/rank0.jsonl] \
      [--k 5.0] [--cur_idx 3]

Use the *fixed* pkl that training actually consumed (paths must match).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

FILES_PER_SAMPLE = 2  # current frame + pre_1s frame, see datasets.random_frames_to_tensor


def load_times(trainer_state_path):
    """step -> time_elapsed from log_history (first entry has no baseline)."""
    with open(trainer_state_path) as f:
        state = json.load(f)
    times = {}
    for e in state.get("log_history", []):
        if "time_elapsed" in e:
            times[int(e["step"])] = float(e["time_elapsed"])
    return times


def mad(x):
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def find_spikes(times, k=5.0):
    steps = np.array(sorted(times))
    t = np.array([times[s] for s in steps])
    if t.size == 0:
        return steps, t, 0.0, 0.0, []
    med = float(np.median(t))
    scale = max(mad(t), 1e-9)
    thr = med + k * 1.4826 * scale  # robust: median + k * scaled MAD
    spike_idx = [int(s) for s in steps[t > thr]]
    return steps, t, med, thr, spike_idx


def per_sample_files(scene, cur_idx):
    """The two npz files this sample loads (datasets.random_frames_to_tensor)."""
    img = scene.get("image", [])
    pre = scene.get("pre_1s_image", [])
    main = img[cur_idx] if cur_idx < len(img) else None
    pre_f = pre[cur_idx] if cur_idx < len(pre) else None
    return [p for p in (main, pre_f) if p]


def summarize_scene(scene, cur_idx):
    img = scene.get("image", [])
    pre = scene.get("pre_1s_image", [])
    act = scene.get("action")
    act = np.asarray(act) if act is not None else np.array([])
    files = per_sample_files(scene, cur_idx)
    roots = {os.path.dirname(p) for p in files}
    with np.errstate(all="ignore"):
        act_abs_max = float(np.abs(act).max()) if act.size else 0.0
        act_nan = int(np.isnan(act).sum()) if act.size else 0
    return {
        "index": None,
        "n_img": len(img),
        "n_pre": len(pre),
        "roots": sorted(roots),
        "files": files,
        "action_abs_max": act_abs_max,
        "action_nan": act_nan,
        "text": scene.get("text", [""] * (cur_idx + 1))[cur_idx] if scene.get("text") else "",
    }


def inspect(pkl_path, indices, cur_idx, sample_pop):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    n = len(data)
    step = max(1, n // sample_pop)  # population stride
    rows = []
    for i in indices:
        if i >= n:
            rows.append({"index": i, "error": f"index {i} >= dataset size {n}"})
            continue
        r = summarize_scene(data[i], cur_idx)
        r["index"] = i
        r["exists"] = [os.path.exists(p) for p in r["files"]]
        rows.append(r)

    pop = [summarize_scene(data[i], cur_idx) for i in range(0, n, step)]
    return rows, pop


def report(times, steps, t, med, thr, spike_idx, rows, pop, hash_log=None):
    n = len(t)
    print("=" * 72)
    print(f"Steps with time_elapsed: {n}")
    if n == 0:
        print("No time_elapsed found — did this run use the updated LoggingTrainer?")
        return 1
    print(f"median={med:.3f}s  MAD={mad(t):.3f}s  spike threshold (median+{1.4826:.4f}*k*MAD)={thr:.3f}s")
    print(f"max step time: {t.max():.3f}s at step {steps[t.argmax()]}  ({t.max()/med:.1f}x median)")
    print(f"Spike steps ({len(spike_idx)}): {spike_idx}")
    print("=" * 72)

    if hash_log:
        with open(hash_log) as f:
            hsteps = [json.loads(l)["step"] for l in f]
        contiguous = hsteps == list(range(min(hsteps), max(hsteps) + 1))
        print(f"data_hashes: {len(hsteps)} steps, contiguous={contiguous}, "
              f"first={hsteps[0]} last={hsteps[-1]}")
        if not contiguous:
            print("  !! hashes not contiguous — step->index identity may be wrong")
        print("-" * 72)

    if spike_idx:
        print("Spike samples (step == index):")
        for r in rows:
            if "error" in r:
                print(f"  step {r['index']}: {r['error']}")
                continue
            roots = ",".join(r["roots"])
            print(f"  step {r['index']}: n_img={r['n_img']} n_pre={r['n_pre']} "
                  f"action_abs_max={r['action_abs_max']:.3f} nan={r['action_nan']} "
                  f"exists={r['exists']}")
            for p in r["files"]:
                print(f"      {p}")
            if r["text"]:
                print(f"      prompt: {r['text'][:60]!r}")
        print("-" * 72)

        print("Population vs spikes (median of non-spike population):")
        ok = [p for p in pop if p.get("index") is None or p.get("index", -1) not in set(spike_idx)]
        for key, label in [("n_img", "n_img"), ("n_pre", "n_pre"), ("action_abs_max", "action_abs_max")]:
            vals = np.array([p[key] for p in ok])
            svals = np.array([r[key] for r in rows if "error" not in r])
            if len(vals) == 0 or len(svals) == 0:
                continue
            pmin, pmax, pmed = vals.min(), vals.max(), np.median(vals)
            print(f"  {label}: pop median={pmed:.2f} range=[{pmin:.2f},{pmax:.2f}]  "
                  f"spikes={[round(v,2) for v in svals]}")
            if not ((svals >= pmin) & (svals <= pmax)).all():
                print(f"    !! spike samples OUTSIDE population range for {label}")
        print("-" * 72)

        # root/mount check: do spike samples share an unusual path root?
        pop_roots = {}
        for p in ok:
            for r in p["roots"]:
                pop_roots[r] = pop_roots.get(r, 0) + 1
        spike_roots = {}
        for r in rows:
            if "error" in r:
                continue
            for x in r["roots"]:
                spike_roots[x] = spike_roots.get(x, 0) + 1
        rare = {r: c for r, c in spike_roots.items() if pop_roots.get(r, 0) <= 2}
        if rare:
            print(f"  Spike samples read files from roots absent in population: {rare}")
        else:
            print("  Spike samples' path roots all present in population (storage looks normal).")
    else:
        print("No spike steps above threshold.")
        print("-" * 72)

    # periodicity — spikes repeating at a fixed interval suggest a sync/collective cause,
    # not random samples.
    if len(spike_idx) >= 4:
        gaps = np.diff(spike_idx)
        print(f"Spike step gaps: {gaps.tolist()}  (fixed gap => external/collective cause, "
              f"not sample content)")
    return 0


def self_test():
    """Runnable check: synthetic pkl + trainer_state -> spikes found, mapping works."""
    import tempfile
    tmp = tempfile.mkdtemp()
    # synthetic pkl: 50 scenes, one with a longer trajectory (the outlier)
    def scene_paths(i, sub):
        return [os.path.join(tmp, f"scene{i:04d}", sub, f"{k:04d}.npz") for k in range(20 if sub == "img" else 10)]
    scenes = []
    for i in range(50):
        scenes.append({
            "image": scene_paths(i, "img"),
            "pre_1s_image": scene_paths(i, "pre"),
            "action": np.zeros((10, 4), dtype=np.float32),
            "pre_1s_action": np.zeros((4, 4), dtype=np.float32),
            "text": ["go straight"] * 4,
        })
    # the spike sample's two files must exist (as they would after a real run)
    for p in [scenes[17]["image"][3], scenes[17]["pre_1s_image"][3]]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"\x00")
    pkl = os.path.join(tmp, "synthetic.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(scenes, f)
    # synthetic trainer_state: step 17 is a spike
    lh = [{"step": s, "loss": 1.0, "time_elapsed": (3.0 if s == 17 else 1.0)} for s in range(1, 30)]
    ts = os.path.join(tmp, "trainer_state.json")
    with open(ts, "w") as f:
        json.dump({"log_history": lh}, f)

    times = load_times(ts)
    assert len(times) == 29, "time_elapsed count"
    steps, t, med, thr, spikes = find_spikes(times)
    assert spikes == [17], f"expected spike at step 17, got {spikes}"
    rows, pop = inspect(pkl, spikes, cur_idx=3, sample_pop=10)
    assert rows[0]["n_img"] == 20 and rows[0]["exists"] == [True, True], rows[0]
    print("self-test OK: spike detection + step->index + sample inspection all consistent")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer_state")
    ap.add_argument("--pkl")
    ap.add_argument("--hash_log", default=None)
    ap.add_argument("--k", type=float, default=5.0, help="MAD multiplier for spike threshold")
    ap.add_argument("--cur_idx", type=int, default=3, help="cur_frame_idx used in training")
    ap.add_argument("--sample_pop", type=int, default=2000, help="max population items to summarize")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.trainer_state or not args.pkl:
        ap.error("--trainer_state and --pkl are required (or use --self-test)")

    times = load_times(args.trainer_state)
    steps, t, med, thr, spike_idx = find_spikes(times, k=args.k)
    rows, pop = (inspect(args.pkl, spike_idx, args.cur_idx, args.sample_pop) if spike_idx
                 else ([], []))
    return report(times, steps, t, med, thr, spike_idx, rows, pop, args.hash_log)


if __name__ == "__main__":
    sys.exit(main())
