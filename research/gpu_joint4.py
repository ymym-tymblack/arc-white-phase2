"""Offline CUDA diagnostics; never part of the deterministic submission.

Example:
  research/gpu_env.sh research/gpu_joint4.py network.npz joint4.json \
      --samples 1000000 --replicates 2 --chunk 8192 --k-block 4 \
      --max-depth 4 --k211-layers 2 4

The JSON manifest and each completed <output_stem>_rep<N>.npz are replaced
atomically. Every replicate replays exactly the same CUDA normal draws in both
passes, including its final partial chunk. All empirical central moments use
normalization 1/N (not unbiased k-statistics). Layer 1 is measured, not zeroed.
Forward GEMMs and moment contractions use float32, without TF32 or autocast;
means, centering, elementwise products, and running sums use float64. Therefore
same-sample centering is exact up to floating-point arithmetic, but contractions
still have float32 rounding error. Independent replicates expose sampling noise,
not this common numerical error. No arrays are standardized.

T[i,j,k] = E[c_i c_j c_k]; K211[i,j,k] = kappa(c_i,c_j,c_k,c_k).
K31[i,j] = kappa(c_i,c_i,c_i,c_j); K22[i,j] = kappa(c_i,c_i,c_j,c_j).
All suffixes are 1-based layers; forward orientation is x @ weights[layer].
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def _atomic_npz(path, values):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, values):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp",
                                     delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(values, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _third_sum(centered, centered32, target, k_block):
    """Contract one B*n*k_block product, never a B*n*n*n tensor."""
    width = centered.shape[1]
    for start in range(0, width, k_block):
        stop = min(start + k_block, width)
        pairs = centered[:, :, None] * centered[:, None, start:stop]
        contracted = centered32.T @ pairs.flatten(1).float()
        target[:, :, start:stop].add_(contracted.reshape(width, width, stop - start).double())


def _preactivation_sum(centered, sums, k_block):
    c32 = centered.float()
    squared = centered.square()
    squared32 = squared.float()
    sums["S"].add_((c32.T @ c32).double())
    sums["K4"].add_(squared.square().sum(0))
    sums["K22"].add_((squared32.T @ squared32).double())
    sums["K31"].add_(((squared * centered).float().T @ c32).double())
    _third_sum(centered, c32, sums["T"], k_block)
    if "K211" in sums:
        width = centered.shape[1]
        for start in range(0, width, k_block):
            stop = min(start + k_block, width)
            pairs = centered[:, :, None] * squared[:, None, start:stop]
            contracted = c32.T @ pairs.flatten(1).float()
            sums["K211"][:, :, start:stop].add_(
                contracted.reshape(width, width, stop - start).double())


@torch.inference_mode()
def measure_network(weights, *, samples, chunk, seed, max_depth=None,
                    k211_layers=(), k_block=4, progress=None, observe=None):
    """Return NumPy moments and exact counts for one independent CUDA stream.

    weights is (depth,n,n), float32. progress receives JSON-compatible events.
    CPU work is serialization/metadata only; all simulation and reductions run
    on CUDA. Device storage scales as depth*n^3 + chunk*n*k_block.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU simulation is not supported")
    if samples <= 0 or chunk <= 0 or k_block <= 0:
        raise ValueError("samples, chunk, and k_block must be positive")
    if weights.ndim != 3 or weights.shape[1] != weights.shape[2]:
        raise ValueError("weights must have shape (depth,n,n)")
    if weights.dtype != np.float32:
        raise ValueError("weights must be float32")
    depth = weights.shape[0] if max_depth is None else max_depth
    if not 1 <= depth <= weights.shape[0]:
        raise ValueError("max_depth must be within the network depth")
    input_width = weights.shape[1]
    width = input_width if observe is None else observe
    if not 1 <= width <= min(input_width, 128):
        raise ValueError("observe must select 1..min(input_width,128) coordinates")
    selected = set(k211_layers)
    if any(layer < 1 or layer > depth for layer in selected):
        raise ValueError("K211 layers must be within the measured 1-based depth")
    if not 0 <= seed < 2**63:
        raise ValueError("RNG seed must be in [0, 2**63)")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    # Replay requires deterministic forward GEMMs as well as identical draws.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    ws = torch.as_tensor(weights[:depth], device=device, dtype=torch.float32)

    def zeros(*shape):
        return torch.zeros(shape, device=device, dtype=torch.float64)

    means = [zeros(width) for _ in range(depth)]
    hmeans = [zeros(width) for _ in range(depth)]
    all_hmeans = [zeros(input_width) for _ in range(depth)]
    all_hsecond = [zeros(input_width) for _ in range(depth)]
    accumulators = []
    for layer in range(1, depth + 1):
        sums = {"S": zeros(width, width), "T": zeros(width, width, width),
                "K4": zeros(width), "K22": zeros(width, width),
                "K31": zeros(width, width), "hcov": zeros(width, width),
                "hthird": zeros(width, width, width)}
        if layer in selected:
            sums["K211"] = zeros(width, width, width)
        accumulators.append(sums)

    counts = []
    pass_seconds = []
    # Explicitly disable any enclosing autocast context as well.
    with torch.autocast(device_type="cuda", enabled=False):
        for pass_number in (1, 2):
            generator = torch.Generator(device=device).manual_seed(seed)
            torch.cuda.synchronize()
            started = time.perf_counter()
            last_report = started
            done = 0
            while done < samples:
                count = min(chunk, samples - done)
                x = torch.randn((count, input_width), generator=generator,
                                device=device, dtype=torch.float32)
                for layer in range(depth):
                    z = x @ ws[layer]
                    x = torch.relu(z)
                    if pass_number == 1:
                        means[layer].add_(z[:, :width].sum(0, dtype=torch.float64))
                        hmeans[layer].add_(x[:, :width].sum(0, dtype=torch.float64))
                        all_hmeans[layer].add_(x.sum(0, dtype=torch.float64))
                        all_hsecond[layer].add_(x.double().square().sum(0))
                    else:
                        centered = z[:, :width].double() - means[layer]
                        _preactivation_sum(centered, accumulators[layer], k_block)
                        del centered
                        hc = x[:, :width].double() - hmeans[layer]
                        hc32 = hc.float()
                        accumulators[layer]["hcov"].add_((hc32.T @ hc32).double())
                        _third_sum(hc, hc32, accumulators[layer]["hthird"], k_block)
                done += count
                now = time.perf_counter()
                if progress is not None and (done == samples or now - last_report >= 10):
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    progress({"pass": pass_number, "count": done, "samples": samples,
                              "elapsed_seconds": elapsed,
                              "samples_per_second": done / elapsed})
                    last_report = time.perf_counter()
            torch.cuda.synchronize()
            pass_seconds.append(time.perf_counter() - started)
            counts.append(done)
            if pass_number == 1:
                for mean, hmean in zip(means, hmeans):
                    mean.div_(done)
                    hmean.div_(done)
                for hmean, hsecond in zip(all_hmeans, all_hsecond):
                    hmean.div_(done)
                    hsecond.div_(done)

    result = {"N": np.int64(counts[1]), "samples": np.int64(samples),
              "mean_count": np.int64(counts[0]), "moment_count": np.int64(counts[1]),
              "rng_seed": np.int64(seed), "chunk": np.int64(chunk),
              "k_block": np.int64(k_block), "depth": np.int64(depth),
              "width": np.int64(width), "input_width": np.int64(input_width),
              "observed_indices": np.arange(width), "pass_seconds": np.asarray(pass_seconds),
              "k211_layers": np.asarray(sorted(selected), dtype=np.int64),
              "estimator": np.asarray("same-sample centered plug-in, divisor N"),
              "arithmetic": np.asarray("fp32 GEMM; fp64 means/products/accumulators; no TF32")}
    for index, sums in enumerate(accumulators):
        layer = index + 1
        for value in sums.values():
            value.div_(counts[1])
        covariance = sums["S"]
        variance = covariance.diagonal()
        sums["K4"].sub_(3 * variance.square())
        sums["K22"].sub_(variance[:, None] * variance[None, :] + 2 * covariance.square())
        sums["K31"].sub_(3 * variance[:, None] * covariance)
        if "K211" in sums:
            # kappa(i,j,k,k) = E[c_i c_j c_k^2] - S_ij S_kk - 2 S_ik S_jk.
            for start in range(0, width, k_block):
                stop = min(start + k_block, width)
                sums["K211"][:, :, start:stop].sub_(
                    covariance[:, :, None] * variance[None, None, start:stop]
                    + 2 * covariance[:, None, start:stop] * covariance[None, :, start:stop])
        result[f"m_{layer}"] = means[index].cpu().numpy()
        result[f"hmean_{layer}"] = hmeans[index].cpu().numpy()
        result[f"hmean_all_{layer}"] = all_hmeans[index].cpu().numpy()
        result[f"hvar_all_{layer}"] = (all_hsecond[index] - all_hmeans[index].square()).cpu().numpy()
        for name, value in sums.items():
            result[f"{name}_{layer}"] = value.cpu().numpy()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("network", type=Path, help="NPZ with float32 weights and integer seed")
    parser.add_argument("output", type=Path, help="aggregate JSON manifest path (.json)")
    parser.add_argument("--samples", type=int, required=True, help="exact samples per replicate")
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--k211-layers", type=int, nargs="*", default=[])
    parser.add_argument("--k-block", type=int, default=4)
    parser.add_argument("--observe", type=int, help="measure joint moments for this prefix; forward still uses every neuron")
    parser.add_argument("--seed", type=int, default=20261015, help="base CUDA RNG seed")
    args = parser.parse_args()
    if args.output.suffix != ".json":
        parser.error("output must be a .json manifest path")
    if args.replicates <= 0 or args.samples <= 0 or args.chunk <= 0 or args.k_block <= 0:
        parser.error("replicates, samples, chunk, and k-block must be positive")
    seeds = [args.seed + replicate * 1_000_003 for replicate in range(args.replicates)]
    if seeds[0] < 0 or seeds[-1] >= 2**63:
        parser.error("replicate RNG seeds must be in [0, 2**63)")
    with np.load(args.network, allow_pickle=False) as network:
        weights = network["weights"]
        network_seed = int(network["seed"].item())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {"network": str(args.network.resolve()), "network_seed": network_seed,
              "samples_per_replicate": args.samples, "replicate_count": args.replicates,
              "rng_seeds": seeds, "chunk": args.chunk, "k_block": args.k_block,
              "network_shape": list(weights.shape), "max_depth": args.max_depth,
              "k211_layers": sorted(set(args.k211_layers)),
              "observe": args.observe,
              "input_distribution": "independent standard Gaussian",
              "estimator": "same-sample centered plug-in, divisor N"}
    import wandb

    run = wandb.init(project="arc-whest-phase2", mode="offline",
                     name=f"joint4_{args.output.stem}", config=config)
    manifest = {"config": config, "wandb_run_id": run.id,
                "wandb_directory": run.dir, "completed_replicates": []}
    try:
        _atomic_json(args.output, manifest)
        for replicate, seed in enumerate(seeds):
            def progress(event):
                metrics = {"replicate": replicate, **event}
                run.log(metrics)
                print(f"rep={replicate} pass={event['pass']} "
                      f"N={event['count']}/{event['samples']} "
                      f"seconds={event['elapsed_seconds']:.1f} "
                      f"samples/s={event['samples_per_second']:.0f}", flush=True)

            values = measure_network(weights, samples=args.samples, chunk=args.chunk,
                                     seed=seed, max_depth=args.max_depth,
                                     k211_layers=args.k211_layers, k_block=args.k_block,
                                     progress=progress, observe=args.observe)
            values.update(network_seed=np.int64(network_seed), seed=np.int64(network_seed),
                          replicate=np.int64(replicate),
                          network_path=np.asarray(str(args.network.resolve())),
                          wandb_run_id=np.asarray(run.id),
                          config_json=np.asarray(json.dumps(config, sort_keys=True)))
            destination = args.output.with_name(f"{args.output.stem}_rep{replicate}.npz")
            _atomic_npz(destination, values)
            manifest["completed_replicates"].append(
                {"replicate": replicate, "path": destination.name, "rng_seed": seed,
                 "mean_count": int(values["mean_count"]),
                 "moment_count": int(values["moment_count"]),
                 "pass_seconds": values["pass_seconds"].tolist()})
            _atomic_json(args.output, manifest)
            print(f"saved {destination}", flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
