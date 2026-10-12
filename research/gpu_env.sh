#!/usr/bin/env bash
# GPU 研究環境(提出用の環境とは別)で Python を実行する。例: research/gpu_env.sh research/gpu_reference.py 3 0 1
exec uv run --no-project --python 3.12 --with torch --with einops --with jaxtyping --with joblib \
  --with matplotlib --with opt_einsum --with pandas --with tqdm --with scipy --with numpy \
  --with wandb --with-editable /kaggle/ext/mlp_cumulant_propagation python "$@"
