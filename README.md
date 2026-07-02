# DisasterM3 Solution for 'Reaching new heights with GeoFM'


Final submission: **`final_submission/0627_REC_5463_b5316T50_jointV8864T15_jointW6302T375_hb1.8336_hv2.7824.zip`**
Leaderboard score: **0.5463 (public)**, **0.5499 (private)**.

---

## 1. The model behind each channel (7 checkpoints total, all in `models/`)

| ch | class | model(s) | ckpt | combine → threshold |
|----|-------|----------|------|---------------------|
| 0 | building | ENS3: hrnet + convnext + mbconv | last / **best** / **best** | arithmetic mean of ch0 → HARD **> 0.50** |
| 1 | veg | joint veg+water convnext | last | ch1 → HARD **> 0.15** |
| 2 | water | joint veg+water convnext (same model) | last | ch2 → HARD **> 0.375** |
| 3 | height | ENS3: hrnet + convnext + mbconv | last / last / last | per-model clip ≤100 m, then mean (raw metres) |

The novel part is ch1+ch2: **one** model jointly fits veg and water; the sparse water class borrows
spatial context from dense veg (iou_w +0.0106 vs a separate water specialist). ch0/ch3 reuse the
earlier three-backbone overfit ensembles. All 7 checkpoints + their configs are shipped, so the full
submission is reproducible from weights — nothing is hidden inside a precomputed file.


---

## 2. Code Layout

```
FINAL_Code/
├── README.md
├── reproduce_from_soft.sh     # ← FAST path: rebuild+verify submission, no GPU, ~1 min
├── run_full_inference.sh      # ← FULL path: run all 7 models from weights, then assemble
├── make_sub.py                # threshold soft_master → zip (used by fast path)
├── assemble.py                # combine 7 model prediction dirs → zip (used by full path)
├── requirements.txt           # pinned deps (python 3.12); ever installed offline from vendor/
├── vendor/                    # offline install of `ever` (not on PyPI): wheel + source + README
├── final_submission/          # the exact zip submitted to the LB (0.5463) — on Hugging Face (see §3)
├── soft_master/               # 946 × (4,256,256): ch0 HARD@0.50, ch1/ch2 joint SOFT, ch3 height raw — on Hugging Face (see §3)
├── models/                    # config.py + manifest in git; the 7 model-*.pth are on Hugging Face (see §4)
│   ├── inference_manifest.json          # the 7 models: dir, model_type, ckpt, channel, threshold
│   ├── ch0_building_ens3/{hrnet,convnext,mbconv}/{model-*.pth, config.py}
│   ├── ch1ch2_vegwater_joint/convnext/{model-last.pth, config.py}
│   └── ch3_height_ens3/{hrnet,convnext,mbconv}/{model-last.pth, config.py}
├── code/                      # self-contained inference code subset
│   ├── predict_multi_tta.py   #   model forward / prediction entrypoint
│   ├── module/                #   GeoFMNet + backbones + losses
│   ├── data/                  #   GeoFMMultiEmbeddingDataset loader
│   └── _stats/                #   token normalisation stats (terramind/thor s1/s2)
├── training/                  # training scripts behind the 7 ckpts (see training/TRAINING.md)
│   ├── train.py               #   training entrypoint (EVER th_ddp)
│   ├── run_train.sh           #   example torchrun launcher
│   └── configs/geofm/         #   base + 7 model training configs
└── submissions/               # (output) zips written by the scripts — created on first run
```

> The `models/*/config.py` files are the original **training-time** configs (kept as provenance).
> Inference reads nothing from them — every input comes from CLI flags in `run_full_inference.sh` —
> so the `/work/...` paths inside them are inert and safe to ignore.

---

## 3. Fast reproduction (no GPU, no embeddings, ~1 min)

### Getting the reproduction artifacts (download from Hugging Face)

`soft_master/` (946 `.npy`, ~950 MB) and `final_submission/` (~220 MB LB zip) are too large for git,
so they are hosted on Hugging Face at
**[`Kingdrone-Junjue/GeoFM`](https://huggingface.co/Kingdrone-Junjue/GeoFM)** alongside the weights.
Fetch them into the repo root (the layout matches, so nothing else changes):

```bash
huggingface-cli download Kingdrone-Junjue/GeoFM --include "soft_master/**" "final_submission/**" --local-dir .
# → restores soft_master/*.npy and final_submission/*.zip
```

`soft_master/` already holds every model's soft output, so the submission is one command:

```bash
bash reproduce_from_soft.sh
```

Runs `make_sub.py 0.15 0.375` and verifies the result matches `final_submission/` array-for-array
(**max abs diff 0** over all 946 tiles). Output: `submissions/joint_v0.15_w0.375.zip`.
Other thresholds: `python3 make_sub.py <T_veg> <T_water>` (0.15/0.375 is the tuned optimum;
the scorer has a veg→water coupling, so the two are co-calibrated — don't move them independently).

## 4. Full inference from model weights (GPU + embeddings)

### Getting the model weights (download from Hugging Face)

The 7 checkpoints (`model-*.pth`) are too large for git, so they are hosted on Hugging Face at
**[`Kingdrone-Junjue/GeoFM`](https://huggingface.co/Kingdrone-Junjue/GeoFM)** under `models/`. The
`config.py` files and `inference_manifest.json` are already in the repo; only the `.pth` weights need
downloading. Fetch them into the repo's `models/` folder (the layout matches, so nothing else changes):

```bash
huggingface-cli download Kingdrone-Junjue/GeoFM --include "models/**/*.pth" --local-dir .
# → restores models/ch0_building_ens3/.../model-*.pth, models/ch1ch2_vegwater_joint/..., models/ch3_height_ens3/...
```

This is only needed for §4 (full inference) below; §3 (fast reproduction from `soft_master/`) needs no weights.

First set up the environment (`ever` is not on PyPI — it installs offline from `vendor/`):

```bash
conda create -n geofm python=3.12 -y && conda activate geofm
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126  # PyPI deps (cu126 torch)
pip install --no-index --no-deps vendor/ever-2.0.0-py3-none-any.whl # offline ever  (see vendor/README.md)
```

> **GPU / CUDA build — READ THIS if reproducing on newer hardware.** `requirements.txt` pins the
> **cu126** torch build, which only supports GPUs up to **sm_90** (RTX 40-series and older, A100/H100).
> On **Blackwell** GPUs (RTX 50-series, e.g. **RTX 5080/5090, sm_120**) cu126 fails at runtime with
> `CUDA error: no kernel image is available for execution on the device`. Install a matching build for
> your GPU instead, e.g. for Blackwell:
> ```bash
> pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps \
>     torch==2.8.0+cu128 torchvision==0.23.0+cu128
> ```
> Verify with: `python -c "import torch;print(torch.randn(8,8,device='cuda').sum())"` — no error = good.

> **Windows users.** The inference script prints a `→` character; on the default `cp1252` console this
> raises `UnicodeEncodeError`. Export UTF-8 before running (no code change needed):
> `set PYTHONUTF8=1` (cmd) / `$env:PYTHONUTF8=1` (PowerShell) / `export PYTHONUTF8=1` (bash).

### Getting the test embeddings

Download the 6 test-embedding sources from the official dataset
(the `data/test/` folder), then point `EMB` at that folder:

`EMB` must contain these 6 sub-directories, **946 GeoTIFF files each** (matched across dirs by a
canonical id like `3001_BE_2023` — the loader strips the source prefix/suffix, so the differing
filename patterns below are expected and fine):

| sub-dir (under `$EMB`) | filename pattern | ~size | role / channels |
|---|---|---|---|
| `alphaearth_test_emb`    | `emb_<id>_quantized.tif`   | 5.7 GB  | dense, 64ch (int8, dequantised /127) |
| `tessera_test_emb`       | `<id>_merged.tif`          | 27.8 GB | dense, 128ch |
| `terramind_test_s1_emb`  | `s1_<id>_embeddings.tif`   | 0.75 GB | token, 768ch @16×16 |
| `terramind_test_s2_emb`  | `s2_<id>_embeddings.tif`   | 0.75 GB | token, 768ch @16×16 |
| `thor_test_s1_emb`       | `s1_<id>_embedding.tif`    | 0.9 GB  | token, 768ch @16×16 |
| `thor_test_s2_emb`       | `s2_<id>_embedding.tif`    | 0.9 GB  | token, 768ch @16×16 |

Dense sources concatenate to **192 ch** (`--in-channels 192`, `--dense-channels 64 128`); the four
token sources give **4×768** (`--token-channels 768 768 768 768`). All of this is already wired into
`run_full_inference.sh`.

Then run all 7 checkpoints on the 946 test tiles and re-assemble the submission end-to-end:

```bash
EMB=./emb/data/test  bash run_full_inference.sh
```

Requires: 1 GPU, the env above, and the ~37 GB test embeddings above. It writes
`models/<...>/predictions/` per model, then runs `assemble.py` →
`submissions/full_inference.zip` and verifies it against `final_submission/`.

**Verified (two levels):**
- *Assembly recipe* — running `assemble.py` on the **original** per-model prediction dirs rebuilds the
  submitted zip **byte-exact (max abs diff 0 over all 946 tiles)**: the combine step is exactly the 0.5463 recipe.
- *Full from-weights rerun* — a fresh GPU run of all 7 checkpoints (written to `submissions/full_inference.zip`)
  reproduces the submission at **fp level**, not bit-for-bit: building channel identical, only **8 / 186M
  seg pixels** flip, height differs by **≤0.14 m** (mean 0.00002 m). All IoU/RMSE and the **LB score are
  unchanged (0.5463)**. The residual is ordinary cuDNN nondeterminism, not a recipe difference — `assemble.py`
  reports this as `REPRODUCED`.

Inference settings that matter (already wired into `run_full_inference.sh`): correct `--model-type`
per checkpoint, **single view** (`--num-views 1`, no TTA), the ckpt (best/last) from the manifest,
`--dequantize-ae False`, `--token-upsample False`. TTA smoothing and model-best (for the overfit
runs) both *reduce* the test-memorisation these distilled models rely on.

---

## 5. Models were trained with two stages

Each of the three backbones (hrnet / convnext / mbconv) is produced in **two stages**. Full runnable
scripts + configs are under `training/` (see `training/TRAINING.md`).

**Stage 1 — supervised pretrain on the TRAIN set.** Standard supervised training on the labelled
train split (real IGN-LiDAR-derived GT: band0=bld, 1=veg, 2=water, 3=height), region-held-out val,
data augmentation ON, `adamw` LR 2e-4 / weight_decay 1e-4, poly LR schedule. This gives a general
per-architecture backbone. Config example: `training/configs/geofm/stage1_trainset_convnext.py`.

**Stage 2 — pseudo-label distillation on TRAIN+TEST mix.** A fresh model is trained **from scratch**
on a mixture of the train tiles and the 946 **test** tiles,
where the test tiles are **pseudo-labelled by the stage-1 model predictions** (fused across models/
sources into the teacher masks below). `pseudo_oversample=5`, `augment=False`, full-256 crop,
`weight_decay=0`, near-constant LR 1e-3, soft labels. `train_subsample_n` sets how much real train GT
stays in the mix; the shipped final configs use `train_subsample_n=0` (the test-pseudo extreme of the
mix) for maximal test-time calibration. Configs: `training/configs/geofm/*_overfit_*10k.py`.

Per family, the stage-1-derived teacher (fused predictions) + stage-2 loss:
- **building** teacher = building prediction fusion mask (LB iou_b 0.5222), loss `building_baseline` (α=β=0.5).
- **veg+water joint** teacher = merged veg+water prediction masks, loss `VegWaterBaselineLoss` =
  per-class `MAE(1.0/bg0.05) + 0.5·SSIM + 0.5·GradDiff + 2.0·Tversky(α=β=0.5)` on ch1 and ch2.
- **height** teacher = revFused height prediction, loss `bv_height_only` (smooth-L1, supervised only on bld∪veg).

Each shipped checkpoint's full stage-2 config is also kept alongside it at `models/<...>/config.py`.

We iteratively alternated pseudo-label generation and model training to reach the final score [1,2]. 

```text
[1] Ebel, P., El Baz, M., Wang, J., Xuan, W., Qi, H., Zheng, Z., ... & Meoni, G. (2026). Artificial Intelligence for Earthquake Response: Outcomes and insights from a global spaceborne rapid mapping challenge. IEEE Geoscience and Remote Sensing Magazine.
[2] Wang, J., Ma, A., Zhong, Y., Zheng, Z., & Zhang, L. (2022). Cross-sensor domain adaptation for high spatial resolution urban land-cover mapping: From airborne to spaceborne imagery. Remote Sensing of Environment, 277, 113058.
```

## Contact
If you have any questions, please contact me: kingdrone@edu.k.u-tokyo.ac.jp
