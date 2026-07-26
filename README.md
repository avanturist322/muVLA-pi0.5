# pi05-mem

**Recurrent memory for pi0.5.** This repository ports the memory mechanism of *mu-VLA*
(a fork of OpenVLA-OFT) onto LeRobot's pi0.5 policy, then trains and evaluates the result
on the memory-intensive robot tasks of MIKASA-Robo. The ported policy is called
**mu-VLA(pi0.5)** throughout.

If none of those three names mean anything yet, the 30-second version:

| name | what it is |
|---|---|
| **pi0.5** (pi0.5, [arXiv:2504.16054](https://arxiv.org/abs/2504.16054)) | a vision-language-action (VLA) model: a PaliGemma-2B vision-language *prefix* plus a Gemma-300M *action expert* that denoises a chunk of future actions by flow matching. The implementation used here is `lerobot.policies.pi05`. |
| **mu-VLA** | an OpenVLA-OFT fork that adds *recurrent memory tokens* to a VLA, so a policy can carry information across environment steps. |
| **MIKASA-Robo** ([site](https://mikasarobo.github.io/)) | a benchmark of 23 tabletop manipulation tasks on ManiSkill 3, built so that solving them *requires* memory. |

## The problem this repo exists to solve

pi0.5 is **memoryless**. Every call to the policy sees exactly one observation: the current
camera frames and the current robot state. Nothing carries over from the previous step.

That is fine for "pick up the red block" and fatal for `RememberColor3-VLA-v0`, where the
agent is shown a coloured cue for a few frames at the start of an episode, the cue then
disappears, and only afterwards must it reach for the cube of that colour. By the time the
decision has to be made, the evidence is no longer in the observation. A memoryless policy
can only guess.

`pi05-mem` gives pi0.5 a hidden state that survives across environment steps, using the
same mechanism, the same flag names and the same defaults as mu-VLA, so a configuration
that works in mu-VLA transfers here without translation.

## mu-VLA(pi0.5) vs plain pi0.5, in one table

| | plain pi0.5 | mu-VLA(pi0.5) = this repo |
|---|---|---|
| state across env steps | none | `num_mem_tokens` MEM vectors, carried step to step |
| prefix sequence | `images → language (+ state token)` | `images → MEM → language (+ state token)` |
| training data order | shuffled frames | `batch_size` time-coherent episode streams |
| backward pass | one step, one backward | TBPTT window of K steps, or EMA (below) |
| loss | flow matching over the action chunk | identical, untouched |
| inference | predict `n_action_steps` actions, execute all of them | receding horizon: predict a chunk, execute one action, re-query |
| extra weights | — | one `nn.Parameter` of shape `[num_mem_tokens, 2048]` |

Everything else — the flow-matching objective, the pretrained weights, the attention block
structure, the processor pipeline — is stock pi0.5. With `use_memory=false` this repo is
behaviourally identical to plain pi0.5, asserted bit-for-bit by
`tests/test_model_memory.py::test_disabled_memory_matches_the_stock_prefix`. That
equivalence is what makes the A/B comparison meaningful.

## Contents

- [How the memory actually works](#how-the-memory-actually-works)
- [Why a custom training loop and dataloader](#why-a-custom-training-loop-and-dataloader)
- [Installation](#installation)
- [Training](#training)
- [Memory-related parameters](#memory-related-parameters)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)

## How the memory actually works

### 1. Where the MEM tokens live

A learnable parameter `MemoryModule.initial_memory` of shape `[num_mem_tokens, 2048]`
(2048 = PaliGemma's width) is injected into the prefix **after the image embeddings and
before the language tokens**:

```
┌──────────── PREFIX (PaliGemma 2B, one bidirectional block) ─────────┐
│  img_0 … img_N  │  MEM_0 … MEM_{m-1}  │  lang_0 … lang_L           │
└─────────────────┴──────────┬──────────┴────────────────────────────┘
                             │ injected by embed_prefix_mem()
                             └─→ read back from prefix_out[:, mem_slice]
                                 × memory_write_scale  =  M_out

┌──────────── SUFFIX (Gemma 300M action expert) ──────────────────────┐
│  x_t: the noised action chunk (chunk_size × max_action_dim)         │
└─────────────────────────────────────────────────────────────────────┘
```

The position is not arbitrary. pi0.5 encodes the robot's proprioceptive state as a
*discrete token inside the text prompt* (256 bins over quantile-normalised state), so
"after vision, before language" is the same slot mu-VLA used (`vision/proprio → MEM → text`)
and it keeps MEM ahead of the state token.

After the forward pass the hidden states at those same positions are read back off
`prefix_out`, scaled, and become the MEM *input embeddings* of the next environment step.
That is the entire recurrence. `PI05Pytorch` normally discards `prefix_out`; returning it
is the only change to the base model's data flow.

Relevant code: `modeling_pi05_mem.py::embed_prefix_mem` (injection) and `::_read_memory`
(read-back); `memory.py::MemoryModule`.

### 2. Why two scale factors exist that mu-VLA did not need

The loop *hidden state at step t → input embedding at step t+1* only closes if input and
output live at the same scale. In mu-VLA's Llama-2 backbone they did (both std ≈ 0.02).
In LeRobot's PaliGemma they do not: `pi_gemma.py` drops Gemma's `sqrt(width)` input
multiplier. Measured on `lerobot/pi05_base` by `scripts/probe_embed_scale.py`:

| tensor | std |
|---|---|
| `embed_image(...)` input embeddings | 4.16 |
| `embed_language_tokens(...)` input embeddings | 9.71 |
| prefix output hidden states | 1.19 |

Hence two parameters whose defaults differ from mu-VLA's:

* `--memory-write-scale 5.8` = mean input std / output std = `(4.16 + 9.71)/2 / 1.19`.
  Averaged over images and text because a MEM token is as much a resident of the prefix
  input space as either neighbour. Without the factor the state would shrink ~5.8× per
  step — a factor of 6000 over five steps, i.e. the memory would simply evaporate.
* `--memory-init-std 4.0` instead of mu-VLA's `0.02`, matching the *image* embedding scale
  (MEM sits next to the image tokens and carries perceptual, not symbolic, content). At
  0.02 the MEM tokens would be two orders of magnitude quieter than their neighbours on
  the first frame of an episode, and the model would effectively not see them.

Both numbers are **measured, not tuned**: a 200-step sweep found the loss insensitive to
`memory_write_scale` over 1.0–5.8. Re-measure both for any other backbone.

### 3. How gradient reaches the memory: TBPTT vs EMA

Training walks `batch_size` episode streams in lockstep, one environment step per
dataloader step (see [below](#why-a-custom-training-loop-and-dataloader)).
`--memory-update` decides how the state crosses those steps:

| `--memory-update` | behaviour |
|---|---|
| `tbptt` (default) | the autograd graph spans K = `--tbptt-length` environment steps. One backward over the accumulated loss, then `detach()` exactly at the window boundary. Gradient flows *through* the recurrence for K steps. |
| `ema` | backward every step; the next input is `alpha*M_out + (1-alpha)*M_in` with **both operands detached** (`--ema-alpha`). No gradient crosses a step boundary. This is the ablation answering "does cross-step credit assignment matter, or is merely carrying a state enough?" |

Wherever `is_first` is true (a new episode began in that batch slot) the state is reset to
`initial_memory` **with gradient**, which is the only path by which `initial_memory` is
trained at all. Priority inside `MemoryModule.reset_episodes` is
`is_first > detach > keep in graph`.

Two consequences that bite if ignored:

* **K multiplies the effective batch.** `train.py` computes
  `effective_accum_steps = effective_tbptt_length * grad_accumulation_steps` and
  `total_optimizer_steps = max_steps // effective_accum_steps`. So changing K alone changes
  both the effective batch size and the schedule length, and a K ablation must compensate
  with `--grad-accumulation-steps`. `jobs/train_config_c_mem_k2.sh` does exactly that:
  K=2 with accumulation 4, to match K=8 with accumulation 1.
* **Under DDP, `initial_memory` gets gradient only on ranks whose batch contained an
  episode start**, so its gradient is explicitly all-reduced
  (`_all_reduce_initial_memory_grad`), and its norm is logged as
  `initial_memory_grad_norm`.

### 4. The attention mask, and why `custom` builds nothing

mu-VLA's invariant is *the context must not attend to the action tokens*. The new memory
state is read off context positions; if the context could see the action tokens, the state
written to memory would contain the current step's action prediction — a quantity that
does not exist in that form at inference time.

In OpenVLA-OFT this had to be enforced with a hand-built additive 4D mask, because OFT
makes the whole multimodal sequence bidirectional, action tokens included. **In pi0.5 the
invariant holds for free.** pi0.5 builds its mask from a block-boundary vector (`att_masks`
plus `make_att_2d_masks`, copied from big_vision/openpi): a token may attend only to tokens
whose cumulative mask is ≤ its own. The whole prefix has cumulative mask 0 and the action
suffix has 1, so the prefix — MEM included — cannot see the suffix, and the suffix sees
everything.

| row \ column | IMG | MEM | LANG | ACT |
|---|---|---|---|---|
| IMG | ✓ | ✓ | ✓ | ✗ |
| MEM | ✓ | ✓ | ✓ | ✗ (`custom`) / ✓ (`full`) |
| LANG | ✓ | ✓ | ✓ | ✗ |
| ACT | ✓ | ✓ | ✓ | ✓ |

| `--attention-mask-mode` | behaviour |
|---|---|
| `custom` (default) | pi0.5's native block mask, unmodified. **Nothing is patched** — there is nothing to port, the target mask already is the pretrained one. The only mode compatible with prefix KV-caching at inference. |
| `full` | ablation: `_open_mem_rows_to_suffix` opens **only the MEM rows** onto the action suffix. The prefix stops being cacheable, so inference takes a joint, non-cached path (`_sample_actions_joint`), roughly `num_inference_steps`× more expensive. |

Note the deliberate asymmetry: mu-VLA's `full` opened the action tokens to the *entire*
context; here only the MEM rows are opened, because opening the whole prefix would throw
away the KV cache and destroy the pretrained block structure pi0.5 was trained with. Under
flow matching the suffix at `t=1` is pure noise anyway, so `full` is semantically weaker
here than it was in OFT. **Use `custom` unless you are deliberately running the ablation.**

### 5. Receding horizon at inference

Memory advances once per forward pass. LeRobot's `select_action` only calls
`predict_action_chunk` when its action queue drains — once every `n_action_steps` steps.
With `n_action_steps=8` the memory would advance 8× more slowly at evaluation than during
training, and the state distribution would go out of distribution immediately.

So `receding_horizon` (default `None` = on iff `use_memory=True`, mu-VLA's rule) requires
`n_action_steps == 1`: predict a chunk, execute only its first action, re-query — and
re-update memory — next step. `PI05MemPolicy.select_action` **raises** rather than warns if
that is violated. A silent fallback is more dangerous than a crash: the eval would finish,
print poor numbers, and read as "memory does not help" when in fact the protocol was broken.

The number of flow-matching denoising steps (`num_inference_steps=10`) does **not** affect
this: memory is read once from `prefix_out`, before denoising begins.

Because MIKASA episodes are 11–20 steps long, the policy uses `chunk_size=5`,
`n_action_steps=1`. The stock pi0.5 chunk of 50 is longer than an entire episode.

## Why a custom training loop and dataloader

Recurrent memory requires that slot `i` of batch `n` and slot `i` of batch `n+1` are two
consecutive frames of the *same* episode. Otherwise the state carried in `mem_state[i]`
belongs to a different trajectory. LeRobot's `LeRobotDataset` + `EpisodeAwareSampler`
shuffles frames and cannot provide that, so `lerobot-train` is not usable here — hence
`src/pi05_mem/train.py`.

`EpisodicLeRobotDataset` (`src/pi05_mem/episodic_dataset.py`) is an `IterableDataset`
holding `batch_size` independent infinite streams. Each stream picks an episode, walks it
frame by frame emitting `is_first` / `is_last`, then picks another. Items are yielded
round-robin, so `DataLoader(batch_size=B, shuffle=False, num_workers=0)` reassembles
exactly one item per stream per batch:

```
            stream 0   stream 1   stream 2   stream 3
batch 0    A/ep7/t0   B/ep2/t0   A/ep1/t0   C/ep4/t0
batch 1    A/ep7/t1   B/ep2/t1   A/ep1/t1   C/ep4/t1
batch 2    A/ep7/t2   B/ep2/t2   A/ep1/t2   C/ep4/t2   <- C/ep4 ends (is_last)
batch 3    A/ep7/t3   B/ep2/t3   A/ep1/t3   B/ep9/t0   <- is_first, memory reset
```

`shuffle=False` and `num_workers=0` are correctness requirements, not preferences:
`num_workers>0` raises, because each worker would replicate all B streams and `batch[i]`
would stop being stream `i`. Episode boundaries come from `meta/episodes/*.parquet`, not
from `frame_index == 0`, so a partially downloaded dataset yields real episodes instead of
fictitious ones. Each item also carries the action chunk padded by repeating the last
action past the end of an episode, plus an `action_is_pad` mask so the loss ignores it.

Full write-up: **[`docs/multitask-dataloader.md`](docs/multitask-dataloader.md)**.

## Installation

Every cache is pinned inside the project root, because on the machine this was built for
`$HOME` is ephemeral and the container overlay is small and wiped between jobs.

### 1. Python environment

```bash
bash setup_env.sh
```

This creates `.venv` (Python 3.12, via `uv`), installs `torch==2.7.1` /
`torchvision==0.22.1` from the cu126 index, clones LeRobot into `.vendor/lerobot` and
installs it **editable with the `[pi]` extra**. LeRobot has to come from source because
pi0.5 is subclassed here, not merely configured. On another machine, edit the `ROOT` and
`UV` paths at the top of the script.

> **Gotcha — no `pyproject.toml`.** `pi05-mem` itself is not a packaged project; it is
> used straight out of `src/`. Every command therefore needs
>
> ```bash
> export PYTHONPATH=$PWD/src HF_HOME=$PWD/.cache/hf
> ```
>
> `jobs/common.sh` (training) and `scripts/eval_env.sh` (evaluation) set this for you.

### 2. Base checkpoint and tokenizer

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/pi05_base'))"
.venv/bin/python scripts/fetch_tokenizer.py
```

> **Gotcha — the tokenizer is gated, and you do not need a token.** LeRobot's pi0.5
> processor hardcodes `google/paligemma-3b-pt-224`, a gated repo, so the pipeline cannot
> start without an HF token. `scripts/fetch_tokenizer.py` downloads the canonical public
> Big Vision SentencePiece model from `storage.googleapis.com`, uses its sha256 to verify
> an ungated community mirror, and copies the verified files into
> `assets/paligemma_tokenizer/`. `processor_pi05_mem.py` points the pipeline there.
> The checksum is not decoration: a substituted tokenizer shifts token ids, the prompt
> silently becomes different text, and the model solves a different task.
> (`download_ckpt.sh` fetches checkpoint and tokenizer in one go, but its tokenizer half
> goes to the gated repo — prefer the two commands above.)

### 3. Data

The MIKASA-Robo LeRobot-v3 datasets live at
[`mikasa-robo/mikasa-robo-vla-lerobot`](https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-lerobot).
Directory names are snake_case (`remember_color_3_vla_v0`); the corresponding gym env ids
are CamelCase (`RememberColor3-VLA-v0`).

```bash
bash download_data.sh remember_color_3_vla_v0          # several names at once is fine
.venv/bin/python scripts/predecode_videos.py --data data/remember_color_3_vla_v0
```

> **Gotcha — video decoding is broken in this container, so datasets are pre-decoded.**
> torchcodec 0.11.1 cannot load: one wheel wants `libavutil.so.57/58/59` while the image
> ships ffmpeg 4.4 (`.so.56`), and the `.so.56` build is compiled against a newer torch
> than 2.7.1. LeRobot's `get_safe_default_video_backend()` selects torchcodec as soon as it
> imports, so the stock video path is unusable. `scripts/predecode_videos.py` decodes each
> camera **once** with the ffmpeg CLI into `data/<task>/cache/<camera>.npy`, which the
> dataset then memory-maps; the original mp4s are untouched. For `remember_color_3_vla_v0`
> that is 3858 frames × 2 cameras = 379 MB — cheaper than any online decoder for episodic
> streaming, which touches frames in scattered order. **Any dataset with `dtype="video"`
> cameras must be pre-decoded before training.** Revisit the approach at hundreds of
> thousands of frames.

The five tasks used by the current experiment are exported as `$TRAIN_TASKS` in
`jobs/common.sh`: `shell_game_push_vla_v0`, `intercept_medium_vla_v0`,
`remember_color_5_vla_v0`, `take_it_back_vla_v0`, `remember_shape_and_color_3x3_vla_v0`.

### 4. Simulators (only needed for evaluation)

MIKASA-Robo runs on ManiSkill 3, in the **same process and the same venv** as the policy —
no separate interpreter, no IPC layer that could silently change a dtype or an axis order:

```bash
uv pip install --python .venv/bin/python mani_skill
```

Two more things must be on `PYTHONPATH`; both are checkouts, not pip packages:

* **MIKASA-Robo** itself, which registers the `*-VLA-v0` gym environments;
* **mu-VLA**, for `experiments.robot.mikasa_robo.mikasa_robo_utils`, from which the
  observation and action conventions are taken verbatim (`make_eval_env`,
  `get_mikasa_images`, `get_mikasa_proprio`, `get_language_instruction`).

`scripts/eval_env.sh` wires all of it. **Source it, do not execute it:**

```bash
source scripts/eval_env.sh mikasa     # or: libero
```

> **Gotchas — simulator environment.** These three are the whole reason a naive run fails:
> * `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` — Vulkan is present but has no
>   default ICD registration in this image.
> * `SAPIEN_PHYSX_CACHE_ROOT` pointed inside the project — `$HOME` is an ephemeral
>   `/home/user`, so PhysX's default cache path vanishes between jobs.
> * gymnasium 1.x removed `Wrapper.__getattr__`, which every MIKASA-Robo and ManiSkill
>   wrapper stack relies on (`AttributeError: ... has no attribute 'device'`).
>   `src/pi05_mem/eval/gym_compat.py` restores it in a few lines; every adapter applies it
>   before importing a simulator.

LIBERO support exists but is a smoke test only here. It additionally needs
`robosuite==1.4.0 bddl==1.0.1 easydict gym==0.25.2 future==0.18.2 imageio[ffmpeg]
cloudpickle` and **`mujoco==3.8.0`** — the single downgrade in the entire build, because
robosuite 1.4.0 fails against mujoco 3.10 with
`TypeError: mj_fullM(): incompatible function arguments`. `MUJOCO_GL=egl`,
`PYOPENGL_PLATFORM=egl` and `LIBERO_CONFIG_PATH` are set by `scripts/eval_env.sh libero`.

`scripts/probe_sim.py mikasa|libero` builds and renders one environment and is the fastest
way to confirm the simulator half of the install.

## Training

### Environment, once per shell

```bash
cd /home/jovyan/users/echerepanov/pi05-mem   # machine-specific: substitute your checkout

export PYTHONPATH=$PWD/src
export HF_HOME=$PWD/.cache/hf
export HF_HUB_OFFLINE=1              # base checkpoint and tokenizer are already on disk
export WANDB_MODE=disabled WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false  # otherwise every torchrun worker warns
```

`jobs/common.sh` exports exactly this, plus `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`
and `$TRAIN_TASKS`, for the cluster jobs; `source jobs/common.sh` after editing the `ROOT`
line at its top is the shorter route. Every absolute path below comes from the machine
this ran on and has to be substituted. Paths relative to the checkout root do not.

### Single-task: `RememberColor3-VLA-v0`

Dataset directory `data/remember_color_3_vla_v0` — 250 episodes, 3858 frames, two cameras,
pre-decoded to `.npy` first (see [Data](#3-data)). The hyperparameters are those of the
real 8×H100 runs; only the dataset list is narrowed to one task.

**mu-VLA(pi0.5) — memory on:**

```bash
.venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    -m pi05_mem.train \
    --data-root "$PWD/data" \
    --data remember_color_3_vla_v0 \
    --output runs/rc3-mem \
    --batch-size 4 \
    --action-horizon 5 \
    --max-steps 48000 \
    --learning-rate 2.5e-5 \
    --lr-schedule cosine \
    --lr-warmup-steps 300 \
    --lr-min-ratio 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --dtype bfloat16 \
    --gradient-checkpointing \
    --seed 42 \
    --use-memory \
    --num-mem-tokens 64 \
    --memory-update tbptt \
    --tbptt-length 8 \
    --attention-mask-mode custom \
    --memory-log-freq 100 \
    --memory-expensive-log-freq 1000 \
    --log-freq 20 \
    --save-freq 12000
```

**Plain pi0.5 — memory off, same budget:**

```bash
.venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    -m pi05_mem.train \
    --data-root "$PWD/data" \
    --data remember_color_3_vla_v0 \
    --output runs/rc3-nomem \
    --batch-size 4 \
    --grad-accumulation-steps 8 \
    --action-horizon 5 \
    --max-steps 48000 \
    --learning-rate 2.5e-5 \
    --lr-schedule cosine \
    --lr-warmup-steps 300 \
    --lr-min-ratio 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --dtype bfloat16 \
    --gradient-checkpointing \
    --seed 42 \
    --log-freq 20 \
    --save-freq 12000
```

The two differ in exactly two places. The baseline drops the memory flags (`--use-memory`,
`--num-mem-tokens`, `--memory-update`, `--tbptt-length`, `--attention-mask-mode` and the
two `--memory-*-log-freq`), and it adds `--grad-accumulation-steps 8`. That accumulation is
not cosmetic: with memory on, TBPTT K=8 already makes the optimizer step once per 8
dataloader steps, so both runs see 48000 × 4 × 8 = 1.53M frames, take 6000 optimizer steps
and have an effective batch of 256. Without it the baseline would take 48000 updates
against the memory run's 6000, and a difference in the eval numbers could be attributed to
that rather than to memory.

On a single GPU, replace `.venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 8`
with `.venv/bin/python`. The global batch drops from 32 to 4, so the same `--max-steps` is
now an eighth of the frames; scale it if the budget is supposed to match.

One caveat about the numbers above. `--max-steps` counts *dataloader* steps, not optimizer
steps, and 48000 of them over a 3858-frame dataset is roughly 400 passes. That budget was
chosen for the five-task mixture below, and **no full-length single-task run was executed
here** — the single-task runs that did happen were 200-step diagnostics (`runs/verify-*`,
`runs/ws-*`, `runs/diag-*`). Lower `--max-steps` accordingly for one small dataset.

### Multi-task: the five MIKASA-Robo training tasks

`--data` takes a comma-separated list — mu-VLA's `MIKASA_ENVS` equivalent — and
`--data-root` is optional sugar so the entries can be bare task names. The training
mixture of the experiment is the one exported as `$TRAIN_TASKS` by `jobs/common.sh`:

```bash
export TRAIN_TASKS=shell_game_push_vla_v0,intercept_medium_vla_v0,remember_color_5_vla_v0,take_it_back_vla_v0,remember_shape_and_color_3x3_vla_v0
```

The v3 campaign below shows the two invocations that ran on 8×H100.
(A third config B with K=8 was also submitted in parallel; only A and C are shown here.)
**v3 changes:** batch_size doubled to 64 per rank (global batch 512), action_horizon increased to 8.

**Plain pi0.5 — memory off (config A v3, the baseline):**

```bash
.venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    -m pi05_mem.train \
    --pretrained lerobot/pi05_base \
    --data-root "$PWD/data" \
    --data "$TRAIN_TASKS" \
    --output runs/config-a-nomem-v3 \
    --batch-size 64 \
    --grad-accumulation-steps 1 \
    --action-horizon 8 \
    --max-steps 15000 \
    --learning-rate 5e-5 \
    --lr-schedule cosine \
    --lr-warmup-steps 750 \
    --lr-min-ratio 0.1 \
    --max-grad-norm 1.0 \
    --dtype bfloat16 \
    --gradient-checkpointing \
    --freeze-vision-encoder \
    --seed 42 \
    --log-freq 20 \
    --save-freq 3750
```

**mu-VLA(pi0.5) — memory on, K=2 (config C v3):**

```bash
.venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    -m pi05_mem.train \
    --pretrained lerobot/pi05_base \
    --data-root "$PWD/data" \
    --data "$TRAIN_TASKS" \
    --output runs/config-c-mem-k2-v3 \
    --batch-size 64 \
    --grad-accumulation-steps 1 \
    --action-horizon 8 \
    --max-steps 30000 \
    --learning-rate 5e-5 \
    --lr-schedule cosine \
    --lr-warmup-steps 750 \
    --lr-min-ratio 0.1 \
    --max-grad-norm 1.0 \
    --dtype bfloat16 \
    --gradient-checkpointing \
    --freeze-vision-encoder \
    --seed 42 \
    --use-memory \
    --num-mem-tokens 64 \
    --memory-update tbptt \
    --tbptt-length 2 \
    --attention-mask-mode custom \
    --memory-write-scale 1.0 \
    --memory-init-std 0.02 \
    --log-freq 20 \
    --save-freq 7500
```

Both v3 runs consume the same 1.53M frames in exactly **15000 optimizer steps** at
an effective batch of 32 (A) and 64 (C) respectively. `runs/<name>/train_config.json` records the resolved
configuration of each, which is the thing to compare if in doubt.

`--dataset-weights` takes one comma-separated weight per `--data` entry. Both runs left it
at its default, so the five datasets are drawn uniformly.

Each stream draws its next *episode* from a dataset sampled by `--dataset-weights`
(default: uniform over datasets, not over frames — a 250-episode dataset and a 25-episode
one are drawn equally often). A batch therefore generally mixes tasks while each slot still
walks one trajectory at a time. A dataset switch is always an episode switch, so memory
resets there exactly as it does anywhere else; `summary.json` records the realised
`dataset_mix` so the actual frame proportions are visible rather than assumed.

All datasets must agree on camera keys, image shape, state dimension and action dimension —
one policy has one input signature — and this is checked twice, once from `meta/info.json`
before any weights are built and once from the loaded shards before any batch is produced.

For a mixture, normalization statistics for `observation.state` and `action` are recomputed
**exactly** from the pooled raw parquet rows of all roots and cached as
`_combined_stats_<hash>.json` next to the datasets (delete it to force a recompute). A
single `--data` keeps reading `meta/stats.json` verbatim. This matters more than it sounds:
pi0.5 discretizes the normalized state into 256 bins **inside the text prompt**, so wrong
quantiles are a corrupted prompt, not a mild rescaling.

### Multi-GPU

```bash
.venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 8 -m pi05_mem.train ...
```

`--batch-size` is streams **per rank**; the global batch is `batch_size × world_size`. Each
rank walks different episodes (the stream seed carries the rank), and the loop refuses to
train if two ranks turn out to be on the same trajectories
(`check_ranks_see_different_data`; the verdict is written to `<output>/ddp_check.json`).
Gradients are averaged manually just before the optimizer step rather than through
`DistributedDataParallel` — see `src/pi05_mem/distributed.py` for why.

### The real runs

The three configurations of the current experiment, budget-matched on every axis that is
not memory (1.53M frames, effective batch 256, 6000 optimizer steps, same seed, same LR
schedule, on 8×H100):

| script | what it runs |
|---|---|
| `jobs/train_config_a_nomem.sh` | **A** — pi0.5 **without** memory. The baseline. Uses the episodic loader anyway, for fairness, and `--grad-accumulation-steps 8`. |
| `jobs/train_config_b_mem.sh` | **B** — mu-VLA(pi0.5): `--num-mem-tokens 64`, TBPTT `--tbptt-length 8`, `custom` mask. K=8 already supplies the 8 microsteps per update, so no gradient accumulation. |
| `jobs/train_config_c_mem_k2.sh` | **C** — as B but `--tbptt-length 2` with `--grad-accumulation-steps 4`, so that **only** K differs from B. |

Read those three scripts before writing your own: they are the authoritative invocations,
and their header comments explain the budget matching in detail. Configs A and B are
reproduced verbatim under [Multi-task](#multi-task-the-five-mikasa-robo-training-tasks)
above; config C is B with `--tbptt-length 2 --grad-accumulation-steps 4` substituted.

On the cluster the runs were submitted as cryri jobs rather than started interactively.
The `jobs/*.yaml` next to the scripts are the descriptors — image, region `SR008`,
instance `a100plus.8gpu.80vG.96C.1456G`, one worker, `medium` priority — and the submit
command was

```bash
/home/jovyan/users/echerepanov/cry jobs/train_config_b_mem.yaml    # machine-specific path
```

where `cry` is the site's cryri wrapper and lives outside this repository. Nothing in the
training code needs it: each yaml's `command:` is just `bash jobs/train_config_<x>.sh`, and
running that script on any 8-GPU node does the same work. The scripts hardcode
`ROOT=/home/jovyan/users/echerepanov/pi05-mem` through `jobs/common.sh`; that one line is
what has to change on another machine.

### v2: stabilized memory hyperparameters (configs A/B/C v2)

The first campaign (configs A/B/C above, `memory_write_scale=5.8`, `memory_init_std=4.0`)
trained, but the memory runs were not healthy: `mem_in_norm` settles at
`~101 * memory_write_scale` regardless of `memory_init_std` (measured with
`scripts/probe_table.py`), so at `ws=5.8` the memory readout sits at norm ~563 against
~188/~439 for the image/language embeddings. RMSNorm's backward attenuates gradient into
a token as `1/||x||`, so the largest token in the prefix gets the least gradient:
`initial_memory` moved `|dx|/|x| = 0.05%` over 24k microsteps (`||x||=1445.88` at both
step 12000 and step 36000, versus its `4.0*sqrt(64*2048)=1448` initialization), and 97% of
optimizer steps hit `--max-grad-norm 1.0` with a clip factor varying 1.15-6.4x, so the
configured LR schedule was not the one actually applied. `--tbptt-length 8` and `2` were
statistically indistinguishable (loss 0.0375 vs 0.0389) - consistent with a memory state
that carries variance but does not receive a training signal to *use* the K-step credit
assignment. Full measurement: `projects/pi05-mem/JOURNAL.md`, entry "почему память не
учится: разбор по кривым обучения".

v2 changes four things, all existing knobs, no code changes to the write rule itself:

| change | v1 | v2 | why |
|---|---|---|---|
| `--memory-write-scale` | `5.8` | `1.0` | brings the memory operating point down to `~101`, in range with the other prefix tokens, so RMSNorm backward no longer strangles its gradient |
| `--memory-init-std` | `4.0` | `0.02` | matches the lower operating point from step 0 instead of traveling `181 -> 563` over the first ~1000 steps |
| `--freeze-vision-encoder` | off | on | removes one source of competing gradient while the memory pathway is being re-validated at the new scale |
| `--learning-rate` | `2.5e-5` | `5e-5` | |

A controlled 200-step pair on the v1 code path already pointed this way before v2 was
run: `diag-quiet` (`ws=1.0, init=0.02`) against `diag-k1` (`ws=5.8, init=4.0`, otherwise
identical) gave loss -6.3%, grad-norm p50 -18%, p90 -30%. That pair used 4 memory tokens
and one task, not the full 64-token five-task mixture, so it is directional evidence, not
a magnitude guarantee - which is what the v2 runs below exist to check at full scale.

`--freeze-vision-encoder` was not a flag before this campaign: `PI05Config` already had
the field (`freeze_vision_encoder`, acted on by
`PaliGemmaWithExpertModel._set_requires_grad`), but `train.py` never surfaced it. It now
does, the same way `--gradient-checkpointing` does - set on `policy_config` before
`make_policy` builds the model, since the flag is only read once, in
`PaliGemmaWithExpertModel.__init__`.

v2 also targets a fixed **optimizer-step** count (15000) rather than a fixed frame count,
in preparation for matching mu-VLA(openvla-oft)'s update count in a later campaign.
`--grad-accumulation-steps 1` on all three configs is deliberate here, not an oversight:
unlike A/B/C v1, the v2 configs are **not** matched to each other on samples-per-update
(K changes the effective batch: A sees 32 samples/update, C 64, B 256). Matching update
*counts* to openvla-oft's 92500 optimizer steps, and separately samples-per-update across
A/B/C, is future work - see `train.py`'s `effective_accum_steps = effective_tbptt_length *
grad_accumulation_steps` and `total_optimizer_steps = max_steps // effective_accum_steps`
for the arithmetic a future campaign has to use.

| script | K | `--max-steps` (microsteps) | optimizer steps | samples/update |
|---|---|---|---|---|
| `jobs/train_config_a_nomem_v2.sh` | - (no memory) | 15000 | 15000 | 32 |
| `jobs/train_config_b_mem_v2.sh` | 8 | 120000 | 15000 | 256 |
| `jobs/train_config_c_mem_k2_v2.sh` | 2 | 30000 | 15000 | 64 |

All three: `--pretrained lerobot/pi05_base`, `--learning-rate 5e-5`,
`--lr-warmup-steps 750` (5% of the 15000 optimizer steps, same ratio as v1's
`300/6000`), `--lr-min-ratio 0.1`, `--freeze-vision-encoder`, `--seed 42`; B and C also
carry `--memory-write-scale 1.0 --memory-init-std 0.02`. Submitted the same way as v1:

```bash
/home/jovyan/users/echerepanov/cry jobs/train_config_a_nomem_v2.yaml
/home/jovyan/users/echerepanov/cry jobs/train_config_b_mem_v2.yaml
/home/jovyan/users/echerepanov/cry jobs/train_config_c_mem_k2_v2.yaml
```

### Other flags worth knowing

`--pretrained` (default `lerobot/pi05_base`; `none` trains from scratch), `--batch-size`,
`--action-horizon` (= policy `chunk_size`), `--max-steps` (dataloader steps, not optimizer
steps), `--grad-accumulation-steps`, `--learning-rate` (2.5e-5), `--lr-schedule
{cosine,multistep,constant}`, `--lr-warmup-steps` (counted in *optimizer* steps),
`--lr-min-ratio`, `--num-steps-before-decay` (multistep only), `--max-grad-norm`,
`--adam-beta1` / `--adam-beta2` / `--adam-eps`, `--weight-decay`, `--dtype`, `--device`,
`--gradient-checkpointing`, `--freeze-vision-encoder` (SigLIP vision tower, off by
default; used by the v2 configs below), `--max-episode-steps`, `--seed`, `--log-freq`,
`--save-freq`.
`python -m pi05_mem.train --help` prints the authoritative list.

## Memory-related parameters

Each of these is a CLI flag of `pi05_mem.train` **and** a field of `PI05MemConfig`
(`src/pi05_mem/configuration_pi05_mem.py`, which carries the same explanations at greater
length). The first six keep mu-VLA's names, defaults and semantics unchanged; the rest are
new. All of them are written into `memory_meta.json` next to the checkpoint, so evaluation
configures itself and cannot silently disagree with training.

| flag | default | what it does |
|---|---|---|
| `--use-memory` | off | Master switch. Off ⇒ no MEM tokens, no memory module, prefix bit-identical to stock pi0.5, every other memory flag inert. |
| `--num-mem-tokens N` | `4` | How many MEM vectors are injected. mu-VLA used 64, and so do configs B and C; the default of 4 was chosen for cost, not for quality. Cost is linear in added prefix length. |
| `--memory-update {tbptt,ema}` | `tbptt` | Cross-step update rule. `tbptt` keeps the autograd graph for K steps and detaches at the boundary; `ema` builds no cross-step graph at all. |
| `--tbptt-length K` | `2` | TBPTT truncation length: gradient flows through K environment steps. Ignored under `ema`. **Also sets the number of microsteps per optimizer step** — see `effective_accum_steps` above; a K sweep must compensate with `--grad-accumulation-steps` or it silently changes the effective batch and the schedule length too. |
| `--ema-alpha A` | `0.1` | Only under `--memory-update ema`: `M_in[t+1] = A*M_out[t] + (1-A)*M_in[t]`, both operands detached. `A=1` reproduces TBPTT with K=1; `A=0` freezes memory. |
| `--attention-mask-mode {custom,full}` | `custom` | `custom` = pi0.5's native block mask, nothing patched, prefix KV cache usable. `full` = ablation opening the MEM rows onto the action suffix, which disables the prefix cache and makes inference ~`num_inference_steps`× slower. |
| `--memory-write-scale S` | `5.8` | Multiplier on the MEM readout before it becomes the next step's input embedding. Compensates for PaliGemma's missing `sqrt(width)` input multiplier. `1.0` gives verbatim mu-VLA feedback. Re-measure with `scripts/probe_embed_scale.py` for a different backbone. |
| `--memory-init-std σ` | `4.0` | Std of the learnable `initial_memory`. mu-VLA used `0.02` (Llama input-embedding scale); `4.0` matches PaliGemma's image-embedding scale. |
| `--memory-log-freq N` | `0` (off) | Cheap memory diagnostics every N gradient steps: memory norms, `initial_memory_grad_norm`. The real jobs use 100. |
| `--memory-expensive-log-freq N` | `0` (off) | Expensive memory diagnostics every N gradient steps: per-token drift, attention mass. The real jobs use 1000. |

Three more settings are memory-critical but belong to the policy config rather than to
`train.py`'s CLI (`src/pi05_mem/factory.py::make_config`):

| config field | value here | why |
|---|---|---|
| `n_action_steps` | `1` | Forced by receding horizon: one env step = one forward = one memory update. `select_action` **raises** if this is > 1 while memory is on. Do not change it while memory is enabled — it is not a setting, it is a bug, and the code treats it as one. |
| `chunk_size` | `5`, set by `--action-horizon` | Stock pi0.5 predicts 50 actions; MIKASA episodes are 11–20 steps and only the first action is ever executed, so 50 would be 10× the attention cost for nothing. Freely tunable. |
| `receding_horizon` | `None` = auto | On iff `use_memory` is on (mu-VLA's rule). Not a training flag; the eval entry points can override it per run with `--receding-horizon {auto,true,false}`. |

**Safe-to-tune summary.** `use_memory`, `num_mem_tokens`, `memory_update`, `ema_alpha`,
`tbptt_length` and `attention_mask_mode` are mu-VLA-compatible and free to sweep.
`memory_write_scale` and `memory_init_std` are *measured* for PaliGemma and must be
re-measured for another backbone. `n_action_steps > 1` with memory is forbidden.

### What a training run writes

Into `--output`:

| path | contents |
|---|---|
| `metrics.jsonl` | one line per dataloader step: loss, LR, `is_first` count, per-dataset frame counts, plus memory norms and `initial_memory_grad_norm` when `--memory-log-freq` is on. Rank 0 only; other ranks write `metrics-rank<N>.jsonl`. |
| `train_config.json` | the fully resolved configuration, world size, global batch size. |
| `summary.json` | final losses, timing, `dataset_mix` (frames actually seen per dataset), the DDP check report. |
| `ddp_check.json` | evidence that the ranks walked different trajectories. |
| `step-NNNNNN/`, `final/` | checkpoints; `--save-freq` controls the intermediates. |

There is **no wandb** — it is explicitly disabled in `jobs/common.sh`, and metrics go to
stdout and to `metrics.jsonl`. `python scripts/inspect_metrics.py runs/<name>` prints a
quick read-out: number of steps, episode resets, memory norms, on how many logged steps
`initial_memory` actually received gradient, and first-10 vs last-10 loss.

### Checkpoints

`<output>/final/` holds the LeRobot policy plus two extra files:

* **`memory_module.pt`** — the `initial_memory` weights, saved separately because LeRobot's
  `save_pretrained()` remaps state-dict keys and **drops** it. Without this the checkpoint
  would still look valid, load without error, and evaluate randomly initialised memory. It
  is loaded back with `strict=True`: better to fail at eval startup than to publish numbers
  indistinguishable from "memory does not help".
* **`memory_meta.json`** — every memory flag, including `receding_horizon`.
  `detect_memory_config(dir)` reads it back, in order: meta file → state-dict probe → "no
  memory". Evaluation is therefore never told the memory settings by hand.

## Evaluation

Memory settings are **not** evaluation CLI flags — they are read from the checkpoint's
`memory_meta.json`. What you choose is the environment, the number of trials, and the
inference regime.

### One environment

```bash
source jobs/common.sh                 # $ROOT, $PY, $TRAIN_TASKS, caches, no-wandb
source scripts/eval_env.sh mikasa     # Vulkan ICD, PhysX cache, PYTHONPATH

"$PY" -m pi05_mem.eval.run_eval \
    --env mikasa --env-id RememberColor3-VLA-v0 \
    --checkpoint runs/config-b-mem/final \
    --data-root data --data "$TRAIN_TASKS" \
    --episodes 100 --seed 4242424242 \
    --receding-horizon auto \
    --output eval_results/_scratch/result.json \
    --video-dir eval_results/_scratch/videos --max-videos 5
```

`--data` must name **the same mixture the checkpoint was trained on**. It is what fixes the
normalization quantiles, and pi0.5 writes the normalized state into the text prompt, so a
different mixture is literally a differently-conditioned policy.

`--receding-horizon {auto,true,false}`: `auto` = on iff the checkpoint has memory (this
sets `n_action_steps=1`); `false` sets `n_action_steps = chunk_size` and is the ablation
that separates "memory helps" from "re-querying every step helps".

Other flags: `--max-steps` (override the env's episode limit), `--dtype`, `--device`,
`--no-resume`, and for LIBERO `--env libero --task-suite --task-id --no-flip-images`.
`--max-videos 0` records none, `-1` records all.

**Per-episode artefacts**, written as they happen so a job killed at episode 97 still keeps
96 episodes of evidence:

* `episodes.jsonl` — one JSON line per episode: success, steps, reward, `forward_calls`,
  `mem_updates`, `mem_delta_max` / `mem_delta_min`, `n_action_steps`, error, video path.
  This is the per-episode statistic the experiment report is built from, and what a resumed
  run picks up.
* `run_meta.json` — checkpoint, seed, dtype, data mixture, inference regime, `max_steps`. A
  resume is accepted only if all of this matches, so a directory reused for a different
  checkpoint costs a re-run instead of contaminating the numbers.
* `result.json` — `{"summary": ..., "episodes": [...]}`, written atomically.
* `videos/episode_NNN.mp4` — all camera views side by side, for the first `--max-videos`
  episodes.

**Three invariants are checked numerically** and reduced to `all_consistent` in the summary:

1. `forward_calls == steps` — the model really is queried on every environment step;
2. `mem_updates ∈ {0, steps}` — memory updates always or never, never partially;
3. `mem_delta_max > 0` — the state actually moves.

A frozen memory is otherwise indistinguishable from a memory that does not help, so the run
**fails** (a dedicated guard exit code) rather than reporting a plausible-looking number.
The same is true if videos were requested and none could be written.

### The full 23-task suite

```bash
bash jobs/eval_suite.sh <run-name> <true|false> [num-trials]
# e.g.
bash jobs/eval_suite.sh config-b-mem true 100
bash jobs/eval_suite.sh config-a-nomem false 100
```

`<run-name>` is a directory under `runs/`. The checkpoint is `runs/<name>/final` and
nothing else: quietly falling back to the newest `step-*` would produce a table that looks
like a controlled comparison but compares two configurations at different training budgets.
To evaluate an intermediate checkpoint on purpose, name it —
`EVAL_CHECKPOINT=$ROOT/runs/<name>/step-004000 bash jobs/eval_suite.sh ...` — which also
gives it a `_`-prefixed output tag so it stays out of the final table. Results land in
`eval_results/<run-name>_rh-<true|false>/`.

Underneath it is `python -m pi05_mem.eval.suite`, which:

* runs the 23 MIKASA tasks of mu-VLA's protocol (100 trials each, seed 4242424242) —
  5 of them are the training tasks and **18 are held out**, which is where a memory
  mechanism has to earn its keep;
* runs **one subprocess per GPU**: ManiSkill builds a Vulkan context per process and dies
  on it often enough that a crash should cost one environment, not the whole suite;
* **resumes** — an environment with a valid `result.json` is skipped, but only if that
  result answers the same question (same checkpoint, seed, dtype, data mixture, trial
  count, video request);
* reclaims a wedged environment after `--env-timeout` (the job script passes 7200 s, a
  measured value: the longest task's 100 episodes take ~32 min);
* refuses to guess how many GPUs it has — pass `--gpus 0,1,...` if enumeration fails,
  rather than silently serialising 23 environments onto one device.

Useful flags: `--envs` (comma-separated subset), `--num-trials`, `--videos-per-env`
(default 5), `--receding-horizon`, `--max-steps`, `--overwrite` (re-run environments that
already have a result — this also forces `--no-resume` downstream, so it really re-measures).

Output per suite:
`<tag>/<Env-Id>/{result.json,episodes.jsonl,run_meta.json,run.log,videos/}` plus
`<tag>/summary.csv` and `<tag>/suite_config.json`.

### Comparing runs

```bash
python scripts/build_summary.py \
    --results eval_results \
    --out eval_results/comparison.csv \
    --baseline config-a-nomem_rh-false
```

One row per task, one column group per run, plus the memory-minus-baseline delta and a
`delta_z` — the delta over the standard error of the delta — so a reader can tell signal
from binomial noise. At n=100 the standard error of a single success rate is about 5
points, so a 3-point difference between two runs is not a result. Rows are split into the
5 training tasks and the 18 held-out ones. Means are **complete-case**: a task enters a
mean only if every run scored it, and the dropped ones are named in `notes`; a crashed
environment is left blank rather than recorded as 0.0. Directories whose name starts with
`_` are treated as scratch and excluded. `scripts/render_report_results.py` turns the CSV
into the markdown tables of the experiment report, so the report and the CSV cannot drift
apart.

### Troubleshooting: the two caches an ephemeral `$HOME` takes with it

Both failures below hit the evaluation suites of 2026-07-26, and both have one root cause:
inside a cryri job `$HOME` is an ephemeral `/home/user`, so anything a library caches under
`~` is missing again on the next job. `scripts/eval_env.sh mikasa` handles both; read this
if you run the simulator without it, or on another machine.

**`EOFError: EOF when reading a line`, and the three `ShellGame*-VLA-v0` environments die
at startup.** They are registered with `asset_download_ids=["ycb"]` and read
`ASSET_DIR/assets/mani_skill2_ycb/info_pick_v0.json`; ManiSkill looks under `~/.maniskill`,
does not find the asset, and calls `prompt_yes_no` — that is, `input()` — on a non-TTY
stdin. `eval_env.sh` exports `MS_ASSET_DIR="$PI05_MEM/.cache/maniskill"` so the asset lives
on NFS instead, and `MS_SKIP_ASSET_DOWNLOAD_PROMPT=1`, which makes `prompt_yes_no` return
`True` instead of reading stdin — so a *future* asset gap costs a slow environment rather
than a dead suite. Fetch the asset once, ahead of time:

```bash
MS_ASSET_DIR=$PWD/.cache/maniskill \
    .venv/bin/python -m mani_skill.utils.download_asset ycb -y
# -> .cache/maniskill/data/assets/mani_skill2_ycb, 86 MB in 470 files
```

> **Gotcha — `MS_ASSET_DIR` changed meaning between ManiSkill versions.** In `mani_skill`
> 3.0.1 (this repo's venv) `mani_skill/__init__.py` reads
> `ASSET_DIR = Path(os.path.join(os.getenv("MS_ASSET_DIR", "~/.maniskill"), "data"))` —
> the code appends `data` itself, so `MS_ASSET_DIR` must point at the **parent** of `data`,
> with no `/data` suffix. In 3.0.0b15 that same variable *was* the `data` directory. An
> `MS_ASSET_DIR=.../maniskill/data` line copied from an older setup therefore looks right
> and searches one level too deep, and it surfaces as the `EOFError` above rather than as a
> missing-path error.

**`OSError: ... libPhysXGpu_64.so: file too short` (exit 1) or `Bus error` (exit -7), on
many environments at once.** `sapien.physx.enable_gpu()` lazily downloads that 236 MB
library into `Path.home()/.sapien/physx/<version>`. The path is hardcoded to `Path.home()`,
so `SAPIEN_PHYSX_CACHE_ROOT` does **not** redirect it — that variable is the PhysX *kernel*
cache, a different thing. The suite starts one process per GPU at the same moment; all of
them find the library missing, all of them download it to the same path, and the ones that
lose the race `dlopen` a partially written file. That destroyed 12 of 23 environments in
`config-a-nomem_rh-false`. Populate an NFS copy once, from a single process, before any
suite:

```bash
bash scripts/seed_physx.sh    # idempotent: a no-op once the cached file is complete
```

`scripts/eval_env.sh mikasa` then symlinks `$HOME/.sapien/physx/<version>` at that copy
before any environment starts, which takes the download off the hot path and the race with
it; it warns loudly if the cache is empty. The script asks the installed sapien for its
version rather than hardcoding one, because both the download URL and the directory name
derive from it.

### Troubleshooting: `ShellGame*` scores exactly 0.00 and the cups are missing from the video

Third failure of the same three environments, unrelated to the two above, and the nastiest
of them: nothing crashes, nothing warns, and the suite reports a success rate. It is only
wrong.

MIKASA-Robo hides the cups during the cue phase by writing `z += HEIGHT_OFFSET = 1000` in
`evaluate()`. That is meant to stay **render-only**: `Actor.pose = ...` writes to
`px.cuda_rigid_body_data` and `_step_action` never calls `gpu_apply`, so PhysX never learns
about the fake `z`. But `ShellGamePush` and `ShellGamePick` also register `goal_site` in
`self._hidden_objects`, and `Actor.hide_visual()` performs a **global**
`gpu_apply_rigid_dynamic_data()` — which commits the cups' fake `z` into the simulation.
From there they free-fall: `vz` goes −0.491, −0.981, −1.471 … = `g·dt` at `dt = 0.05`, and
by step 20 the cups are 4.9 m below the table. The dense reward `1 - tanh(10d)` underflows
to exactly `0.0` in float32, so the task is unwinnable and the video shows an empty table.
`ShellGameTouch` registers no hidden objects and is unaffected — that is the only difference
between the three environments, their cue logic is identical.

What puts the scene back is `render()`: at `render_mode="rgb_array"` it calls
`show_visual()` over the same list (`sapien_env.py:1373-1374`). So the contract is stated in
`make_eval_env` itself — *"the eval loop calls `env.render()` itself once per simulator
step"* (`mikasa_robo_utils.py:399`) — and mu-VLA's loop honours it
(`run_mikasa_robo_eval.py:750`).

**If you write your own rollout loop against MIKASA-Robo, call `env.render()` after every
`env.step()`, unconditionally** — not only on the episodes you are recording, or the recorded
ones get different physics from the rest. `eval/rollout.py` here does exactly that;
`tests/test_rollout_render_contract.py` pins it (render happens once per step, after the
step, whether or not a video is being written), and `my_tmp/test_shellgame_render.py` is the
GPU regression, with the negative control that without rendering Push/Pick *must* fail — a
test that only checks the fixed path would prove nothing.

Symptom-to-cause table for the three ShellGame failures, since they look alike from the
outside:

| symptom | cause | fix |
|---|---|---|
| `EOFError` at env creation, suite dies | ycb asset missing under an ephemeral `$HOME` | `MS_ASSET_DIR` on NFS + `MS_SKIP_ASSET_DOWNLOAD_PROMPT=1` |
| `libPhysXGpu_64.so: file too short`, `Bus error`, many envs at once | parallel processes racing to download into `Path.home()` | `scripts/seed_physx.sh` + symlink in `eval_env.sh` |
| success rate exactly 0.00, cups absent from the video, no error at all | eval loop never called `env.render()` | render after every step, unconditionally |

Note that your **training data is not affected** by the third one even if you collected it
with the same environments: demonstration collection runs with `num_envs=10`, and
`reconfiguration_freq` is 0 when `num_envs > 1`, so `goal_site` is hidden once instead of
per episode and the bad global apply does not recur.

## Tests

```bash
export PYTHONPATH=$PWD/src
.venv/bin/python -m pytest tests -q                # CPU suite
.venv/bin/python -m pytest tests -q --run-slow     # + GPU tests that build the real model
```

Without `--run-slow` the GPU tests are skipped. The GPU ones are the tests that matter,
because they pin invariants rather than "the code runs":

* MEM tokens land between the image and the language embeddings, at the expected prefix
  positions;
* the read-back state is on the same scale as the image embeddings;
* memory actually changes the prediction (it is not inert ballast);
* gradient reaches `initial_memory`, flows across a K=2 TBPTT window, and is genuinely
  blocked by the detach;
* the `custom` and `full` masks differ in exactly the MEM rows, and `full` does not mutate
  the input mask;
* recurrent inference carries memory within an episode and resets it between episodes;
* `select_action` refuses a multi-step action queue while memory is on;
* with `use_memory=False` the prefix reproduces stock `PI05Pytorch.embed_prefix` exactly.

On the CPU side, `tests/test_episodic_dataset.py` and `tests/test_multitask_dataset.py`
pin stream time-coherence, episode-boundary handling, action-chunk padding, mixture
weights and the pooled statistics; `tests/test_eval_suite*.py` and
`tests/test_rollout_accounting.py` pin the evaluation guards and the resume logic;
`tests/test_build_summary.py` pins the complete-case aggregation.

## Repository layout

```
src/pi05_mem/
  configuration_pi05_mem.py   PI05MemConfig: every memory parameter, with its rationale
  memory.py                   MemoryModule: initial_memory, reset_episodes, ema_update
  modeling_pi05_mem.py        PI05MemPytorch / PI05MemPolicy: injection, read-back,
                              mask modes, sample_actions_mem, the select_action guard
  processor_pi05_mem.py       the stock pi0.5 processor with a local tokenizer path
  episodic_dataset.py         B time-coherent streams over N datasets
  shard.py                    one dataset root: rows, episode bounds, tasks, build_item
  frame_sources.py            .npy memmap (video datasets) / parquet PNG (image datasets)
  dataset_stats.py            per-root statistics; exact pooled quantiles for a mixture
  factory.py                  build config + policy + processors from datasets
  memory_meta.py              memory_meta.json / memory_module.pt, detect_memory_config
  distributed.py              manual gradient sync, DDP data-divergence audit
  train.py                    the training loop (TBPTT / EMA) and its CLI
  eval/
    run_eval.py               one env, N episodes, per-episode log, videos, guards
    suite.py                  23 envs in parallel, resumable, summary.csv
    envs.py                   MIKASA / LIBERO adapters and their conventions
    rollout.py                one episode, with the memory accounting
    loader.py                 load a checkpoint together with its memory config
    gym_compat.py             the gymnasium 1.x Wrapper.__getattr__ shim
jobs/           the invocations actually used: train A/B/C, eval suite, cryri yamls
scripts/        setup, probes, diagnostics, summary building
tests/          pytest suite (CPU by default; --run-slow adds the GPU tests)
docs/           long-form documentation
assets/         the verified PaliGemma tokenizer
data/           LeRobot v3 datasets (+ cache/*.npy pre-decoded frames)
runs/           training outputs
eval_results/   evaluation outputs, one directory per (run, inference regime)
```

## Where the deeper documentation lives

* **[`docs/multitask-dataloader.md`](docs/multitask-dataloader.md)** — the streaming
  dataloader, multi-task mixing, pooled normalization statistics, and the test that pins
  each property.
* **The module docstrings are load-bearing.** `configuration_pi05_mem.py` documents every
  parameter and why its default is what it is; `suite.py`, `run_eval.py` and
  `build_summary.py` document every guard and the failure it exists to prevent.
* Four long-form reports (in Russian) were written for this project and live with the
  maintainer's project notes rather than in the repository: the architecture and the list
  of modifications; every new parameter with its motivation; the MIKASA experiment
  protocol; and the multi-task dataloader. Ask for them if you need the reasoning behind a
  design choice that the code comments do not already cover.

## Known limitations

* `memory_write_scale` and `memory_init_std` are measured, not tuned. A 200-step sweep
  found the loss insensitive to `write_scale` over 1.0–5.8, so treat 5.8 as a starting
  point with the right order of magnitude, not an optimum.
* The default `num_mem_tokens=4` is a cost choice, not a quality one; the experiment runs
  use 64, and the ablation has not been done.
* `attention_mask_mode=full` is expensive at inference and semantically weaker under flow
  matching than the corresponding mode was in OpenVLA-OFT. It exists for ablation parity.
* MIKASA episodes are 11–20 steps at fps=10, i.e. 1–2 seconds. The memory horizon that can
  be demonstrated on this data is bounded by that.
* The bundled LIBERO dataset is about 3% downloaded (14 complete episodes) and is a smoke
  test only. Its `q01`/`q99` were computed from that 3%; re-run
  `scripts/augment_quantile_stats.py --force` if more of it is ever fetched.
* **No success rates are quoted anywhere in this README, deliberately.** The evaluation
  suites were still running when it was written. Read them out of
  `eval_results/comparison.csv` once they finish.
