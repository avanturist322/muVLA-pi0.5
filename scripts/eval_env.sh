# Environment for simulator evaluation. Source it, do not execute:
#
#     source scripts/eval_env.sh mikasa    # or: libero
#
# Both simulators run inside pi05-mem's own .venv - there is no separate interpreter
# and no subprocess split. The variables below are the whole reason a naive run fails:
# ManiSkill needs an explicit Vulkan ICD and a writable PhysX cache, LIBERO needs EGL
# and a generated config file. Everything else was verified by scripts/probe_sim.py.

# Machine-specific locations. Override PI05_MEM_ROOT / MU_VLA_PATH / MIKASA_ROBO_PATH
# in your shell (or edit the defaults below) before sourcing this file.
PI05_MEM="${PI05_MEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MU_VLA="${MU_VLA_PATH:-/path/to/mu-vla}"
MIKASA_ROBO="${MIKASA_ROBO_PATH:-/path/to/MIKASA-Robo}"
export MU_VLA_PATH="$MU_VLA"

export HF_HOME="$PI05_MEM/.cache/hf"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PI05_MEM/src"

case "${1:-mikasa}" in
  mikasa)
    # Vulkan is present but has no default ICD registration in this image.
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
    # $HOME is ephemeral here, so PhysX's default cache path vanishes between jobs.
    export SAPIEN_PHYSX_CACHE_ROOT="$PI05_MEM/.cache/sapien/physx"
    mkdir -p "$SAPIEN_PHYSX_CACHE_ROOT"
    # Same reason, one directory further: ManiSkill looks for downloadable assets under
    # ~/.maniskill, which is that same ephemeral $HOME. Without this, every job decides
    # the `ycb` asset is missing and *prompts* to download it - and a prompt on a
    # non-TTY stdin is `EOFError: EOF when reading a line`, which killed all three
    # ShellGame envs in all three eval suites on 2026-07-26. The assets are downloaded
    # once to a persistent directory with:
    #   MS_ASSET_DIR=$PI05_MEM/.cache/maniskill \
    #       .venv/bin/python -m mani_skill.utils.download_asset ycb -y
    export MS_ASSET_DIR="$PI05_MEM/.cache/maniskill"
    # Second guard, and the one that actually addresses the crash rather than its cause:
    # `prompt_yes_no` (download_asset.py:19-23) returns True instead of calling input()
    # when this is "1". So a missing asset becomes a download attempt, not an EOFError
    # that takes the env process down. MS_ASSET_DIR should keep it from ever firing;
    # this is here so that a *future* asset gap costs a slow env, not a dead suite.
    export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
    # Third thing that hides in the ephemeral $HOME, and the one that actually killed the
    # 2026-07-26 07:26 suite: sapien.physx.enable_gpu() downloads libPhysXGpu_64.so (236 MB)
    # into `Path.home()/.sapien/physx/<ver>` - hardcoded to Path.home(), so
    # SAPIEN_PHYSX_CACHE_ROOT above does NOT cover it (that is the kernel cache, a
    # different thing). With N env processes starting at once they all see the file
    # missing, all download it, and the losers dlopen a half-written file:
    #   OSError: .../libPhysXGpu_64.so: file too short   (exit 1)
    # or a SIGBUS on the mmap                            (exit -7)
    # That took out 12 of 23 envs in config-a-nomem_rh-false. Seeding a symlink to an
    # NFS copy *before* any env starts removes the download, and with it the race.
    SAPIEN_PHYSX_VERSION="${SAPIEN_PHYSX_VERSION:-105.1-physx-5.3.1.patch0}"
    _physx_so="$PI05_MEM/.cache/sapien/physx-lib/$SAPIEN_PHYSX_VERSION/libPhysXGpu_64.so"
    if [ -s "$_physx_so" ]; then
        mkdir -p "$HOME/.sapien/physx/$SAPIEN_PHYSX_VERSION"
        ln -sfn "$_physx_so" "$HOME/.sapien/physx/$SAPIEN_PHYSX_VERSION/libPhysXGpu_64.so"
    else
        echo "eval_env.sh: WARNING no cached PhysX lib at $_physx_so;" \
             "envs will race to download it (see scripts/seed_physx.sh)" >&2
    fi
    unset _physx_so
    # mu-vla for experiments.robot.*, MIKASA-Robo for the env registrations.
    export PYTHONPATH="$PI05_MEM/src:$MU_VLA:$MIKASA_ROBO"
    ;;
  libero)
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    # LIBERO writes a config.yaml on first import; point it at the one already generated.
    export LIBERO_CONFIG_PATH="$MU_VLA/.cache/libero"
    export LIBERO_PATH="${LIBERO_PATH:-$MU_VLA/data/extern/LIBERO}"
    export PYTHONPATH="$PI05_MEM/src:$LIBERO_PATH"
    ;;
  *)
    echo "usage: source scripts/eval_env.sh [mikasa|libero]" >&2
    ;;
esac
