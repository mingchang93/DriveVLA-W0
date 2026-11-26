#!/bin/bash
# DTS full-chain preset: 132 quantized Linear layers, refresh interval 5, seed 42.
#
# Same recipe as pt_dts132_i200_s42.sh except the DTS refresh interval (5 instead of 200).
#
# 132 = 28 language-model layers x 6 projections
#       - 6 excluded layers (0, 1, 2, 25, 26, 27) x 6 projections
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export TZ=Asia/Shanghai
export TAG=dts132_i5_s42
export SEED=42

export HIF8_PT=dts
export HIF8_SFT=dts
export HIF8_INTERVAL=5
export IG_PT='lm_head,o_proj,layers.0.,layers.1.,layers.2.,layers.25.,layers.26.,layers.27.'
export IG_SFT="$IG_PT"

# Pin the launcher's DTS recipe and checkpoint cadence to the 196-layer baseline.
export SAVE_STEPS_PT=1000
export HIF8_DTS_STATIC=
export HIF8_DTS_HISTORY=
export HIF8_DTS_ALGO=
export HIF8_DTS_STAT=
export HIF8_DTS_STAT_N=
export HIF8_DTS_STAT_ROLES=

exec bash "$ROOT/chain_ptlr_seed.sh"
