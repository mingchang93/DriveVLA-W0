#!/bin/bash
# DTS full-chain preset: 168 quantized Linear layers, refresh interval 200, seed 42.
#
# Matches Yichen's pt_dts196_i200_s43 recipe except:
#   - exclude o_proj from quantization, leaving 168 Linear layers
#   - use the default seed, 42
#
# 168 = 28 language-model layers x 6 projections (q, k, v, gate, up, down)
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export TZ=Asia/Shanghai
export TAG=dts168_i200_s42
export SEED=42

export HIF8_PT=dts
export HIF8_SFT=dts
export HIF8_INTERVAL=200
export IG_PT='lm_head,o_proj'
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
