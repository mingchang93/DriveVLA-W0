#!/bin/bash
# 按新机器上"正在正常训练"的那个容器的配方重建：privileged + 新机器实际存在的设备节点。
# 新机器没有 /dev/devmm_svm、/dev/ummu、/dev/uburma（lmc 在旧 a56 上挂的那三个），
# 多了 /dev/hisi_soc_cache_mgmt。
set -u
IMG=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-950-openeuler24.03-py3.12
NAME=zyc-w0
docker rm -f "$NAME" >/dev/null 2>&1
DEVS=""
for d in /dev/davinci0 /dev/davinci1 /dev/davinci2 /dev/davinci3 \
         /dev/davinci4 /dev/davinci5 /dev/davinci6 /dev/davinci7 \
         /dev/davinci_manager /dev/hisi_hdc /dev/hisi_soc_cache_mgmt \
         /dev/devmm_svm /dev/ummu /dev/uburma; do
  [ -e "$d" ] && DEVS="$DEVS --device=$d"
done
docker run -itd --name "$NAME" --privileged \
  --net=host --ipc=host --shm-size=64g $DEVS \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /mnt:/mnt -v /data1:/data1 -v /data2:/data2 \
  "$IMG" bash >/dev/null
echo "$(hostname -s) run_exit=$?  挂了 $(echo $DEVS | tr ' ' '\n' | grep -c device) 个设备"
docker ps --filter name="$NAME" --format "  状态: {{.Status}}"
echo "  --- torch_npu 自检 ---"
docker exec "$NAME" bash -lc '
  source /mnt/sfs_turbo/lmc/venv-hif8/bin/activate
  python -c "
import torch, torch_npu
print(\"    npu available:\", torch.npu.is_available())
print(\"    device count: \", torch.npu.device_count())
from transformers.utils import is_torch_bf16_gpu_available
print(\"    bf16 supported:\", is_torch_bf16_gpu_available())
" 2>&1 | grep -vE "UserWarning|warnings.warn|console log level|Different containers|character device|number of devices|^ [0-9]\."
'
