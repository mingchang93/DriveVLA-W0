# -*- coding: utf-8 -*-
"""抽样校验已解压的 JPEG 是否完好。

为什么需要：`tar -xf` 返回 0 只保证归档结构完整、写入没报错，
不保证文件内容逐字节正确（普通 tar 没有内容校验和）。
已经删掉归档的那几个包（camera_3/7/21/25）没法再比对，
但 JPEG 自身有结构可查：必须以 FFD8（SOI）开头、FFD9（EOI）结尾。
截断的文件缺尾标记，这是最能抓出问题的廉价检查。

用法: python3 verify_jpg.py [每个 log 抽样数] [最多检查多少个 log]
"""
import os
import random
import sys

ROOT = "/mnt/sfs_turbo/datasets/nuplan-v1.1/images"
PER_LOG = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAX_LOG = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9

random.seed(12345)   # 固定抽样，便于复查


def check(path):
    """返回 None 表示正常，否则返回问题描述。"""
    try:
        sz = os.path.getsize(path)
        if sz == 0:
            return "大小为 0"
        if sz < 1024:
            return "过小 %d 字节" % sz
        with open(path, "rb") as f:
            head = f.read(2)
            if head != b"\xff\xd8":
                return "缺 SOI 头 %r" % head
            f.seek(-2, os.SEEK_END)
            if f.read(2) != b"\xff\xd9":
                return "缺 EOI 尾（疑似截断）"
    except Exception as e:
        return "读取失败 %r" % (e,)
    return None


logs = sorted(os.listdir(ROOT))
print("images 下 %d 个 log，每个抽 %d 张" % (len(logs), PER_LOG))

checked = bad = empty_log = 0
bad_list = []
for i, lg in enumerate(logs[:MAX_LOG]):
    d = os.path.join(ROOT, lg)
    if not os.path.isdir(d):
        continue
    jpgs = []
    for cam in os.listdir(d):
        cd = os.path.join(d, cam)
        if not os.path.isdir(cd):
            continue
        fs = [x for x in os.listdir(cd) if x.endswith(".jpg")]
        if fs:
            jpgs += [os.path.join(cd, x) for x in random.sample(fs, min(PER_LOG, len(fs)))]
    if not jpgs:
        empty_log += 1
        bad_list.append((lg, "该 log 下没有 jpg"))
        continue
    for p in random.sample(jpgs, min(PER_LOG, len(jpgs))):
        r = check(p)
        checked += 1
        if r:
            bad += 1
            if len(bad_list) < 25:
                bad_list.append((os.path.relpath(p, ROOT), r))

print("抽查 %d 张，异常 %d 张，空 log %d 个" % (checked, bad, empty_log))
if bad_list:
    print("前若干问题:")
    for p, r in bad_list[:25]:
        print("   %s  ->  %s" % (p, r))
else:
    print("全部通过：抽样文件都有完整的 SOI/EOI 标记，没有截断")
