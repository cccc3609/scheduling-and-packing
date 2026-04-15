import zipfile
import io
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from envs.scheduling_env import SchedulingEnv
from sb3_contrib.common.wrappers import ActionMasker
from models.attention_extractor import AttentionFeatureExtractor

def mask_fn(env):
    return env.get_wrapper_attr("_get_action_mask")()

sched_path = "./experiments/exp_20260408_140951/models/scheduling_joint_c30.zip"

print("1. 从 zip 里提取 policy.pth ...")
with zipfile.ZipFile(sched_path, 'r') as zf:
    print("  zip 内容:", zf.namelist())
    # SB3 保存的权重文件通常叫 policy.pth
    with zf.open("policy.pth") as f:
        data = io.BytesIO(f.read())
        state_dict = torch.load(data, map_location="cpu")
print("2. 读取成功，keys 数量:", len(state_dict))
print("   前5个key:", list(state_dict.keys())[:5])