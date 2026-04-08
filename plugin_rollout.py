"""兼容旧的 rollout 插件入口。"""

from monitor.integrations.ms_swift import monkey_patch_rollout

ROLLOUT_GPU_ID = 0

__all__ = ["ROLLOUT_GPU_ID", "monkey_patch_rollout"]
