"""Rollout 进程内打 RolloutEvent，供 TensorBoard Rollout/GPU_X_Phase 使用。"""
import functools

from macro import RolloutEvent
from monitor.monitor import Monitor


ROLLOUT_GPU_ID = 0


def monkey_patch_rollout(monitor: Monitor | None):
    """包装 SwiftRolloutDeploy.infer，在每次生成前后打 GENERATE_START / GENERATE_END。"""
    from swift.llm.infer.rollout import SwiftRolloutDeploy

    original_infer = SwiftRolloutDeploy.infer

    @functools.wraps(original_infer)
    async def patched_infer(self, infer_requests, request_config=None, *, use_tqdm=None):
        if monitor:
            monitor.add_event(RolloutEvent.GENERATE_START, gpu_id=ROLLOUT_GPU_ID)
        try:
            result = await original_infer(
                self, infer_requests, request_config=request_config, use_tqdm=use_tqdm
            )
            return result
        finally:
            if monitor:
                monitor.add_event(RolloutEvent.GENERATE_END, gpu_id=ROLLOUT_GPU_ID)

    SwiftRolloutDeploy.infer = patched_infer
    print("[MonkeyPatch] Rollout infer -> RolloutEvent.GENERATE_START/END")
