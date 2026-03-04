"""Rollout 进程内打 PhaseEvent（ROLLOUT_PHASE_*），供 TensorBoard Rollout/GPU_X_Phase 使用。"""
import functools

from macro import PhaseEvent
from monitor.monitor import Monitor


ROLLOUT_GPU_ID = 0


def monkey_patch_rollout(monitor: Monitor | None):
    """包装 SwiftRolloutDeploy.infer，在每次生成前后打 ROLLOUT_PHASE_START / ROLLOUT_PHASE_END。"""
    from swift.llm.infer.rollout import SwiftRolloutDeploy

    original_infer = SwiftRolloutDeploy.infer

    @functools.wraps(original_infer)
    async def patched_infer(self, infer_requests, request_config=None, *, use_tqdm=None):
        if monitor:
            monitor.add_event(
                PhaseEvent.ROLLOUT_PHASE_START,
                gpu_id=ROLLOUT_GPU_ID,
                mode="external",
                role="rollout",
            )
        try:
            result = await original_infer(
                self, infer_requests, request_config=request_config, use_tqdm=use_tqdm
            )
            return result
        finally:
            if monitor:
                monitor.add_event(
                    PhaseEvent.ROLLOUT_PHASE_END,
                    gpu_id=ROLLOUT_GPU_ID,
                    mode="external",
                    role="rollout",
                )

    SwiftRolloutDeploy.infer = patched_infer
    print("[MonkeyPatch] Rollout infer -> PhaseEvent.ROLLOUT_PHASE_START/END (role=rollout)")
