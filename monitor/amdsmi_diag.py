import time

from amdsmi import (
    AmdSmiTemperatureMetric,
    AmdSmiTemperatureType,
    amdsmi_get_gpu_activity,
    amdsmi_get_gpu_memory_total,
    amdsmi_get_gpu_memory_usage,
    amdsmi_get_gpu_metrics_info,
    amdsmi_get_gpu_vram_usage,
    amdsmi_get_processor_handles,
    amdsmi_get_processor_type,
    amdsmi_get_temp_metric,
    amdsmi_init,
    amdsmi_shut_down,
)


def _safe_call(label, func, *args):
    try:
        return func(*args), None
    except Exception as error:
        return None, f"{label} failed: {error}"


def main():
    amdsmi_init()
    try:
        handles = amdsmi_get_processor_handles()
        print("handles:", len(handles))
        for idx, handle in enumerate(handles):
            ptype, ptype_err = _safe_call("processor_type", amdsmi_get_processor_type, handle)
            if ptype_err:
                print(f"gpu {idx} processor_type error: {ptype_err}")
                continue
            print(f"gpu {idx} processor_type: {ptype}")

            activity, activity_err = _safe_call("activity", amdsmi_get_gpu_activity, handle)
            if activity_err:
                print(f"gpu {idx} {activity_err}")
            else:
                print(f"gpu {idx} activity: {activity}")

            vram_info, vram_err = _safe_call("vram_usage", amdsmi_get_gpu_vram_usage, handle)
            if vram_err:
                print(f"gpu {idx} {vram_err}")
            else:
                print(f"gpu {idx} vram_usage: {vram_info}")

            mem_used, mem_used_err = _safe_call(
                "memory_usage", amdsmi_get_gpu_memory_usage, handle
            )
            if mem_used_err:
                print(f"gpu {idx} {mem_used_err}")
            else:
                print(f"gpu {idx} memory_usage: {mem_used}")

            mem_total, mem_total_err = _safe_call(
                "memory_total", amdsmi_get_gpu_memory_total, handle
            )
            if mem_total_err:
                print(f"gpu {idx} {mem_total_err}")
            else:
                print(f"gpu {idx} memory_total: {mem_total}")

            metrics, metrics_err = _safe_call("gpu_metrics", amdsmi_get_gpu_metrics_info, handle)
            if metrics_err:
                print(f"gpu {idx} {metrics_err}")
            else:
                keys_preview = list(metrics.keys())[:10]
                print(f"gpu {idx} gpu_metrics keys: {keys_preview}")

            temp, temp_err = _safe_call(
                "temperature",
                amdsmi_get_temp_metric,
                handle,
                AmdSmiTemperatureType.EDGE,
                AmdSmiTemperatureMetric.CURRENT,
            )
            if temp_err:
                print(f"gpu {idx} {temp_err}")
            else:
                print(f"gpu {idx} temperature_milli: {temp}")
            time.sleep(0.05)
    finally:
        amdsmi_shut_down()


if __name__ == "__main__":
    main()
