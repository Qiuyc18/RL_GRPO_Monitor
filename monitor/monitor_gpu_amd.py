from amdsmi import (
    AmdSmiTemperatureMetric,
    AmdSmiTemperatureType,
    amdsmi_init,
    amdsmi_shut_down,
    amdsmi_get_processor_handles,
    amdsmi_get_processor_type,
    amdsmi_get_gpu_activity,
    amdsmi_get_gpu_vram_usage,
    amdsmi_get_temp_metric,
)
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def _is_amd_gpu(processor_type):
    if processor_type.get("processor_type") == "AMDSMI_PROCESSOR_TYPE_AMD_GPU":
        return True
    return False


def _normalize_number(value):
    if value is None:
        return None
    if isinstance(value, str):
        if value.upper() == "N/A":
            return None
        try:
            return float(value)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_percent(value):
    return _normalize_number(value)


class AMDMonitor:
    def __init__(self):
        self._handles = []
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        amdsmi_init()
        handles = amdsmi_get_processor_handles()
        gpu_handles = []
        for handle in handles:
            if _is_amd_gpu(amdsmi_get_processor_type(handle)):
                gpu_handles.append(handle)
        self._handles = gpu_handles
        self._initialized = True

    def shutdown(self):
        if not self._initialized:
            return
        amdsmi_shut_down()
        self._initialized = False

    def read_metrics(self):
        rows = []
        for gpu_id, handle in enumerate(self._handles):
            activity = {}
            vram_info = {}
            temperature_milli = None
            try:
                activity = amdsmi_get_gpu_activity(handle)
            except Exception as error:
                logger.warning("GPU %s activity read failed: %s", gpu_id, error)
            try:
                vram_info = amdsmi_get_gpu_vram_usage(handle)
            except Exception as error:
                logger.warning("GPU %s VRAM read failed: %s", gpu_id, error)
            try:
                temperature_milli = amdsmi_get_temp_metric(
                    handle,
                    AmdSmiTemperatureType.EDGE,
                    AmdSmiTemperatureMetric.CURRENT,
                )
            except Exception as error:
                logger.warning("GPU %s temperature read failed: %s", gpu_id, error)
            gpu_utilization = _normalize_percent(activity.get("gfx_activity"))
            vram_used = _normalize_number(vram_info.get("vram_used"))
            vram_total = _normalize_number(vram_info.get("vram_total"))
            memory_utilization = None
            if vram_total and vram_used is not None:
                memory_utilization = round((vram_used / vram_total) * 100, 2)
            temperature = None
            if temperature_milli is not None:
                temperature = round(float(temperature_milli) / 1000.0, 2)

            rows.append(
                {
                    "gpu_id": gpu_id,
                    "gpu_utilization": gpu_utilization,
                    "memory_utilization": memory_utilization,
                    "temperature": temperature,
                }
            )
        return rows
