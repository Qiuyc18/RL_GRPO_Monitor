from pynvml import (
    NVML_TEMPERATURE_GPU,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


class NVIDIAMonitor:
    def __init__(self):
        self._handles = []
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        self._handles = [nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
        self._initialized = True

    def shutdown(self):
        if not self._initialized:
            return
        nvmlShutdown()
        self._initialized = False

    def read_metrics(self):
        rows = []
        for gpu_id, handle in enumerate(self._handles):
            utilization = nvmlDeviceGetUtilizationRates(handle)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

            memory_utilization = None
            if memory_info.total:
                memory_utilization = round(
                    (float(memory_info.used) / float(memory_info.total)) * 100, 2
                )
            rows.append(
                {
                    "gpu_id": gpu_id,
                    "gpu_utilization": utilization.gpu,
                    "memory_utilization": memory_utilization,
                    "temperature": temperature,
                }
            )
        return rows
