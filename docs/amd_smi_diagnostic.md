# 监控指标缺失诊断

## 摘要
在服务器 MI250X（4 张卡，8 个 GPU 芯片）节点上，AMD SMI 与 ROCm SMI 无法读取 GPU 活动/功耗/时钟/温度等关键指标，仅能读取部分显存使用。问题源于驱动/固件层面无法导出 SMU metrics 表，导致上层接口返回 `UNEXPECTED_DATA` / `NOT_SUPPORTED`。

## 影响
- GPU 利用率（GPU%）、功耗、电压、频率、温度等监控指标不可用
- 监控工具/代码只能拿到显存使用（MEM_USAGE）

## 环境信息
- GPU: AMD MI250X（4 张卡，8 个 GPU 芯片）
- ROCm: 6.2.0
- amdsmi: 24.6.2+2b02a07
- amdgpu 驱动: 6.8.5
  ```
  modinfo amdgpu | head -n 5
  filename:       /lib/modules/5.15.0-25-generic/updates/dkms/amdgpu.ko
  version:        6.8.5
  ```

## 报错信息

### 1) amdsmi 读取指标报错
- 诊断脚本结果显示：
  - activity: `AMDSMI_STATUS_UNEXPECTED_DATA`
  - vram_usage: `AMDSMI_STATUS_NOT_SUPPORTED`
  - gpu_metrics: `AMDSMI_STATUS_UNEXPECTED_DATA`
  - temperature: 0

- 
```
python -m monitor.amdsmi_diag
# 部分输出：
gpu 0 activity failed: Error code: 43 | AMDSMI_STATUS_UNEXPECTED_DATA
gpu 0 vram_usage failed: Error code: 2 | AMDSMI_STATUS_NOT_SUPPORTED
gpu 0 gpu_metrics failed: Error code: 43 | AMDSMI_STATUS_UNEXPECTED_DATA
gpu 0 temperature_milli: 0
...
```

### 2) amd-smi metric 大量 N/A
`amd-smi metric` 输出中大部分字段为 `N/A` 或 `0`，仅 MEM_USAGE 有数据：
```
amd-smi metric
# 片段：
USAGE: N/A
POWER: N/A
CLOCK: N/A
TEMPERATURE: 0 °C
MEM_USAGE:
  TOTAL_VRAM: 65520 MB
  USED_VRAM: 10 MB
```

### 3) rocm-smi 显示几乎全部 N/A


### 4) 内核日志明确提示 SMU 指标导出失败
```
sudo dmesg | rg -i 'amdgpu|kfd|smi' | tail -n 50
# 关键日志：
amdgpu: Failed to export SMU metrics table!
amdgpu: Failed to retrieve enabled ppfeatures!
amdgpu: PPT feature is not enabled, power values can't be fetched.
```

## 结论
该问题不是用户态监控代码导致，而是驱动/固件层 SMU 指标表无法导出。因此 AMD SMI / ROCm SMI 上层 API 无法读取 GPU 活动、功耗、频率、温度等指标。

## 建议处理方向
1) 检查/修复 MI250X 的 firmware 与 SMU 相关功能是否完整加载
2) 检查是否存在禁用 PPT/SMU 的内核启动参数或系统配置
3) 考虑升级或回退到官方更稳定的 ROCm/驱动组合（MI250X 支持矩阵）
4) 若驱动层无法修复，监控只能退化为“显存使用量”为主的指标

## 截图/补充证据
- [ ] `amd-smi metric` 截图
- [ ] `rocm-smi` 截图
- [ ] `dmesg` 关键日志截图
