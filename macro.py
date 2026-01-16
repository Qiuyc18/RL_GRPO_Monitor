import enum

class Event(enum.Enum):
    # --- 阶段 1: 生产 (Rollout) ---
    # 真正的网络 IO 等待时间（你的训练卡在等 vLLM 卡）
    VLLM_WAIT_START = 0   
    VLLM_WAIT_END = 1     
    
    # --- 阶段 2: 整理 (Data Prep) ---
    # 数据搬运与 Tokenization（CPU -> GPU 显存的第一波上涨）
    TOKENIZE_START = 2    
    TOKENIZE_END = 3      
    
    # --- 阶段 3: 评分 (Reward) ---
    # 你的 Math 评分函数耗时（通常是 CPU 密集型，监测是否阻塞了 GPU）
    REWARD_CALC_START = 4 
    REWARD_CALC_END = 5   
    
    # --- 阶段 4: 训练 (Optimization) ---
    # Forward Pass：显存开始攀升（加载 Activation）
    FORWARD_START = 6     
    FORWARD_END = 7       
    
    # Backward Pass：显存达到峰值（存储 Gradients，最容易 OOM 的时刻）
    BACKWARD_START = 8    
    BACKWARD_END = 9