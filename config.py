# config.py

# === 1. 神经网络"脑容量"上限 ===
MAX_PARTS_CAPACITY = 120
MAX_SCHED_TASKS_CAPACITY = 120
NUM_MACHINES = 3

# === 2. 训练配置 (保持不变) ===
TRAIN_CONFIG = {
    # 零件数量范围 [30, 100]
    "min_parts": 30,
    "max_parts": 100,

    # 板材尺寸范围 [Min, Max]
    "min_plate_dim": 150,
    "max_plate_dim": 350,

    # 交期因子范围
    "due_date_factor_range": (1.1, 2.5),

    "total_cycles": 50,
    "steps_per_cycle": 30000,
    # 学习率配置
    "lr_start": 1e-3,
    "lr_end": 1e-5
}

# === 3. 💰 成本模型配置 (Cost Model) ===
# 假设单位：距离=cm, 时间=min, 货币=元
COST_CONFIG = {
    "cutting_speed": 10.0,

    # [材料成本]
    # 普通钢板约 0.05元/cm² (200x200板材约 2000元)
    "cost_material": 0.05,

    # [库存持有成本] (提前完工)
    # 占用仓库资金，假设为材料价值的万分之一/每分钟
    "cost_earliness": 0.0005,

    # [延期违约成本] (迟到)
    # 罚款通常是库存成本的 20~50 倍
    "cost_tardiness": 0.001,

    # 默认板材尺寸 (仅初始化用)
    "default_plate_size": (200, 200)
}

# === 4. 测试配置 ===
TEST_SCENARIOS = [
    # 1. 基础基准 (训练时的典型情况)
    {
        "name": "Standard Benchmark",
        "num_parts": 50,
        "plate_size": (200, 200)
    },

    # 2. 小规模测试 (零件少，容易排，看JIT是否精准)
    {
        "name": "Small Scale (Low Load)",
        "num_parts": 30,
        "plate_size": (150, 150)  # 板子也变小
    },

    # 3. 大规模压力测试 (零件多，考验填缝能力和调度抗压能力)
    {
        "name": "Large Scale (High Load)",
        "num_parts": 100,
        "plate_size": (250, 250)
    },

    # 4. 异形板材测试 - 扁长条 (考验排样策略对长宽比的适应性)
    {
        "name": "Wide Plate (Strip)",
        "num_parts": 60,
        "plate_size": (400, 120)
    },

    # 5. 异形板材测试 - 竖长条
    {
        "name": "Tall Plate (Tower)",
        "num_parts": 60,
        "plate_size": (120, 400)
    },

    # 6. 极限大板 (模拟大件加工)
    {
        "name": "Huge Industrial Plate",
        "num_parts": 100,
        "plate_size": (500, 500)
    }
]