
#神经网络
MAX_PARTS_CAPACITY = 120
MAX_SCHED_TASKS_CAPACITY = 120
NUM_MACHINES = 3


# 训练配置


# 成本
# 假设单位：距离=cm, 时间=min, 货币=元
COST_CONFIG = {
    "cutting_speed": 10.0,
    # 每张投入加工的板材均需一次固定装夹/准备时间（min）
    "plate_setup_time": 0.1,

    # [材料成本]
    # 普通钢板约 0.05元/cm² (200x200板材约 2000元)
    "cost_material": 0.05,

    # [库存持有成本] (提前完工)
    # 占用仓库资金，假设为材料价值的万分之一/每分钟
    "cost_earliness": 0.0005,

    # [延期违约成本] (迟到)
    # 罚款通常是库存成本的 20~50 倍
    "cost_tardiness": 0.002,

    # 默认板材尺寸 (仅初始化用)
    "default_plate_size": (200, 200)
}

#特征配置
FEATURE_CONFIG = {
    # 将板材宽度离散化为 20 个格子，描述轮廓高低
    "skyline_bins": 20,
}

TRAIN_CONFIG = {

    'min_parts': 30,
    'max_parts': 60,

    'min_plate_dim': 200,
    'max_plate_dim': 250,

    # 【课程式训练节奏控制】
    'total_cycles': 50,  # 总循环数。保障扣除 Phase 1 & 2 之后，有足够轮数微调

    # 【核心提速】：加快 Tensorboard 反馈频率
    # 原本 100,000 步太久了。现在改为 30,000 步保存一次并更新指标。
    'steps_per_cycle': 50000,

    # 学习率退火策略
    'lr_start': 3e-4,  # 初始探索学习率
    'lr_end': 1e-5  # 收敛期微调学习率
}


TEST_SCENARIOS =[
    # 场景1：基础基准 (Standard) - 常规零件数量与标准板材
    {
        "name": "1. Standard Scale",
        "num_parts": 40,
        "plate_size": (200, 200)
    },

    # 场景2：小规模密集测试 (Small & Tight) - 零件少但板材也小，极容易触发交期违约
    {
        "name": "2. Small Scale",
        "num_parts": 30,
        "plate_size": (150, 150)
    },

    # 场景3：大规模高压测试 (Large Scale) - 验证网络在面对较多零件时的 Attention 聚合能力
    {
        "name": "3. Large Scale",
        "num_parts": 80,
        "plate_size": (250, 250)
    },

    # 场景4：异形板材测试 - 扁长条 (Wide Plate) - 考验 RL 与底层 Skyline/MaxRects 对极端长宽比容器的适应度
    {
        "name": "4. Wide Plate (Strip)",
        "num_parts": 50,
        "plate_size": (300, 100)
    },

    # 场景5：异形板材测试 - 竖长条 (Tall Plate)
    {
        "name": "5. Tall Plate (Tower)",
        "num_parts": 50,
        "plate_size": (100, 300)
    },

    # 场景6：极限规模满载 (Extreme Load) - 逼近动作空间和状态空间的 120 维极限
    {
        "name": "6. Extreme High Load",
        "num_parts": 120,
        "plate_size": (300, 300)
    }
]
