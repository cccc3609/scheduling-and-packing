"""
models/comm_encoders.py  —  可训练通信编码器 + 联合奖励计算

协同优化改进：
  1. GlobalCostFunction: 统一成本函数，两个 agent 共享同一评价标准
  2. JointRewardCalculator: Nesting 终局时调用真实 Scheduling agent 做完整调度，
     替代 EDD 代理，确保 Nesting 的奖励反映 Scheduling 的实际策略质量
  3. SchedulingIntentEncoder / NestingResultEncoder 增加 forward() 方法，
     参数可纳入对应 agent 的优化器，通信向量可被梯度更新
"""

import torch
import torch.nn as nn
import numpy as np
import copy


# ─────────────────────────────────────────────────────────────────────────────
# 全局成本函数：两个 agent 共享的终局评价标准
# ─────────────────────────────────────────────────────────────────────────────

class GlobalCostFunction:
    """
    统一成本计算。Nesting 和 Scheduling 的终局奖励都从这里派生，
    确保两个 agent 优化同一个目标函数。

    总成本 = 材料成本 + JIT 成本(提前/延迟)

    核心原则：
    - 这个类只负责"算分"，不负责"给奖励"
    - 两个 agent 看到的终局成本来自同一次计算，消除评价标准的分歧
    """

    def __init__(self, cost_mat=0.05, cost_hold=0.0005, cost_tard=0.002,
                 cutting_speed=10.0):
        self.cost_mat = cost_mat
        self.cost_hold = cost_hold
        self.cost_tard = cost_tard
        self.cutting_speed = cutting_speed

    def compute(self, plates, orders, parts_pool, plate_w, plate_h,
                order_finish_times):
        """
        统一计算总成本。

        参数:
            plates: 排样后的板材列表
            orders: 订单字典 {oid: {'due_date': ...}}
            parts_pool: 零件列表
            plate_w, plate_h: 板材尺寸
            order_finish_times: {oid: finish_time}
                                 来自真实调度 rollout 或 EDD 模拟

        返回:
            cost_dict: 包含各项成本和评价指标的字典
        """
        final_plates = [p for p in plates if len(p.placed_parts) > 0]
        total_part_area = sum(p['area'] for p in parts_pool)
        consumed_area = len(final_plates) * (plate_w * plate_h)
        utilization = total_part_area / consumed_area if consumed_area > 0 else 0.001

        # 材料成本
        cost_material = max(0.0, consumed_area - total_part_area) * self.cost_mat

        # JIT 成本
        cost_jit = 0.0
        total_delay = 0.0
        late_count = 0
        for oid, fin in order_finish_times.items():
            if oid not in orders:
                continue
            due = orders[oid]['due_date']
            ops = [p for p in parts_pool if p['order_id'] == oid]
            val = sum(p['area'] for p in ops)
            proc = max(1.0, sum(2 * (p['w'] + p['h']) for p in ops) / self.cutting_speed)
            diff = fin - due
            ratio = abs(diff) / proc
            coef = 0.0 if ratio <= 0.025 else min(1.0, (ratio - 0.025) / 0.05)
            if diff > 0:
                cost_jit += val * self.cost_tard * coef * abs(diff)
                total_delay += diff
                late_count += 1
            else:
                cost_jit += val * self.cost_hold * coef * abs(diff)

        total_cost = cost_material + cost_jit
        intrinsic = max(1.0, total_part_area * self.cost_mat)

        return {
            'cost_material': cost_material,
            'cost_jit': cost_jit,
            'cost_total': total_cost,
            'utilization': utilization,
            'plate_count': len(final_plates),
            'total_delay': total_delay,
            'late_count': late_count,
            'penalty_ratio': total_cost / intrinsic,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 联合奖励计算器
# ─────────────────────────────────────────────────────────────────────────────

class JointRewardCalculator:
    """
    解决奖励解耦的核心组件。

    问题：
    - 原来 Nesting 用 EDD 贪心模拟调度结果来算自己的终局奖励
    - Scheduling 用自己真实策略的结果算终局奖励
    - 两个 agent 评价同一个排样方案时得出不同的成本 → 优化方向矛盾

    解决：
    - Phase 1 (无 scheduling partner): 用 EDD 模拟 → 算成本 → Nesting 奖励
    - Phase 2+ (有 scheduling partner): 用真实 Scheduling agent rollout → 算成本
      → 同一个成本数字同时给 Nesting（终局奖励）和 Scheduling（作为 baseline 参考）

    使用方式：
        在 NestingSchedulingEnv 的 terminated 块中：
            cost_dict = self.joint_reward_calc.compute_terminal_cost(...)
            nesting_reward = self.joint_reward_calc.to_nesting_reward(cost_dict)

        在 SchedulingEnv 的 terminated 块中：
            # scheduling 侧已经有真实的 order_finish_times，直接用 GlobalCostFunction
    """

    def __init__(self, global_cost_fn: GlobalCostFunction, num_machines=3):
        self.global_cost_fn = global_cost_fn
        self.num_machines = num_machines
        # Phase 3 联合微调时，记录上一轮的成本作为 baseline（方差缩减）
        self._cost_ema = None
        self._ema_alpha = 0.1

    def compute_terminal_cost(
        self,
        plates, orders, parts_pool, plate_w, plate_h,
        sched_env=None, sched_model=None
    ):
        """
        计算终局成本。如果 scheduling partner 可用，用真实调度；否则 EDD fallback。

        参数:
            plates, orders, parts_pool, plate_w, plate_h: 排样结果
            sched_env: SchedulingEnv 实例（可选）
            sched_model: Scheduling agent（可选）

        返回:
            cost_dict: GlobalCostFunction.compute() 的输出
        """
        if sched_env is not None and sched_model is not None:
            return self._compute_with_real_scheduling(
                sched_env, sched_model, plates, orders, parts_pool, plate_w, plate_h
            )
        else:
            return self._compute_with_edd(plates, orders, parts_pool, plate_w, plate_h)

    def _compute_with_real_scheduling(
        self, sched_env, sched_model, plates, orders, parts_pool, plate_w, plate_h
    ):
        """用真实 Scheduling agent 做一次完整调度 rollout"""
        try:
            # scheduling env 的 reset() 会自动从内部 nesting_env 读取 history_plates
            # 但这里我们直接构造 task_pool 避免重复 nesting rollout
            sched_env_unwrapped = sched_env.unwrapped if hasattr(sched_env, 'unwrapped') else sched_env

            # 保存状态
            old_task_pool = sched_env_unwrapped.task_pool
            old_mask = sched_env_unwrapped.scheduled_mask.copy()
            old_machine_times = sched_env_unwrapped.machine_times.copy()
            old_orders = copy.deepcopy(sched_env_unwrapped.orders_snapshot)

            # 直接构造 task pool（不需要再次 nesting rollout）
            speed = max(0.1, self.global_cost_fn.cutting_speed)
            task_pool = []
            for plate in plates:
                if not plate.placed_parts:
                    continue
                cut = sum(2 * (p[2] + p[3]) for p in plate.placed_parts) / speed
                oids = list(set(int(p[4]) for p in plate.placed_parts))
                val = sum(p[2] * p[3] for p in plate.placed_parts)
                due = min((orders[o]['due_date'] for o in oids if o in orders),
                          default=9999.0)
                task_pool.append({'cut': cut, 'due': due, 'oids': oids, 'val': val})

            # 注入并运行
            sched_env_unwrapped.task_pool = task_pool
            sched_env_unwrapped.scheduled_mask = np.zeros(sched_env_unwrapped.max_tasks, dtype=bool)
            sched_env_unwrapped.machine_times = np.zeros(sched_env_unwrapped.num_machines)
            sched_env_unwrapped.orders_snapshot = copy.deepcopy(orders)
            for oid in sched_env_unwrapped.orders_snapshot:
                sched_env_unwrapped.orders_snapshot[oid]['finished_time'] = 0.0
                # 补全可能缺失的字段
                if 'total_area' not in sched_env_unwrapped.orders_snapshot[oid]:
                    ops = [p for p in parts_pool if p['order_id'] == oid]
                    sched_env_unwrapped.orders_snapshot[oid]['total_area'] = sum(p['area'] for p in ops)
                    sched_env_unwrapped.orders_snapshot[oid]['proc_time'] = max(
                        1.0, sum(2 * (p['w'] + p['h']) for p in ops) / speed)

            obs = sched_env_unwrapped._get_obs()
            mask = sched_env_unwrapped._get_action_mask()

            done = False
            max_steps = len(task_pool) + 5  # 安全上限
            step_count = 0
            while not done and step_count < max_steps:
                action, _ = sched_model.predict(obs, action_masks=mask, deterministic=True)
                obs, _, done, _, _ = sched_env_unwrapped.step(action)
                if not done:
                    mask = sched_env_unwrapped._get_action_mask()
                step_count += 1

            # 读取真实 finish times
            order_finish_times = {
                oid: o['finished_time']
                for oid, o in sched_env_unwrapped.orders_snapshot.items()
            }

            # 恢复状态
            sched_env_unwrapped.task_pool = old_task_pool
            sched_env_unwrapped.scheduled_mask = old_mask
            sched_env_unwrapped.machine_times = old_machine_times
            sched_env_unwrapped.orders_snapshot = old_orders

            return self.global_cost_fn.compute(
                plates, orders, parts_pool, plate_w, plate_h,
                order_finish_times
            )

        except Exception as e:
            print(f"  [JointReward] Scheduling rollout failed ({e}), EDD fallback")
            return self._compute_with_edd(plates, orders, parts_pool, plate_w, plate_h)

    def _compute_with_edd(self, plates, orders, parts_pool, plate_w, plate_h):
        """EDD 贪心模拟 fallback"""
        cutting_speed = self.global_cost_fn.cutting_speed
        local = {oid: {'due_date': v['due_date'], 'finished_time': 0.0}
                 for oid, v in orders.items()}

        tasks = []
        for plate in plates:
            if not plate.placed_parts:
                continue
            cut = sum(2 * (p[2] + p[3]) for p in plate.placed_parts) / cutting_speed
            oids = list(set(int(p[4]) for p in plate.placed_parts))
            due = min((local[o]['due_date'] for o in oids if o in local), default=999.0)
            tasks.append({'cut': cut, 'due': due, 'oids': oids})

        tasks.sort(key=lambda t: t['due'])
        mach = [0.0] * self.num_machines

        for task in tasks:
            m = int(min(range(len(mach)), key=lambda i: mach[i]))
            end = mach[m] + task['cut']
            mach[m] = end
            for oid in task['oids']:
                if oid in local:
                    local[oid]['finished_time'] = max(local[oid]['finished_time'], end)

        order_finish_times = {oid: v['finished_time'] for oid, v in local.items()}
        return self.global_cost_fn.compute(
            plates, orders, parts_pool, plate_w, plate_h,
            order_finish_times
        )

    def to_reward(self, cost_dict, w_terminal=10.0):
        """
        将 cost_dict 转化为标量奖励（两个 agent 共享同一个转换逻辑）。
        使用 EMA baseline 做方差缩减。
        """
        import math
        pr = cost_dict['penalty_ratio']

        # 更新 EMA baseline
        if self._cost_ema is None:
            self._cost_ema = pr
        else:
            self._cost_ema = self._ema_alpha * pr + (1 - self._ema_alpha) * self._cost_ema

        # advantage = -(当前成本 - baseline)
        # 好于平均 → 正奖励，差于平均 → 负奖励
        advantage = -(pr - self._cost_ema)

        # 用 tanh 压缩，映射到 [-w_terminal, +w_terminal]
        scaled = w_terminal * math.tanh(advantage * 2.0)
        return float(np.clip(scaled, -w_terminal * 2, w_terminal * 2))


# ─────────────────────────────────────────────────────────────────────────────
# 可训练通信编码器
# ─────────────────────────────────────────────────────────────────────────────

class SchedulingIntentEncoder(nn.Module):
    """
    Scheduling agent 在 episode 开始时，看到所有订单信息，
    输出 16 维"调度意图向量" m_s，传给 Nesting agent。

    改进：
    - 新增 forward() 方法，支持有梯度的前向传播
    - 参数可纳入 Scheduling agent 的优化器
    - encode_orders() 保留为无梯度的推理接口
    """

    def __init__(self, input_dim: int = 12, comm_dim: int = 16):
        super().__init__()
        self.comm_dim = comm_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, comm_dim),
            nn.Tanh(),
        )

    def _extract_features(self, orders: dict, parts_pool: list,
                          num_machines: int, cutting_speed: float) -> list:
        """提取 12 维手工特征"""
        if not orders:
            return [0.0] * self.input_dim

        dues = [o['due_date'] for o in orders.values()]
        total_area = sum(p['area'] for p in parts_pool)
        total_proc = sum(2 * (p['w'] + p['h']) for p in parts_pool) / max(0.1, cutting_speed)
        num_orders = len(orders)

        features = [
            np.mean(dues) / max(1.0, total_proc),
            np.std(dues) / max(1.0, total_proc),
            np.min(dues) / max(1.0, total_proc),
            np.max(dues) / max(1.0, total_proc),
            num_machines / max(1.0, num_orders),
            total_proc / max(1.0, num_machines * np.max(dues)),
            len([d for d in dues if d < total_proc]) / num_orders,
            total_area / max(1.0, num_orders),
            np.std([len([p for p in parts_pool if p['order_id'] == oid])
                    for oid in orders]) / max(1.0, num_orders),
            num_orders / 10.0,
            total_proc / max(1.0, np.min(dues)),
            np.mean([o.get('finished_time', 0.0) for o in orders.values()]),
        ]
        return features

    def forward(self, feat_tensor: torch.Tensor) -> torch.Tensor:
        """有梯度的前向传播（训练时）"""
        return self.encoder(feat_tensor)

    def encode_orders(self, orders: dict, parts_pool: list,
                      num_machines: int, cutting_speed: float) -> np.ndarray:
        """无梯度的推理接口"""
        features = self._extract_features(orders, parts_pool, num_machines, cutting_speed)
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            m_s = self.encoder(feat_tensor).squeeze(0).numpy()
        return m_s.astype(np.float32)


class NestingResultEncoder(nn.Module):
    """
    Nesting agent 排样完成后，输出 8 维"排样结果摘要" m_n。

    改进：
    - 新增 forward() 支持梯度传播
    - 参数可纳入 Nesting agent 的优化器
    """

    def __init__(self, input_dim: int = 8, comm_dim: int = 8):
        super().__init__()
        self.comm_dim = comm_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, comm_dim),
            nn.Tanh(),
        )

    def _extract_features(self, history_plates: list, orders: dict,
                          parts_pool: list, plate_area: float) -> list:
        """提取 8 维手工特征"""
        if not history_plates:
            return [0.0] * self.input_dim

        utils = [p.utilization for p in history_plates]
        num_plates = len(history_plates)

        due_stds = []
        frag_ratios = []
        for plate in history_plates:
            if not plate.placed_parts:
                continue
            oids = [int(p[4]) for p in plate.placed_parts]
            dues = [orders[o]['due_date'] for o in oids if o in orders]
            if len(dues) > 1:
                due_stds.append(np.std(dues))
            for oid in set(oids):
                total = len([p for p in parts_pool if p['order_id'] == oid])
                on_this = oids.count(oid)
                frag_ratios.append(1.0 - on_this / max(1, total))

        total_part_area = sum(p['area'] for p in parts_pool)
        expected_plates = max(1.0, total_part_area / max(1.0, plate_area))

        features = [
            num_plates / max(1.0, expected_plates),
            np.mean(utils),
            np.std(utils) if len(utils) > 1 else 0.0,
            np.mean(due_stds) if due_stds else 0.0,
            np.std(due_stds) if len(due_stds) > 1 else 0.0,
            np.mean(frag_ratios) if frag_ratios else 0.0,
            min(utils) if utils else 0.0,
            num_plates / 10.0,
        ]
        return features

    def forward(self, feat_tensor: torch.Tensor) -> torch.Tensor:
        """有梯度的前向传播"""
        return self.encoder(feat_tensor)

    def encode_result(self, history_plates: list, orders: dict,
                      parts_pool: list, plate_area: float) -> np.ndarray:
        """无梯度的推理接口"""
        features = self._extract_features(history_plates, orders, parts_pool, plate_area)
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            m_n = self.encoder(feat_tensor).squeeze(0).numpy()
        return m_n.astype(np.float32)