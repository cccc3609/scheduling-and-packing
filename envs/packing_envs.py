import gymnasium as gym
import numpy as np
import copy
import math
from gymnasium import spaces

from heuristic.blf_skyline_maxrects import PlateLayoutManager
from heuristic.scheduler import SchedulerStateMachine
from config import MAX_PARTS_CAPACITY, TRAIN_CONFIG, MAX_SCHED_TASKS_CAPACITY, COST_CONFIG, FEATURE_CONFIG


class NestingSchedulingEnv(gym.Env):
    def __init__(self, plate_size=(200, 200)):
        super(NestingSchedulingEnv, self).__init__()

        self.max_capacity = MAX_PARTS_CAPACITY
        self.max_sched_capacity = MAX_SCHED_TASKS_CAPACITY
        self.current_num_parts = TRAIN_CONFIG['min_parts']

        # === 1. 物理参数 ===
        self.CUTTING_SPEED = COST_CONFIG['cutting_speed']
        self.COST_MAT = COST_CONFIG['cost_material']
        self.COST_HOLD = COST_CONFIG['cost_earliness']
        self.COST_TARD = COST_CONFIG['cost_tardiness']

        self.fixed_w, self.fixed_h = plate_size
        self.plate_w = 200
        self.plate_h = 200
        self.episode_time_scale = 100.0

        # === 2. 权重配置 ===
        self.w_util = 1.0
        self.w_jit = 12.0
        self.w_grouping = 0.5
        self.w_step_compact = 0.5
        self.w_new_plate = 2.0

        # === 3. 空间定义 ===
        # === 3. 空间定义 ===
        self.skyline_bins = FEATURE_CONFIG.get('skyline_bins', 20)
        self.obs_feature_dim = 26 + self.skyline_bins

        # 🔥 恢复：120个零件 × 2种旋转(0不转,1转) × 3种底层策略(BLF/Skyline/MaxRects) = 720维
        self.num_strategies = 3
        self.action_space = spaces.Discrete(self.max_capacity * 2 * self.num_strategies)

        self.observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'),
            shape=(self.max_capacity * self.obs_feature_dim,), dtype=np.float32
        )

        self.scheduler_state_machine = SchedulerStateMachine(num_machines=3)
        self.scheduler_model = None
        self.parts_pool = []
        self.orders = {}
        self.packed_indices = set()
        self.active_plates = []
        self.history_plates = []
        self.cost_metrics = {}

    def set_scheduling_partner(self, model):
        self.scheduler_model = model

    # 注意：这里的 def reset 必须和上面的 def set_scheduling_partner 头部对齐
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 动态解析参数或随机生成参数
        if options and 'num_parts' in options:
            self.current_num_parts = options['num_parts']
        else:
            min_p, max_p = TRAIN_CONFIG['min_parts'], TRAIN_CONFIG['max_parts']
            self.current_num_parts = np.random.randint(min_p, max_p + 1)

        self.current_num_parts = min(self.current_num_parts, self.max_capacity)

        if options and 'plate_size' in options:
            self.plate_w, self.plate_h = options['plate_size']
        else:
            min_d, max_d = TRAIN_CONFIG['min_plate_dim'], TRAIN_CONFIG['max_plate_dim']
            self.plate_w = np.random.randint(min_d, max_d + 1)
            self.plate_h = np.random.randint(min_d, max_d + 1)

        # 生成纯连续随机数据
        self.parts_pool, self.orders = self._generate_random_orders()

        # 初始化物理环境
        total_perimeter = sum([2 * (p['w'] + p['h']) for p in self.parts_pool])
        self.episode_time_scale = max(10.0, total_perimeter / self.CUTTING_SPEED)

        self.packed_indices = set()
        self.scheduler_state_machine.reset()
        self.active_plates = [PlateLayoutManager(width=self.plate_w, height=self.plate_h)]
        self.history_plates = []
        self.cost_metrics = {}

        # 5. 返回初始观察值和动作掩码字典 (符合 gymnasium 的标准 API)
        return self._get_obs(), {"action_mask": self._get_action_mask()}

    def _evaluate_placement_quality(self, plate, part_w, part_h, part_due, part_area, x, y, w, h, current_util):
        geo_score = current_util * 10.0
        touching = 0
        if x == 0: touching += 1
        if y == 0: touching += 1
        if x + w == self.plate_w: touching += 1
        if y + h == self.plate_h: touching += 1
        geo_score += touching * 0.2

        time_penalty = 0.0
        existing_parts = plate.placed_parts
        if existing_parts:
            oids = list(set([int(p[4]) for p in existing_parts]))
            dues = [self.orders[oid]['due_date'] for oid in oids if oid in self.orders]
            if dues:
                min_d, max_d = min(dues), max(dues)
                if part_due < min_d:
                    drop = (min_d - part_due) / self.episode_time_scale
                    time_penalty += drop * 15.0
                new_min = min(min_d, part_due)
                new_max = max(max_d, part_due)
                expansion = (new_max - new_min) - (max_d - min_d)
                time_penalty += (expansion / self.episode_time_scale) * 2.0

        return (self.w_util * geo_score) - (self.w_jit * time_penalty)

    def step(self, action):
        if isinstance(action, np.ndarray): action = int(action)

        # 🔥 恢复动作解码：解析出 策略ID、是否旋转、零件索引
        strategy_id = action % self.num_strategies
        is_rotated = (action // self.num_strategies) % 2
        part_index = action // (self.num_strategies * 2)

        if part_index >= self.current_num_parts or part_index in self.packed_indices:
            return self._get_obs(), -100.0, True, False, {}

        part = self.parts_pool[part_index]
        self.packed_indices.add(part_index)
        reward = 0.0

        # 提取当前零件的宽高，如果 RL 决定旋转，则交换宽高
        part_w, part_h = part['w'], part['h']
        if is_rotated == 1:
            part_w, part_h = part['h'], part['w']

        best_idx, best_score = -1, -float('inf')

        # 遍历所有激活的板材，寻找最佳放置位置
        for idx, plate in enumerate(self.active_plates):
            sim_plate = copy.deepcopy(plate)
            # 严格使用 RL 选定的策略和旋转后的宽高
            ok, sx, sy, sw, sh, _ = sim_plate.place_part(part_w, part_h, part['order_id'], strategy_id)
            if ok:
                score = self._evaluate_placement_quality(
                    plate, part_w, part_h, part['due_date'], part['area'],
                    sx, sy, sw, sh, sim_plate.utilization
                )
                if score > best_score:
                    best_score, best_idx = score, idx

        # ==========================================================
        # ⚠️ 修复了缩进：以下动作执行逻辑必须在 for 循环的外部！
        # ==========================================================
        if best_idx != -1:
            target = self.active_plates[best_idx]
            old_util = target.utilization  # 记录排入前的利用率

            s, x, y, w, h, _ = target.place_part(part_w, part_h, part['order_id'], strategy_id)
            # 如果 RL 选的策略排不下（被干涉），触发兜底机制：强制用最保守的 MaxRects (策略2) 补救
            if not s:
                s, x, y, w, h, _ = target.place_part(part_w, part_h, part['order_id'], 2)

            new_util = target.utilization  # 记录排入后的利用率

            # ==========================================================
            # 🔥 空间密集奖励 1：缝隙填得越满，立刻给加分！
            # ==========================================================
            reward += (new_util - old_util) * 20.0

            # ==========================================================
            # 🔥 时间密集惩罚：(保留项目原有的核心灵魂，防止急缓混排)
            # ==========================================================
            oids = list(set([int(p[4]) for p in target.placed_parts]))
            if len(oids) > 1:
                dues = [self.orders[o]['due_date'] for o in oids if o in self.orders]
                if len(dues) > 1:
                    due_std = np.std(dues)
                    step_jit_penalty = (due_std / max(1.0, self.episode_time_scale)) * self.w_grouping
                    reward -= step_jit_penalty

            # 原有的紧凑度奖励代码
            cx_sum, cy_sum, count = 0, 0, 0
            for p in target.placed_parts:
                cx_sum += p[0] + p[2] / 2
                cy_sum += p[1] + p[3] / 2
                count += 1
            curr_cx, curr_cy = x + w / 2, y + h / 2
            max_d = (self.plate_w ** 2 + self.plate_h ** 2) ** 0.5
            if count > 1:
                d = ((curr_cx - cx_sum / count) ** 2 + (curr_cy - cy_sum / count) ** 2) ** 0.5
                reward += (1.0 - d / max_d) * self.w_step_compact
            else:
                d = (curr_cx ** 2 + curr_cy ** 2) ** 0.5
                reward += (1.0 - d / max_d) * self.w_step_compact

        else:
            # ============================================================
            # 🔥 空间密集惩罚 2 (核武器)：开新板子时的“废料暴击”！
            # ============================================================
            if self.active_plates:
                # 当 RL 决定开新板子时，意味着上一张板子被永远“封印”了。
                # 此时立刻清算上一张板子的浪费率！
                last_plate = self.active_plates[-1]
                waste_ratio = 1.0 - last_plate.utilization

                # 如果上一张板子还空着 40%，RL 却不去选“填缝件”，而是强行开新板，立刻遭受重罚！
                reward -= waste_ratio * 50.0

            new_plate = PlateLayoutManager(width=self.plate_w, height=self.plate_h)
            s, x, y, w, h, _ = new_plate.place_part(part_w, part_h, part['order_id'], strategy_id)
            if s:
                self.active_plates.append(new_plate)
                reward -= self.w_new_plate
            else:
                reward -= 50.0

        terminated = len(self.packed_indices) == self.current_num_parts
        info = {}  # ⚠️ 修复：必须初始化 info 字典

        if terminated:
            self.history_plates = self.active_plates
            final_plates = [p for p in self.history_plates if len(p.placed_parts) > 0]

            # 1. 物理面积统计
            total_part_area = sum([p['area'] for p in self.parts_pool])
            consumed_area = len(final_plates) * (self.plate_w * self.plate_h)
            utilization = total_part_area / consumed_area if consumed_area > 0 else 0.001

            wasted_area = max(0.0, consumed_area - total_part_area)
            cost_material = wasted_area * self.COST_MAT  # 这里的成本变成了“纯浪费罚款”

            # --- 下游：调度与 JIT 成本计算 ---
            order_finishes = self._simulate_with_edd(final_plates)

            cost_jit, total_delay = 0.0, 0.0
            for oid, finish_time in order_finishes.items():
                due = self.orders[oid]['due_date']
                order_parts = [p for p in self.parts_pool if p['order_id'] == oid]
                order_value = sum([p['area'] for p in order_parts])
                order_proc_time = max(1.0, sum([2 * (p['w'] + p['h']) for p in order_parts]) / self.CUTTING_SPEED)

                diff = finish_time - due
                abs_diff = abs(diff)
                ratio = abs_diff / order_proc_time
                R_FREE, R_FULL = 0.025, 0.075

                if ratio <= R_FREE:
                    coef = 0.0
                elif ratio <= R_FULL:
                    coef = (ratio - R_FREE) / (R_FULL - R_FREE)
                else:
                    coef = 1.0

                if diff > 0:
                    cost_jit += order_value * (self.COST_TARD * coef) * abs_diff
                    total_delay += diff
                else:
                    cost_jit += order_value * (self.COST_HOLD * coef) * abs_diff

            # --- 终局 Reward 计算 ---
            # 现在的 total_cost 是纯纯的“罚款总和”（浪费的材料钱 + 迟到的违约金）
            total_penalty_cost = cost_material + cost_jit

            # 基准尺度：用订单的固有材料总价值作为分母，用于归一化
            intrinsic_value = max(1.0, total_part_area * self.COST_MAT)

            # 损失率 = 总罚款 / 固有总价值
            # (如果损失率为0，说明0浪费且0迟到；如果为1.0，说明浪费和违约金抵得上这批货本身的价值了)
            penalty_ratio = total_penalty_cost / intrinsic_value

            if penalty_ratio <= 1.0:
                # 损失在 100% 以内：线性给分 (损失为0得20分，损失0.5得0分，损失1.0得-20分)
                scaled_reward = (0.5 - penalty_ratio) * 40.0
            else:
                # 损失惨重：启动对数平滑防梯度消失
                scaled_reward = -20.0 - math.log(max(1e-5, penalty_ratio)) * 10.0

            scaled_reward = np.clip(scaled_reward, -100.0, 50.0)
            reward += scaled_reward

            self.cost_metrics = {
                "cost_material": cost_material,  # 记录为浪费成本
                "cost_jit": cost_jit,
                "cost_total": total_penalty_cost,
                "utilization": utilization,
                "plate_count": len(final_plates),
                "total_delay": total_delay,
                "norm_reward": scaled_reward
            }
            info["episode_metrics"] = self._compute_metrics()

        info["action_mask"] = self._get_action_mask()
        return self._get_obs(), reward, terminated, False, info

    def _simulate_with_edd(self, plates_list):
        """
        固定 EDD（最早交期优先）启发式调度，用于 nesting 终局奖励计算。
        使用局部 orders 副本，绝不修改 self.orders。
        信号稳定，不随 scheduling agent 训练状态变化。
        """
        import copy
        local_orders = {oid: {'due_date': v['due_date'], 'finished_time': 0.0}
                        for oid, v in self.orders.items()}

        tasks = []
        for idx, plate in enumerate(plates_list):
            cut_time = sum([2 * (p[2] + p[3]) for p in plate.placed_parts]) / self.CUTTING_SPEED
            oids = list(set([int(p[4]) for p in plate.placed_parts]))
            min_due = min([local_orders[o]['due_date'] for o in oids if o in local_orders],
                          default=999.0)
            tasks.append({'cut': cut_time, 'due': min_due, 'oids': oids})

        # EDD 排序后贪心分配最早空闲机器
        tasks.sort(key=lambda t: t['due'])
        machine_times = [0.0] * self.scheduler_state_machine.num_machines

        for task in tasks:
            m_idx = int(min(range(len(machine_times)), key=lambda m: machine_times[m]))
            end_t = machine_times[m_idx] + task['cut']
            machine_times[m_idx] = end_t
            for oid in task['oids']:
                if oid in local_orders:
                    local_orders[oid]['finished_time'] = max(local_orders[oid]['finished_time'], end_t)

        return {oid: v['finished_time'] for oid, v in local_orders.items()}




        # 替换 packing_envs.py 中的数据生成逻辑，完全移除分类概念
    def _generate_random_orders(self):
            data, orders = [], {}
            cnt, oid = 0, 0

            # 估算总加工量，用于界定时间窗的基础尺度
            avg_perim = 2 * (0.25 * self.plate_w + 0.25 * self.plate_h)
            total_work = self.current_num_parts * avg_perim
            est_makespan = (total_work / max(0.1, self.CUTTING_SPEED) / 3) * 1.3

            while cnt < self.current_num_parts:
                # 随机决定这个订单包含几个零件 (1 到 15 个不等)
                batch = np.random.randint(1, 16)
                if cnt + batch > self.current_num_parts:
                    batch = self.current_num_parts - cnt

                temp_parts = []
                order_p = 0

                for _ in range(batch):
                    # 连续随机分布生成长宽比例 (不设定具体的形状类别)
                    # 使用 Beta 分布让长宽更偏向于真实的工业零件分布（小件偏多，大件偏少）
                    w_ratio = np.random.beta(a=2, b=5) * 0.9 + 0.05
                    h_ratio = np.random.beta(a=2, b=5) * 0.9 + 0.05

                    # 随机翻转长宽
                    if np.random.rand() > 0.5:
                        w_ratio, h_ratio = h_ratio, w_ratio

                    w = max(1, int(w_ratio * self.plate_w))
                    h = max(1, int(h_ratio * self.plate_h))

                    order_p += 2 * (w + h)
                    temp_parts.append({'w': w, 'h': h, 'area': w * h})

                # 计算该订单自身绝对需要的切割时间
                self_time = order_p / max(0.1, self.CUTTING_SPEED)

                poisson_shifts = np.random.poisson(lam=2.0)
                buffer_time = (poisson_shifts + 0.1) * (est_makespan / 2.0)

                final_due = self_time + buffer_time

                # 不带任何类型标签，纯净记录
                orders[oid] = {'due_date': final_due, 'finished_time': 0.0}

                for p in temp_parts:
                    data.append({
                        'w': p['w'], 'h': p['h'], 'area': p['area'],
                        'due_date': final_due, 'order_id': oid,
                        'original_idx': cnt
                    })
                    cnt += 1

                oid += 1

            np.random.shuffle(data)
            return data, orders
    def _compute_metrics(self):
        m = self.cost_metrics.copy()
        m['late_orders_count'] = sum([1 for o in self.orders.values() if o['finished_time'] > o['due_date']])
        if self.history_plates:
            utils = [p.utilization for p in self.history_plates]
            m['raw_avg_utilization'] = np.mean(utils)
            m['adj_avg_utilization'] = np.mean(sorted(utils)[1:]) if len(utils) > 1 else np.mean(utils)
        else:
            m['raw_avg_utilization'] = 0.0;
            m['adj_avg_utilization'] = 0.0
        return m

    def _get_obs(self):
        obs = np.zeros((self.max_capacity, self.obs_feature_dim), dtype=np.float32)
        skyline_feat = np.zeros(self.skyline_bins, dtype=np.float32)
        act_util_avg, act_util_max, act_util_min = 0.0, 0.0, 0.0
        act_free_area_ratio, act_max_free_area, max_free_w, max_free_h = 0.0, 0.0, 0.0, 0.0

        if self.active_plates:
            target_plate = self.active_plates[-1]
            skyline_feat = target_plate.get_normalized_skyline(self.skyline_bins)
            utils = [p.utilization for p in self.active_plates]
            act_util_avg, act_util_max, act_util_min = np.mean(utils), np.max(utils), np.min(utils)

            all_free_rects = [r for p in self.active_plates for r in p.free_rects]
            total_active_area = self.plate_w * self.plate_h * len(self.active_plates)

            if all_free_rects and total_active_area > 0:
                act_free_area_ratio = sum([r[2] * r[3] for r in all_free_rects]) / total_active_area
                act_max_free_area = max([r[2] * r[3] for r in all_free_rects]) / (self.plate_w * self.plate_h)
                max_free_w = max([r[2] for r in all_free_rects]) / self.plate_w
                max_free_h = max([r[3] for r in all_free_rects]) / self.plate_h

        act_cnt_norm = len(self.active_plates) / 10.0
        rem_idxs = [i for i in range(self.current_num_parts) if i not in self.packed_indices]
        mach_times = self.scheduler_state_machine.get_state()
        curr_time = np.min(mach_times) if len(mach_times) > 0 else 0.0
        safe_time_scale = max(1.0, self.episode_time_scale)

        if rem_idxs:
            rem_areas = [self.parts_pool[i]['area'] for i in rem_idxs]
            rem_dues = [self.parts_pool[i]['due_date'] for i in rem_idxs]
            plate_area_unit = self.plate_w * self.plate_h
            rem_avg_area, rem_max_area = np.mean(rem_areas) / plate_area_unit, np.max(rem_areas) / plate_area_unit
            rem_total_ratio = (sum(rem_areas) / plate_area_unit) / 10.0
            rem_due_diffs = [(d - curr_time) / safe_time_scale for d in rem_dues]
            rem_avg_due, rem_min_due, rem_due_std = np.mean(rem_due_diffs), np.min(rem_due_diffs), np.std(rem_due_diffs)
        else:
            rem_avg_area, rem_max_area, rem_total_ratio, rem_avg_due, rem_min_due, rem_due_std = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        progress = len(self.packed_indices) / max(1, self.current_num_parts)

        if self.scheduler_model:
            m_rel = (mach_times - curr_time) / safe_time_scale
            mach_load_avg, mach_load_std = np.mean(m_rel), np.std(m_rel)
        else:
            mach_load_avg, mach_load_std = 0.0, 0.0

        act_due_std, order_frag_idx, proj_tardiness, mach_idle_gap = 0.0, 0.0, 0.0, 0.0
        if self.active_plates and self.active_plates[-1].placed_parts:
            last_plate = self.active_plates[-1]
            last_plate_oids = [int(p[4]) for p in last_plate.placed_parts]
            last_dues = [self.orders[o]['due_date'] for o in last_plate_oids if o in self.orders]

            if len(last_dues) > 1: act_due_std = np.std(last_dues) / safe_time_scale

            frag_ratios = []
            for oid in list(set(last_plate_oids)):
                total_p = len([p for p in self.parts_pool if p['order_id'] == oid])
                packed_p = last_plate_oids.count(oid)
                frag_ratios.append(1.0 - packed_p / max(1, total_p))
            if frag_ratios: order_frag_idx = np.mean(frag_ratios)

            cut_time = sum([2 * (p[2] + p[3]) for p in last_plate.placed_parts]) / max(0.1, self.CUTTING_SPEED)
            if last_dues:
                risk = (curr_time + cut_time - min(last_dues)) / safe_time_scale
                proj_tardiness = max(0.0, risk)

        if len(mach_times) > 0: mach_idle_gap = (np.max(mach_times) - curr_time) / safe_time_scale

        global_feats = [
            progress, rem_avg_area, rem_max_area, rem_total_ratio,
            rem_avg_due, rem_min_due, rem_due_std, act_cnt_norm, act_util_avg, act_util_max, act_util_min,
            act_free_area_ratio, act_max_free_area, max_free_w, max_free_h, mach_load_avg, mach_load_std,
            act_due_std, order_frag_idx, proj_tardiness, mach_idle_gap
        ]

        skyline_list = skyline_feat.tolist()

        for i in range(self.max_capacity):
            if i < self.current_num_parts:
                part = self.parts_pool[i]
                is_packed = 1.0 if i in self.packed_indices else 0.0
                local_feats = [part['w'] / self.plate_w, part['h'] / self.plate_h,
                               part['area'] / (self.plate_w * self.plate_h),
                               (part['due_date'] - curr_time) / safe_time_scale, is_packed]
                obs[i] = np.array(local_feats + global_feats + skyline_list, dtype=np.float32)
        return obs.flatten()

    def _get_action_mask(self):
        total_actions = self.max_capacity * 6
        mask = __import__('numpy').zeros(total_actions, dtype=bool)
        import numpy as _np

        if len(self.packed_indices) >= self.current_num_parts:
            return _np.ones(total_actions, dtype=bool)

        unpacked_idxs = [i for i in range(self.current_num_parts) if i not in self.packed_indices]

        # 战术 1: FFD — 面积最大的 2 个（奠基件）
        top_area_idxs = sorted(unpacked_idxs, key=lambda i: self.parts_pool[i]['area'], reverse=True)[:2]

        # 战术 2: EDD — 交期最急的 2 个（保交期）
        top_due_idxs = sorted(unpacked_idxs, key=lambda i: self.parts_pool[i]['due_date'])[:2]

        # 战术 3: 填缝 — 当前板材最大空矩形能容纳的件中，面积最大的 2 个
        # ✅ 修复：从"全局最小"改为"当前空间可容纳"
        bottom_area_idxs = []
        if self.active_plates and self.active_plates[-1].free_rects:
            free_rects = self.active_plates[-1].free_rects
            max_free_w = max(r[2] for r in free_rects)
            max_free_h = max(r[3] for r in free_rects)
            # 能放入（考虑旋转）的候选件
            fittable = [
                i for i in unpacked_idxs
                if (self.parts_pool[i]['w'] <= max_free_w and self.parts_pool[i]['h'] <= max_free_h)
                   or (self.parts_pool[i]['h'] <= max_free_w and self.parts_pool[i]['w'] <= max_free_h)
            ]
            bottom_area_idxs = sorted(fittable, key=lambda i: self.parts_pool[i]['area'], reverse=True)[:2]

        # fallback：板材全满时回退到全局最小面积
        if not bottom_area_idxs:
            bottom_area_idxs = sorted(unpacked_idxs, key=lambda i: self.parts_pool[i]['area'])[:2]

        candidate_set = set(top_area_idxs + top_due_idxs + bottom_area_idxs)

        for i in candidate_set:
            mask[i * 6: (i + 1) * 6] = True

        return mask