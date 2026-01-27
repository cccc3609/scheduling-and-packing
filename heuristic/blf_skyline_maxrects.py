import numpy as np


class PlateLayoutManager:
    """
    高级排样管理器 (基于 Wei et al. 2011 论文改进)
    集成了:
    1. BLF (Bottom-Left Fill): 改进版，能回溯填补内部空洞。
    2. Skyline (Wei's Heuristic): 基于最小浪费(Min Waste)和最大契合度(Max Fitness)的评分策略。
    3. MaxRects (Best Area Fit): 经典的分割空闲矩形算法，增加了短边适应规则。
    """

    def __init__(self, width=1.0, height=1.0):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        # 通用状态
        self.placed_parts = []  # (x, y, w, h, order_id, is_rotated)
        self.used_area = 0.0

        # --- 算法特定状态 ---

        # MaxRects: 空闲矩形列表
        self.free_rects = [(0.0, 0.0, self.width, self.height)]

        # Skyline: 水平线段列表 [(x, y, width), ...]，始终按 x 排序
        # 初始时只有一条底边
        self.skyline = [(0.0, 0.0, self.width)]

        # BLF: 候选点列表 [(x, y), ...]
        self.blf_points = [(0.0, 0.0)]

    def get_normalized_skyline(self, num_bins=20):
        """
        将连续的天际线离散化为固定长度的向量。
        :param num_bins: 分辨率 (例如20)
        :return: np.array [num_bins], 值在 0~1 之间 (相对于板高)
        """
        height_map = np.zeros(num_bins, dtype=np.float32)
        bin_width = self.width / num_bins

        # Skyline 是 [(x, y, width), ...]
        for sx, sy, sw in self.skyline:
            # 计算该线段覆盖了哪些 bin
            start_idx = int(sx / bin_width)
            end_idx = int((sx + sw) / bin_width)

            # 防止浮点误差导致的越界
            end_idx = min(end_idx, num_bins)

            # 归一化高度
            norm_h = sy / self.height

            # 填充高度图
            if start_idx < end_idx:
                height_map[start_idx: end_idx] = norm_h
            else:
                # 处理极短线段落在单个bin内的情况
                if start_idx < num_bins:
                    height_map[start_idx] = max(height_map[start_idx], norm_h)

        return height_map

    def place_part(self, part_w, part_h, order_id, strategy_id, min_rem_w=0.0, min_rem_h=0.0):
        """
        尝试放置零件。
        参数:
            min_rem_w, min_rem_h: 剩余所有待排零件中的最小宽和高。
            用于 Skyline 算法判断形成的空隙是否为"死区(Waste)"。
        """
        # 1. 尝试不旋转
        # 传入剩余最小尺寸用于高级评估
        res_norm = self._try_strategies(part_w, part_h, strategy_id, min_rem_w, min_rem_h)

        # 2. 尝试旋转
        res_rot = None
        if part_w != part_h:
            # 旋转后，宽变高，高变宽。注意 min_rem 不需要变，因为那是全局统计
            res_rot = self._try_strategies(part_h, part_w, strategy_id, min_rem_w, min_rem_h)

        final_x, final_y, is_rotated = None, None, False

        # 3. 决策：选择最佳姿态
        # 如果两种姿态都可行，需要比较它们的得分 (对于 Skyline 来说 res 中包含了 score)
        # 这里简化处理：如果 strategy 是 Skyline，我们对比 waste；否则优先不旋转

        candidates = []
        if res_norm: candidates.append((*res_norm, False))
        if res_rot: candidates.append((*res_rot, True))

        if not candidates:
            return False, None, None, None, None, False

        if strategy_id == 1:  # Skyline 使用了复杂的评分 (Score, Fitness, Y, X)
            # res 格式: (x, y, waste, fitness)
            # 排序逻辑: Waste 越小越好 -> Fitness 越大越好 -> Y 越小越好 -> X 越小越好
            # Python sort 是升序，所以 Fitness 取负
            candidates.sort(key=lambda p: (p[2], -p[3], p[1], p[0]))
        else:
            # BLF 和 MaxRects 依然遵循 Y 优先原则
            candidates.sort(key=lambda p: (p[1], p[0]))

        final_x, final_y = candidates[0][0], candidates[0][1]
        is_rotated = candidates[0][-1]

        # 4. 执行放置
        placed_w = part_h if is_rotated else part_w
        placed_h = part_w if is_rotated else part_h

        self.placed_parts.append((final_x, final_y, placed_w, placed_h, order_id, is_rotated))
        self.used_area += placed_w * placed_h

        # === 同步更新所有算法状态 ===
        rect = (final_x, final_y, placed_w, placed_h)
        self._update_maxrects(rect)
        self._update_skyline(rect)
        self._update_blf(rect)

        return True, final_x, final_y, placed_w, placed_h, is_rotated

    def _try_strategies(self, w, h, strategy_id, min_w, min_h):
        if strategy_id == 0:
            return self._find_blf(w, h)
        elif strategy_id == 1:
            return self._find_skyline_wei(w, h, min_w, min_h)
        elif strategy_id == 2:
            return self._find_maxrects(w, h)
        return None

    # ==========================================
    # 策略 0: Improved BLF with Fill
    # ==========================================
    def _find_blf(self, w, h):
        for x, y in self.blf_points:
            if x + w > self.width or y + h > self.height: continue
            if not self._check_overlap((x, y, w, h)):
                return x, y
        return None

    def _update_blf(self, rect):
        px, py, pw, ph = rect
        # 移除被覆盖的点
        self.blf_points = [p for p in self.blf_points if not (px <= p[0] < px + pw and py <= p[1] < py + ph)]
        # 添加新点
        new_candidates = [(px + pw, py), (px, py + ph)]
        for nx, ny in new_candidates:
            if nx < self.width and ny < self.height:
                if not self._check_is_covered(nx, ny):
                    self.blf_points.append((nx, ny))
        self.blf_points.sort(key=lambda p: (p[1], p[0]))

    def _check_is_covered(self, x, y):
        for (px, py, pw, ph, _, _) in self.placed_parts:
            if px <= x < px + pw and py <= y < py + ph: return True
        return False

    # ==========================================
    # 策略 1: Skyline (Wei et al. 2011 Heuristic)
    # ==========================================
    def _find_skyline_wei(self, w, h, min_rem_w, min_rem_h):
        """
        论文核心评估函数：
        1. 寻找所有能够支撑零件宽度的 Skyline 片段。
        2. 计算 Local Waste (局部浪费) 和 Fitness Number (契合度)。
        3. 返回最佳位置及其得分。
        """
        best_waste = float('inf')
        best_fitness = -1
        best_pos = None  # (x, y, waste, fitness)

        # 遍历每一个线段作为潜在的左下角起始点
        for i, (sx, sy, sw) in enumerate(self.skyline):

            # 1. 检查宽度可行性 (需要连续的线段总长 >= w)
            available_w = 0.0
            idx = i
            valid_start = True

            # 累加后续等高线段
            while idx < len(self.skyline) and self.skyline[idx][1] == sy:
                # 检查连续性 (当前线段起点 == 上一线段终点)
                if idx > i:
                    prev = self.skyline[idx - 1]
                    if abs((prev[0] + prev[2]) - self.skyline[idx][0]) > 1e-5:
                        break  # 断开了

                available_w += self.skyline[idx][2]
                if available_w >= w:
                    break
                idx += 1

            if available_w < w:
                continue  # 宽度不够

            # 2. 检查高度可行性
            if sy + h > self.height:
                continue

            # 3. 检查悬空重叠 (Double check)
            # Skyline 简化模型可能忽略上方悬空的零件，必须严格检查
            if self._check_overlap((sx, sy, w, h)):
                continue

            # === 4. 计算评分 (Paper Fig. 4 & 5) ===
            waste = 0.0
            fitness = 0

            # A. Wasted space to the right (放置后该线段剩余的长度)
            # 如果剩余长度小于最小零件宽，则视为浪费
            remainder = available_w - w
            if 0 < remainder < min_rem_w:
                waste += remainder * min_rem_h  # 假设这块区域高度至少为 min_h 也没法用

            # B. Wasted space to the left (左邻居)
            # 获取左边的线段高度
            h_left = self.height  # 默认边界墙
            if i > 0:
                # 左邻居是上一条线段
                prev_s = self.skyline[i - 1]
                # 必须紧贴着才算邻居
                if abs((prev_s[0] + prev_s[2]) - sx) < 1e-5:
                    h_left = prev_s[1]

            # 适应度计算：左边贴合
            if abs((sy + h) - h_left) < 1e-5:
                fitness += 1
            elif h_left > sy:
                # 左边比我高，形成了一个垂直面
                # 检查是否完全贴合
                if h == h_left - sy:
                    fitness += 1
                # 检查是否形成死区 (Paper Fig 4b)
                # 如果 (左墙高度 - 我放置后的高度) < 最小零件高 -> 上方形成废料
                gap = h_left - (sy + h)
                if 0 < gap < min_rem_h:
                    waste += gap * min_rem_w  # 估算废料面积

            # C. Wasted space to the right neighbor (右邻居)
            # 找到放置位置右侧紧邻的线段
            # 放置结束的X坐标
            end_x = sx + w
            h_right = self.height  # 默认边界墙

            # 在 skyline 中找 end_x 对应的位置
            # 由于我们可能跨越了多个线段，需要找覆盖 end_x 的那个线段的下一个
            # 或者简单的，如果正好填满 available_w，看下一个线段

            # 简化逻辑：找 x >= end_x 的第一条线段
            for r_idx in range(i, len(self.skyline)):
                s_curr = self.skyline[r_idx]
                if s_curr[0] >= end_x - 1e-5:  # 找到了右侧邻居
                    if abs(s_curr[0] - end_x) < 1e-5:  # 紧贴
                        h_right = s_curr[1]
                    break

            # 适应度计算：右边贴合
            if abs((sy + h) - h_right) < 1e-5:
                fitness += 1
            elif h_right > sy:
                gap = h_right - (sy + h)
                if 0 < gap < min_rem_h:
                    waste += gap * min_rem_w

            # D. Bottom Fit
            # 如果放置宽度正好等于线段宽度 (Exact match)
            if abs(w - self.skyline[i][2]) < 1e-5:
                fitness += 1

            # E. Top Fit (触顶)
            if abs((sy + h) - self.height) < 1e-5:
                fitness += 1

            # === 5. 更新最佳解 ===
            # 优先级: Waste 越小越好 > Fitness 越大越好 > Y 越低越好

            # 如果还没找到解，或者找到了更好的
            if best_pos is None:
                best_pos = (sx, sy, waste, fitness)
                best_waste = waste
                best_fitness = fitness
            else:
                # Lexicographical comparison
                if waste < best_waste - 1e-5:  # 显著更小的浪费
                    best_pos = (sx, sy, waste, fitness)
                    best_waste = waste
                    best_fitness = fitness
                elif abs(waste - best_waste) < 1e-5:
                    if fitness > best_fitness:
                        best_pos = (sx, sy, waste, fitness)
                        best_fitness = fitness
                    elif fitness == best_fitness:
                        # Tie-breaker: lowest Y (already iterated by Y implicitly if sorted? No, sorted by X)
                        if sy < best_pos[1]:
                            best_pos = (sx, sy, waste, fitness)

        if best_pos:
            return best_pos  # (x, y, waste, fitness)
        return None

    def _update_skyline(self, rect):
        px, py, pw, ph = rect
        new_skyline = []
        p_right = px + pw
        p_top = py + ph

        for sx, sy, sw in self.skyline:
            s_right = sx + sw

            # 完全不相交
            if s_right <= px or sx >= p_right:
                new_skyline.append((sx, sy, sw))
                continue

            # 相交，被覆盖的部分被切除/抬升
            # 左侧残留
            if sx < px:
                new_skyline.append((sx, sy, px - sx))
            # 右侧残留
            if s_right > p_right:
                new_skyline.append((p_right, sy, s_right - p_right))

        # 添加新顶边
        new_skyline.append((px, p_top, pw))

        # 排序与合并
        new_skyline.sort(key=lambda s: s[0])
        merged = []
        for seg in new_skyline:
            if not merged:
                merged.append(seg)
            else:
                last = merged[-1]
                # 必须高度相同，且无缝连接
                if abs(last[1] - seg[1]) < 1e-5 and abs((last[0] + last[2]) - seg[0]) < 1e-5:
                    merged[-1] = (last[0], last[1], last[2] + seg[2])
                else:
                    merged.append(seg)
        self.skyline = merged

    # ==========================================
    # 策略 2: MaxRects (Best Area Fit + Short Side Fit)
    # ==========================================
    def _find_maxrects(self, w, h):
        best_idx = -1
        best_fit = float('inf')  # Primary: Area fit
        best_ssf = float('inf')  # Secondary: Short Side Fit

        for i, (fx, fy, fw, fh) in enumerate(self.free_rects):
            if fw >= w and fh >= h:
                area_fit = (fw * fh) - (w * h)
                short_side_fit = min(fw - w, fh - h)

                if area_fit < best_fit:
                    best_fit = area_fit
                    best_ssf = short_side_fit
                    best_idx = i
                elif abs(area_fit - best_fit) < 1e-5:
                    # Tie-breaker
                    if short_side_fit < best_ssf:
                        best_ssf = short_side_fit
                        best_idx = i

        if best_idx != -1: return self.free_rects[best_idx][0], self.free_rects[best_idx][1]
        return None

    def _update_maxrects(self, rect):
        px, py, pw, ph = rect
        new_free = []
        for fx, fy, fw, fh in self.free_rects:
            if not (px + pw <= fx or px >= fx + fw or py + ph <= fy or py >= fy + fh):
                if px < fx + fw and px + pw > fx:
                    if py + ph < fy + fh: new_free.append((fx, py + ph, fw, fy + fh - (py + ph)))
                if px < fx + fw and px + pw > fx:
                    if py > fy: new_free.append((fx, fy, fw, py - fy))
                if py < fy + fh and py + ph > fy:
                    if px > fx: new_free.append((fx, fy, px - fx, fh))
                if py < fy + fh and py + ph > fy:
                    if px + pw < fx + fw: new_free.append((px + pw, fy, fx + fw - (px + pw), fh))
            else:
                new_free.append((fx, fy, fw, fh))

        # 剪枝优化
        new_free.sort(key=lambda r: r[2] * r[3], reverse=True)
        final_free = []
        for i, r1 in enumerate(new_free):
            is_contained = False
            for j, r2 in enumerate(final_free):
                if self._is_contained(r1, r2):
                    is_contained = True
                    break
            if not is_contained: final_free.append(r1)
        self.free_rects = final_free

    def _check_overlap(self, rect):
        x, y, w, h = rect
        eps = 1e-5
        for (px, py, pw, ph, _, _) in self.placed_parts:
            if not (x + w <= px + eps or x >= px + pw - eps or y + h <= py + eps or y >= py + ph - eps):
                return True
        return False

    def _is_contained(self, r1, r2):
        return r1[0] >= r2[0] and r1[1] >= r2[1] and r1[0] + r1[2] <= r2[0] + r2[2] and r1[1] + r1[3] <= r2[1] + r2[3]

    @property
    def utilization(self):
        return self.used_area / (self.width * self.height)