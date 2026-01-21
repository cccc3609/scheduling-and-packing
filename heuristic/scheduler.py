import numpy as np


class SchedulerStateMachine:
    """
    调度状态机：只负责维护机器状态和记录日志，决策权移交给 Agent。
    """

    def __init__(self, num_machines=3):
        self.num_machines = num_machines
        self.reset()

    def reset(self):
        self.machines_avail_time = np.zeros(self.num_machines)
        self.log = []

    def execute_assignment(self, machine_idx, cut_path_length, plate_idx):
        """
        执行 Agent 下达的指令
        """
        start_time = self.machines_avail_time[machine_idx]
        processing_time = cut_path_length + 0.1
        end_time = start_time + processing_time

        # 更新机器状态
        self.machines_avail_time[machine_idx] = end_time

        # 记录日志
        self.log.append({
            "machine_id": int(machine_idx),
            "start": float(start_time),
            "end": float(end_time),
            "plate_idx": int(plate_idx)
        })

        return end_time

    def get_state(self):
        """返回机器当前的可用时间"""
        return self.machines_avail_time.copy()