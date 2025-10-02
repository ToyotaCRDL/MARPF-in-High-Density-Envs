import copy
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from solver.a_star import AStarFollowingConflict
from pypibt import PIBT


class PathFindingDenseEnv:
    def __init__(
        self, 
        size_x: int, 
        size_y: int, 
        static_obstacles: list[tuple[int, int]] = None,
        max_calc_time: int = 180
    ):
        """
        This class plans target paths for multiple agents and racks while avoiding collisions.

        Args:
            size_x (int): Grid width.
            size_y (int): Grid height.
            static_obstacles (list[tuple[int, int]] | None): Coordinates that are
                permanently blocked.
        """
        if static_obstacles is None:
            static_obstacles = []

        self.size_x = size_x
        self.size_y = size_y
        self.dimension = (size_x, size_y)
        self.static_obss = static_obstacles

        self.ag_start_pos_lst = []
        self.rk_start_pos_lst = []
        self.rk_cur_pos_lst = []
        self.null_ag_pos_lst = []
        
        self.all_posible_pos_lst = [
            (x, y)
            for x in range(self.size_x)
            for y in range(self.size_y)
            if (x, y) not in self.static_obss
        ]

        self.tgt_rk_goal_pos_lst = []
        self.tgt_rk_num = None

        self.tgt_rk_path_lst = []
        self.tgt_rk_path_lst_plot = []
        self.num_of_obs_racks_on_path = []

        self.rk_transport_task_lst = []
        
        self.time_count_start = time.time()
        self.max_calc_time = max_calc_time
        self.timed_out = False
    
    def _update_null_ag_pos_lst(self) -> None:
        """Update the list of positions currently available to null agents."""
        self.null_ag_pos_lst = list(
            set(self.all_posible_pos_lst) - set(self.rk_cur_pos_lst)
        )

    def _get_closest_null_ag_pos(
        self, 
        key_pos: tuple[int, int]
    ) -> None:
        """Return index and position of the closest null agent to key_pos."""
        self._update_null_ag_pos_lst()
        distances = np.abs(np.array(self.null_ag_pos_lst) - np.array(key_pos)).sum(
            axis=1
        )
        min_index = np.argmin(distances)
        return min_index, self.null_ag_pos_lst[min_index]

    def _update_goal_rk(self) -> None:
        """Update indices of racks that have reached their goals."""
        self.goal_rk_idx_lst = [
            i_tgt_rk
            for i_tgt_rk in range(self.tgt_rk_num)
            if self.tgt_cur_steps[i_tgt_rk]
            == len(self.tgt_rk_path_lst[i_tgt_rk]) - 1
        ]

    def _is_num_of_obs_racks_on_path_increased(self) -> bool:
        """
        Check whether the number of obstructing racks along each target path
        increased compared to the previous check.
        """
        num_prev = copy.deepcopy(self.num_of_obs_racks_on_path)

        num_new = []
        for i_tgt in range(len(self.tgt_rk_path_lst)):
            num_of_obs_racks = 0
            for i in range(self.tgt_cur_steps[i_tgt], len(self.tgt_rk_path_lst[i_tgt]) - 1):
                if self.tgt_rk_path_lst[i_tgt][i] in self.rk_cur_pos_lst:
                    num_of_obs_racks += 1
            num_new.append(num_of_obs_racks)

        if len(num_prev) == 0:
            self.num_of_obs_racks_on_path = copy.deepcopy(num_new)
            return False

        for i in range(len(num_prev)):
            if num_prev[i] < num_new[i]:
                self.num_of_obs_racks_on_path = copy.deepcopy(num_new)
                return True

        self.num_of_obs_racks_on_path = copy.deepcopy(num_new)
        return False

    def _search_target_paths(
        self, 
        rks: list[dict], 
        a_star_max_iter: int = -1
    ) -> tuple[bool, list]:
        """
        Run PP to find paths for target racks.
        """
        mov_obss = []
        mov_obss_edges = []
        solution_dict_all = [None for _ in range(self.tgt_rk_num)]

        print("tgt_rk_path_lst: --------------------------")
        for rk in rks:
            other_rk_starts_moc_obss = [
                (r["start"][0], r["start"][1], t)
                for t in [0, 1]
                for r in rks
                if r != rk
            ]
            other_rk_goals = [r["goal"] for r in rks if r != rk]
            env = AStarFollowingConflict(
                self.dimension,
                rk,
                set(self.static_obss + other_rk_goals),
                moving_obstacles=mov_obss + other_rk_starts_moc_obss,
                moving_obstacle_edges=mov_obss_edges,
                a_star_max_iter=a_star_max_iter,
                agent_start_pos_lst=self.rk_start_pos_lst,
                null_agent_pos_lst=self.null_ag_pos_lst,
                considering_cycle_conflict=False,
            )
            solution = env.compute_solution()

            if len(solution) == 0:
                print("No solution found at init")
                return False, rk

            mov_obss_tmp = []

            mov_obss_tmp += [(s["x"], s["y"], s["t"]) for s in solution]
            solution_dict_all[rk["name"]] = solution
            print(solution)

            for i in range(len(mov_obss_tmp) - 1):
                mov_obss_edges.append(
                    (
                        mov_obss_tmp[i][0],
                        mov_obss_tmp[i][1],
                        mov_obss_tmp[i + 1][0],
                        mov_obss_tmp[i + 1][1],
                    )
                )
            mov_obss += mov_obss_tmp

        return True, solution_dict_all

    def _init_phans_loop(
        self, 
        a_star_max_iter: int = -1
    ) -> None:
        """
        Initialization loop:
        - Relax conflicts and run target path search (PP).
        """
        # PP
        self._update_null_ag_pos_lst()
        rks = [
            {"start": self.rk_start_pos_lst[i], "goal": self.tgt_rk_goal_pos_lst[i], "name": i}
            for i in range(self.tgt_rk_num)
        ]

        # Process in descending order of distance from start to goal
        dst_from_rk_to_goal_lst = [
            abs(self.rk_start_pos_lst[i][0] - self.tgt_rk_goal_pos_lst[i][0])
            + abs(self.rk_start_pos_lst[i][1] - self.tgt_rk_goal_pos_lst[i][1])
            for i in range(self.tgt_rk_num)
        ]
        rks = [
            a
            for a, _ in sorted(
                zip(rks, dst_from_rk_to_goal_lst), key=lambda x: x[1], reverse=True
            )
        ]

        flag, solution_dict_all = self._search_target_paths(rks, a_star_max_iter)
        while not flag:
            # If pathfinding fails for an rack, prioritize it in the next attempt.
            failed_rk = solution_dict_all
            rks = [failed_rk] + [a for a in rks if a != failed_rk]
            flag, solution_dict_all = self._search_target_paths(rks, a_star_max_iter)

        self.tgt_rk_path_lst = []
        for i_rk in range(self.tgt_rk_num):
            path_tmp = []
            if solution_dict_all[i_rk] is not None:
                for t in range(len(solution_dict_all[i_rk])):
                    path_tmp.append(
                        (solution_dict_all[i_rk][t]["x"], solution_dict_all[i_rk][t]["y"])
                    )
            self.tgt_rk_path_lst.append(path_tmp)

        print("---------------------------------------------")

    def run_phans_loop(
        self,
        a_star_max_iter: int = 100000, 
        max_loop: int = 1000
    ) -> list[list[tuple[int, int]]]:
        """
        Move racks step-by-step following the target paths.

        This loop:
        - Advances target racks along their target paths.
        - Detects following conflicts (including edge conflicts) and vertex
          conflicts against racks that already reached their goals.
        - Identifies obstructing racks and moves null agents to vacate those cells.
        """
        # -----------------------------------
        # 1. Plan target paths
        # -----------------------------------
        self.rk_cur_pos_lst = copy.deepcopy(self.rk_start_pos_lst)
        self._init_phans_loop(a_star_max_iter)

        # -----------------------------------
        # 2. Evacuate obstructing racks and move target racks
        # -----------------------------------

        # Initialize
        if len(self.rk_start_pos_lst) < self.tgt_rk_num:
            self.max_count_evacuated_rk = 1
        else:
            self.max_count_evacuated_rk = len(self.ag_start_pos_lst) - self.tgt_rk_num
            
        all_path_lst = [copy.deepcopy(self.rk_cur_pos_lst)]
        self.tgt_cur_steps = [0] * len(self.tgt_rk_path_lst)

        loop = 0
        self._update_goal_rk()
        tgt_rks_cur_pos_lst = self.rk_cur_pos_lst[: self.tgt_rk_num]
        
        for i_tgt_rk in [i for i in range(self.tgt_rk_num) if i not in self.goal_rk_idx_lst]:
            self.tgt_cur_steps[i_tgt_rk] = (
                len(self.tgt_rk_path_lst[i_tgt_rk])
                - self.tgt_rk_path_lst[i_tgt_rk][::-1].index(tgt_rks_cur_pos_lst[i_tgt_rk])
                - 1
            )

        # Start loop
        while (
            any(
                self.tgt_cur_steps[i] < len(self.tgt_rk_path_lst[i]) - 1
                for i in range(self.tgt_rk_num)
            )
            and loop < max_loop
        ):
            # Identify obstructing racks
            
            # next blocking racks on target paths
            blocking_rk_pos_lst = [] 
            # their remaining distance along the blocked target’s path to its goal
            next_blocking_rk_dist_from_goal_lst = []

            for i_tgt_rk in range(self.tgt_rk_num):
                if self.tgt_cur_steps[i_tgt_rk] < len(self.tgt_rk_path_lst[i_tgt_rk]) - 1:
                    next_tgt_pos_lst = self.tgt_rk_path_lst[i_tgt_rk][
                        self.tgt_cur_steps[i_tgt_rk] + 1 :
                    ]

                    # Get the next blocking rack position
                    for pos in next_tgt_pos_lst:
                        if pos in self.rk_cur_pos_lst and pos not in tgt_rks_cur_pos_lst:
                            dst_to_goal = (
                                len(self.tgt_rk_path_lst[i_tgt_rk])
                                - self.tgt_rk_path_lst[i_tgt_rk].index(pos)
                                + 1
                            )
                            if pos not in blocking_rk_pos_lst:
                                blocking_rk_pos_lst.append(pos)
                                next_blocking_rk_dist_from_goal_lst.append(dst_to_goal)
                            else:
                                idx = blocking_rk_pos_lst.index(pos)
                                if next_blocking_rk_dist_from_goal_lst[idx] < dst_to_goal:
                                    # If multiple target racks share the same blocker,
                                    # increase its priority by updating with the larger distance.
                                    next_blocking_rk_dist_from_goal_lst[idx] = dst_to_goal

            # Order the obstructing racks
            blocking_rk_pos_lst = [
                pos
                for _, pos in sorted(
                    zip(
                        next_blocking_rk_dist_from_goal_lst,
                        blocking_rk_pos_lst,
                    ),
                    key=lambda x: x[0],
                    reverse=True,
                )
            ]

            # Define dependency among target racks:
            # If a target rack appears earlier on another's path, move the earlier one first.
            high_priority_blocking_rk_pos_lst = []
            dependency_relation = []  # (A, B): A depends on B

            for i_tgt_rk in [i for i in range(self.tgt_rk_num) if i not in self.goal_rk_idx_lst]:
                tgt_rk_next_pos = self.tgt_rk_path_lst[i_tgt_rk][self.tgt_cur_steps[i_tgt_rk] + 1]
                tgt_rk_cur_idx = self.tgt_cur_steps[i_tgt_rk]

                for j_tgt_rk in range(self.tgt_rk_num):
                    if i_tgt_rk != j_tgt_rk:
                        from_idx = self.tgt_cur_steps[j_tgt_rk]
                        to_idx = min(len(self.tgt_rk_path_lst[j_tgt_rk]) - 1, tgt_rk_cur_idx)
                        if tgt_rk_next_pos in self.tgt_rk_path_lst[j_tgt_rk][from_idx:to_idx]:
                            if j_tgt_rk in [x for x, _ in dependency_relation]:
                                dependency_relation.append((i_tgt_rk, j_tgt_rk))
                            else:
                                dependency_relation = [(i_tgt_rk, j_tgt_rk)] + dependency_relation

            for i_tgt_rk, j_tgt_rk in dependency_relation:
                tgt_rk_next_pos = self.tgt_rk_path_lst[i_tgt_rk][self.tgt_cur_steps[i_tgt_rk] + 1]
                from_idx = self.tgt_cur_steps[j_tgt_rk]
                high_priority_blocking_rk_pos_lst += self.tgt_rk_path_lst[j_tgt_rk][
                    from_idx : min(
                        len(self.tgt_rk_path_lst[j_tgt_rk]),
                        self.tgt_rk_path_lst[j_tgt_rk].index(tgt_rk_next_pos) + 2,
                    )
                ][::-1]

            for pos in high_priority_blocking_rk_pos_lst:
                if pos in blocking_rk_pos_lst:
                    blocking_rk_pos_lst.remove(pos)
                    blocking_rk_pos_lst = [pos] + blocking_rk_pos_lst

            # Choose the empty vertices
            self._update_null_ag_pos_lst()
            available_null_ags_pos_lst = copy.deepcopy(list(set(self.null_ag_pos_lst)))
            selected_null_ags_pos_lst = []

            # 何個の障害エージェントを退避させるか
            evacuated_rk_count = 0
            max_count_evacuated_rk = self.max_count_evacuated_rk
            for i_tgt_rk in range(self.tgt_rk_num):
                tgt_idx_on_tgt_rk_path = self.tgt_cur_steps[i_tgt_rk]
                if tgt_idx_on_tgt_rk_path < len(self.tgt_rk_path_lst[i_tgt_rk]) - 2:
                    tgt_next_pos = self.tgt_rk_path_lst[i_tgt_rk][tgt_idx_on_tgt_rk_path+1]
                    if tgt_next_pos not in blocking_rk_pos_lst:
                        max_count_evacuated_rk += 1

            mov_obss = []
            solution_dict_all = []
            null_ags_path_lst = []       
            for i, brk_pos in enumerate(blocking_rk_pos_lst):
                # Generally choose the null agent with the smallest h-value,
                # but avoid interfering with target racks' movements.

                dst_from_brk_to_goal = float("inf")
                for i_tgt_rk in range(self.tgt_rk_num):
                    if brk_pos in self.tgt_rk_path_lst[i_tgt_rk]:
                        dst_from_brk_to_goal = min(
                            dst_from_brk_to_goal,
                            len(self.tgt_rk_path_lst[i_tgt_rk])
                            - self.tgt_rk_path_lst[i_tgt_rk].index(brk_pos)
                            - 1,
                        )

                # Preserve vertices between the target and obstructing racks on target paths
                tgt_preserved_pos = []
                dst_penalty_dct = {}

                for i_tgt_rk in range(self.tgt_rk_num):
                    tgt_idx_on_tgt_rk_path = self.tgt_cur_steps[i_tgt_rk]
                    
                    if brk_pos in self.tgt_rk_path_lst[i_tgt_rk]:
                        brk_idx_on_tgt_rk_path = self.tgt_rk_path_lst[i_tgt_rk].index(brk_pos)

                        if tgt_idx_on_tgt_rk_path < brk_idx_on_tgt_rk_path:
                            for pos in self.tgt_rk_path_lst[i_tgt_rk][tgt_idx_on_tgt_rk_path:brk_idx_on_tgt_rk_path+1]:
                                tgt_preserved_pos.append(pos)

                    for i_pos, pos in enumerate(self.tgt_rk_path_lst[i_tgt_rk][tgt_idx_on_tgt_rk_path:]):
                        len_to_goal = len(self.tgt_rk_path_lst[i_tgt_rk][self.tgt_cur_steps[i_tgt_rk]:]) - i_pos - 1
                        if pos in dst_penalty_dct.keys():
                            dst_penalty_dct[pos] = max(dst_penalty_dct[pos], len_to_goal)
                        else:
                            dst_penalty_dct[pos] = len_to_goal

                while len([null_ag_pos for null_ag_pos in available_null_ags_pos_lst \
                    if null_ag_pos not in tgt_preserved_pos]) > 0 \
                    and evacuated_rk_count < self.max_count_evacuated_rk:
                    
                    tmp_available_null_ags_pos_lst = [
                        null_ag_pos
                        for null_ag_pos in available_null_ags_pos_lst
                        if null_ag_pos not in tgt_preserved_pos
                    ]

                    dst_penalty_array = np.array(
                        [dst_penalty_dct.get(pos, 0) for pos in tmp_available_null_ags_pos_lst]
                    )
                    distances = (
                        np.abs(np.array(tmp_available_null_ags_pos_lst) - np.array(brk_pos))
                        .sum(axis=1)
                    ) + dst_penalty_array
                    min_index = np.argmin(distances)
                    pos = tmp_available_null_ags_pos_lst[min_index]

                    # Plan paths for the null agent to move to the obstructing rack's positions.
                    obss = self.static_obss + list(
                        set(tgt_rks_cur_pos_lst)
                        - set(selected_null_ags_pos_lst)
                        - set(blocking_rk_pos_lst)
                    )

                    rk = {
                        "start": pos,
                        "goal": brk_pos,
                        "name": i,
                    }
                    
                    tgt_preserved_pos = []
                    for i_tgt_rk in range(self.tgt_rk_num):
                        if rk["goal"] in self.tgt_rk_path_lst[i_tgt_rk]:
                            tgt_preserved_pos += [
                                pos
                                for pos in self.tgt_rk_path_lst[i_tgt_rk][
                                    self.tgt_cur_steps[i_tgt_rk] : self.tgt_rk_path_lst[
                                        i_tgt_rk
                                    ].index(rk["goal"])
                                ]
                            ]
                        
                    env = AStarFollowingConflict(
                        self.dimension,
                        rk,
                        set(obss + tgt_preserved_pos),
                        moving_obstacles=mov_obss,
                        moving_obstacle_edges=[],
                        a_star_max_iter=10000,
                        is_dst_add=False,
                    )
                    solution = env.compute_solution()
                    if len(solution) == 0:
                        # fail
                        available_null_ags_pos_lst.remove(pos)
                        continue
                    else:
                        # success
                        selected_null_ags_pos_lst.append(pos)
                        available_null_ags_pos_lst.remove(pos)
                        evacuated_rk_count += 1
      
                        mov_obss += [(s["x"], s["y"], s["t"]) for s in solution]
                        solution_dict_all.append(solution)
                        break

            for solution_dict in solution_dict_all:
                null_ags_path_lst.append([(s['x'], s['y']) for s in solution_dict])

            # Execute movements
            rk_next_pos_lst = copy.deepcopy(self.rk_cur_pos_lst)
            rk_pre_pos_lst = []

            one_tgt_rk_goal_flag = True
            while one_tgt_rk_goal_flag and loop < max_loop:
                # Move target racks if possible
                for i_tgt_rk in [
                    i for i in range(self.tgt_rk_num) if i not in self.goal_rk_idx_lst
                ]:
                    tgt_rk_next_pos = self.tgt_rk_path_lst[i_tgt_rk][
                        self.tgt_cur_steps[i_tgt_rk] + 1
                    ]
                    if (
                        tgt_rk_next_pos not in rk_next_pos_lst
                        and tgt_rk_next_pos not in rk_pre_pos_lst
                    ):
                        rk_pre_pos = copy.deepcopy(rk_next_pos_lst[i_tgt_rk])
                        rk_pre_pos_lst.append(rk_pre_pos)
                        rk_next_pos_lst[i_tgt_rk] = copy.deepcopy(tgt_rk_next_pos)
                        one_tgt_rk_goal_flag = False

                        self.rk_transport_task_lst.append([rk_pre_pos, tgt_rk_next_pos])

                # Move null agents by one step if possible
                for i_null_ag in range(len(null_ags_path_lst)):
                    if len(null_ags_path_lst[i_null_ag]) > 1:
                        # null_ag_pos_idx_lst_on_null_ag_path: 
                        #   indices where the null agent can be on its path.
                        #   Typically 0, but due to Manhattan distance selection,
                        #   it may differ (if it wasn't the nearest one).
                        null_ag_pos_idx_lst_on_null_ag_path = [
                            i
                            for i, pos in enumerate(null_ags_path_lst[i_null_ag])
                            if pos not in self.rk_cur_pos_lst[self.tgt_rk_num:]
                            and pos not in rk_next_pos_lst[self.tgt_rk_num:]
                        ]

                        if len(null_ag_pos_idx_lst_on_null_ag_path) > 0:
                            null_ag_cur_pos_idx = max(null_ag_pos_idx_lst_on_null_ag_path)
                            null_ag_cur_pos = null_ags_path_lst[i_null_ag][null_ag_cur_pos_idx]

                            if (
                                null_ag_cur_pos not in rk_next_pos_lst
                                and null_ag_cur_pos_idx < len(null_ags_path_lst[i_null_ag]) - 1
                            ):
                                null_ag_next_pos = null_ags_path_lst[i_null_ag][
                                    null_ag_cur_pos_idx + 1
                                ]

                                # Move a null agent by swapping with the adjacent rack:
                                #   |■|□|  -->  |□|■|
                                if (
                                    null_ag_next_pos in self.rk_cur_pos_lst
                                    and null_ag_next_pos in rk_next_pos_lst
                                ):
                                    rk_next_pos_lst[
                                        rk_next_pos_lst.index(null_ag_next_pos)
                                    ] = copy.deepcopy(null_ag_cur_pos)
                                    null_ags_path_lst[i_null_ag] = null_ags_path_lst[i_null_ag][1:]
                                    self.rk_transport_task_lst.append([null_ag_next_pos, null_ag_cur_pos])

                print(rk_next_pos_lst[: self.tgt_rk_num])
                self.tgt_rk_path_lst_plot.append(copy.deepcopy(self.tgt_rk_path_lst))
                all_path_lst.append(copy.deepcopy(rk_next_pos_lst))
                self.rk_cur_pos_lst = copy.deepcopy(rk_next_pos_lst)

                loop += 1

            self._update_goal_rk()
            tgt_rks_cur_pos_lst = self.rk_cur_pos_lst[: self.tgt_rk_num]
            for i_tgt_rk in [i for i in range(self.tgt_rk_num) if i not in self.goal_rk_idx_lst]:
                self.tgt_cur_steps[i_tgt_rk] = (
                    len(self.tgt_rk_path_lst[i_tgt_rk])
                    - self.tgt_rk_path_lst[i_tgt_rk][::-1].index(
                        tgt_rks_cur_pos_lst[i_tgt_rk]
                    )
                    - 1
                )
            self._update_goal_rk()

        return all_path_lst, self.rk_transport_task_lst

    def _assign_tasks(
        self,
        dependency_relation: Dict[int, List[int]],
        ag_class: List[Dict],
        unassigned_task_idx_lst: List[int],
        completed_task_idx_lst: List[int],
        task_assignment_fixed: bool = True,
    ):
        """
        Assign tasks to agents using the Hungarian algorithm.
        """
        # ---------------------------
        # Assign tasks to agents
        # ---------------------------

        # Enumerate agents without an assigned task
        unassigned_ag_idx_lst = [
            i_ag for i_ag in range(len(ag_class)) if ag_class[i_ag]["assigned_task_idx"] is None
        ]

        unassigned_ag_pos_dct: Dict[Tuple[int, int], int] = {}
        for i_ag in unassigned_ag_idx_lst:
            unassigned_ag_pos_dct[ag_class[i_ag]["cur_pos"]] = i_ag

        unassigned_task_pos_dct: Dict[Tuple[int, int], int] = {}
        for i_task in unassigned_task_idx_lst:
            if all(dep in completed_task_idx_lst for dep in dependency_relation[i_task]):
                unassigned_task_pos_dct[self.rk_transport_task_lst[i_task][0]] = i_task

        if len(unassigned_ag_pos_dct) == 0 or len(unassigned_task_pos_dct) == 0:
            # No available agent or task to assign
            return ag_class, unassigned_task_idx_lst

        ag_pos_keys = list(unassigned_ag_pos_dct.keys())
        ag_indices = list(unassigned_ag_pos_dct.values())
        task_pos_keys = list(unassigned_task_pos_dct.keys())

        ags = np.array(ag_pos_keys, dtype=np.float32)
        tasks = np.array(task_pos_keys, dtype=np.float32)
        cost_matrix = np.sum(np.abs(ags[:, np.newaxis, :] - tasks[np.newaxis, :, :]), axis=2)

        # Consider loading/unloading costs
        cost_matrix += 1
        for i_ag in ag_indices:
            i_ag_in_matrix = ag_indices.index(i_ag)
            if ag_class[i_ag]["status"] == "unloading":
                cost_matrix[i_ag_in_matrix] += 1  # unloading cost
                cur_pos = ag_class[i_ag]["cur_pos"]
                if cur_pos in task_pos_keys:
                    i_task_in_matrix = task_pos_keys.index(cur_pos)
                    # If moving the very rack just handled, waive the extra cost
                    cost_matrix[i_ag_in_matrix, i_task_in_matrix] -= 2

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i_ag_in_matrix, i_task_in_matrix in zip(row_ind, col_ind):
            i_ag = unassigned_ag_pos_dct[tuple(ags[i_ag_in_matrix])]
            i_task = unassigned_task_pos_dct[tuple(tasks[i_task_in_matrix])]
            task_pickup_pos = self.rk_transport_task_lst[i_task][0]
            task_delivery_pos = self.rk_transport_task_lst[i_task][1]

            if task_assignment_fixed:
                unassigned_task_idx_lst.remove(i_task)

            ag_class[i_ag]["assigned_task_idx"] = i_task
            ag_class[i_ag]["assigned_task_pickup_pos"] = task_pickup_pos
            ag_class[i_ag]["assigned_task_delivery_pos"] = task_delivery_pos

            if (
                ag_class[i_ag]["cur_pos"] == ag_class[i_ag]["assigned_task_pickup_pos"]
                and all(
                    dep in completed_task_idx_lst
                    for dep in dependency_relation[ag_class[i_ag]["assigned_task_idx"]]
                )
            ):
                if ag_class[i_ag]["status"] == "unloading":
                    ag_class[i_ag]["status"] = "delivering"
                else:
                    ag_class[i_ag]["status"] = "loading"
            elif ag_class[i_ag]["status"] == "unloading":
                pass
            else:
                ag_class[i_ag]["status"] = "picking"

        return ag_class, unassigned_task_idx_lst

    def search_ag_path_assign_fixed_ivf_loading_cost(self):
        """
        Assign tasks once (fixed), then plan agent paths considering loading/unloading priority.
        """
        dependency_relation: Dict[int, List[int]] = {
            i_task: [] for i_task in range(len(self.rk_transport_task_lst))
        }

        for i_task in range(len(self.rk_transport_task_lst)):
            pickup_i, delivery_i = self.rk_transport_task_lst[i_task]

            for j_task in range(i_task):
                pickup_j, delivery_j = self.rk_transport_task_lst[j_task]

                # Dependency 1: Own pickup equals other's pickup/delivery (rack not at pickup yet)
                if pickup_i == pickup_j or pickup_i == delivery_j:
                    dependency_relation[i_task].append(j_task)

                # Dependency 2: Own delivery equals other's pickup/delivery
                if delivery_i == pickup_j or delivery_i == delivery_j:
                    dependency_relation[i_task].append(j_task)

        completed_task_idx_lst: List[int] = []

        ag_class: List[Dict] = []
        for i_ag in range(len(self.ag_start_pos_lst)):
            ag_class.append(
                {
                    "cur_pos": self.ag_start_pos_lst[i_ag],
                    "assigned_task_idx": None,
                    "assigned_task_pickup_pos": None,
                    "assigned_task_delivery_pos": None,
                    "status": "idle",  # picking, delivering, idle, unloading, loading
                    "pass": [],
                }
            )

        all_ag_path_lst = [
            [
                [self.ag_start_pos_lst[i_ag][0], self.ag_start_pos_lst[i_ag][1], 0]
                for i_ag in range(len(self.ag_start_pos_lst))
            ]
        ]
        all_racks_path_lst = [self.rk_start_pos_lst]

        unassigned_task_idx_lst = [i for i in range(len(self.rk_transport_task_lst))]
        t = 0
        while len(unassigned_task_idx_lst) > 0 or not all(
            [ag["status"] == "idle" for ag in ag_class]
        ):
            if (time.time() - self.time_count_start) > self.max_calc_time:
                self.timed_out = True
                break
            ag_class, unassigned_task_idx_lst = self._assign_tasks(
                dependency_relation,
                ag_class,
                unassigned_task_idx_lst,
                completed_task_idx_lst,
                task_assignment_fixed=True,
            )

            # ---------------------------
            # Plan with PIBT
            # ---------------------------
            grid = np.full(self.dimension, True)
            for obs in self.static_obss:
                grid[obs[0], obs[1]] = False
            ags_start = [ag["cur_pos"] for ag in ag_class]
            priority = [
                abs(ag["assigned_task_pickup_pos"][0] - ag["cur_pos"][0])
                + abs(ag["assigned_task_pickup_pos"][1] - ag["cur_pos"][1])
                if ag["assigned_task_idx"] is not None
                else 0
                for ag in ag_class
            ]
            ags_goal = []
            for i_ag, ag in enumerate(ag_class):
                if ag["status"] == "picking":
                    ags_goal.append(ag["assigned_task_pickup_pos"])
                elif ag["status"] == "loading":
                    # loading: highest priority
                    priority[i_ag] = self.size_x * self.size_y * 2
                    ags_goal.append(ag["assigned_task_pickup_pos"])
                elif ag["status"] == "delivering":
                    # delivering: high priority
                    priority[i_ag] = self.size_x * self.size_y
                    ags_goal.append(ag["assigned_task_delivery_pos"])
                elif ag["status"] == "unloading":
                    # unloading: highest priority
                    priority[i_ag] = self.size_x * self.size_y * 2
                    ags_goal.append(ag["cur_pos"])
                else:
                    ags_goal.append(ag["cur_pos"])

            pibt_calc_time = self.max_calc_time - (time.time() - self.time_count_start)
            if pibt_calc_time < 0:
                self.timed_out = True
                break
            pibt = PIBT(grid, ags_start, ags_goal, max_calc_time=pibt_calc_time)
            configs, _ = pibt.run_all_steps(priority=priority, max_timestep=2)

            if configs is None:
                print("Timed out during PIBT search.")
                return None, None
            if len(configs) == 1:
                next_pos_lst = configs[0]
            else:
                next_pos_lst = configs[1]

            # ---------------------------
            # Update states
            # ---------------------------
            ag_next_pos_lst = []
            rack_next_pos_lst = copy.deepcopy(all_racks_path_lst[-1])
            just_completed_task_idx_lst: List[int] = []
            ag_idx_lst = sorted(
                [i_ag for i_ag in range(len(ag_class))],
                key=lambda i_ag: ag_class[i_ag]["assigned_task_idx"]
                if ag_class[i_ag]["assigned_task_idx"] is not None
                else 999,
            )
            for i_ag in ag_idx_lst:
                ag_class[i_ag]["cur_pos"] = next_pos_lst[i_ag]
                ag_class[i_ag]["pass"].append(next_pos_lst[i_ag])

                if ag_class[i_ag]["status"] == "unloading":
                    if ag_class[i_ag]["assigned_task_idx"] is not None:
                        ag_class[i_ag]["status"] = "picking"
                    else:
                        ag_class[i_ag]["status"] = "idle"
                    ag_next_pos_lst.append(
                        [next_pos_lst[i_ag][0], next_pos_lst[i_ag][1], 0]
                    )
                elif ag_class[i_ag]["status"] == "picking":
                    i_task = ag_class[i_ag]["assigned_task_idx"]

                    if (
                        ag_class[i_ag]["cur_pos"]
                        == ag_class[i_ag]["assigned_task_pickup_pos"]
                    ):
                        if all(
                            dep in just_completed_task_idx_lst + completed_task_idx_lst
                            for dep in dependency_relation[i_task]
                        ):
                            # Pickup complete and dependency cleared -> start delivery
                            ag_class[i_ag]["status"] = "loading"
                            ag_next_pos_lst.append(
                                [next_pos_lst[i_ag][0], next_pos_lst[i_ag][1], 1]
                            )
                        else:
                            # Arrived at pickup but dependency not cleared -> release task
                            unassigned_task_idx_lst.append(
                                ag_class[i_ag]["assigned_task_idx"]
                            )
                            ag_class[i_ag]["assigned_task_idx"] = None
                            ag_class[i_ag]["assigned_task_pickup_pos"] = None
                            ag_class[i_ag]["assigned_task_delivery_pos"] = None
                            ag_class[i_ag]["status"] = "idle"
                    else:
                        ag_next_pos_lst.append(
                            [next_pos_lst[i_ag][0], next_pos_lst[i_ag][1], 0]
                        )

                elif ag_class[i_ag]["status"] == "loading":
                    i_task = ag_class[i_ag]["assigned_task_idx"]
                    if (
                        ag_class[i_ag]["cur_pos"]
                        == ag_class[i_ag]["assigned_task_pickup_pos"]
                        and all(
                            dep in just_completed_task_idx_lst + completed_task_idx_lst
                            for dep in dependency_relation[i_task]
                        )
                    ):
                        # Pickup complete and dependency cleared -> start delivery
                        ag_class[i_ag]["status"] = "delivering"

                    ag_next_pos_lst.append(
                        [next_pos_lst[i_ag][0], next_pos_lst[i_ag][1], 1]
                    )
                elif ag_class[i_ag]["status"] == "delivering":
                    if (
                        next_pos_lst[i_ag]
                        == ag_class[i_ag]["assigned_task_delivery_pos"]
                    ):
                        # (2) Delivery succeeded
                        i_rack = rack_next_pos_lst.index(
                            ag_class[i_ag]["assigned_task_pickup_pos"]
                        )
                        rack_next_pos_lst[i_rack] = next_pos_lst[i_ag]

                        ag_next_pos_lst.append(
                            [next_pos_lst[i_ag][0], next_pos_lst[i_ag][1], 1]
                        )
                        just_completed_task_idx_lst.append(
                            ag_class[i_ag]["assigned_task_idx"]
                        )

                        # Release task on completion
                        ag_class[i_ag]["status"] = "unloading"
                        ag_class[i_ag]["assigned_task_idx"] = None
                        ag_class[i_ag]["assigned_task_pickup_pos"] = None
                        ag_class[i_ag]["assigned_task_delivery_pos"] = None
                    elif next_pos_lst[i_ag] == ag_class[i_ag]["assigned_task_pickup_pos"]:
                        # (3) Delivery failed (blocked) -> carry rack forward anyway
                        i_rack = rack_next_pos_lst.index(
                            ag_class[i_ag]["assigned_task_pickup_pos"]
                        )
                        rack_next_pos_lst[i_rack] = next_pos_lst[i_ag]
                        ag_next_pos_lst.append(
                            [next_pos_lst[i_ag][0], next_pos_lst[i_ag][1], 1]
                        )
                    else:
                        # Moved to an unrelated cell due to PIBT detour -> release task
                        unassigned_task_idx_lst.append(
                            ag_class[i_ag]["assigned_task_idx"]
                        )
                        ag_class[i_ag]["assigned_task_idx"] = None
                        ag_class[i_ag]["assigned_task_pickup_pos"] = None
                        ag_class[i_ag]["assigned_task_delivery_pos"] = None
                        ag_class[i_ag]["status"] = "idle"
                else:
                    ag_next_pos_lst.append(
                        [next_pos_lst[i_ag][0], next_pos_lst[i_ag][1], 0]
                    )

            all_ag_path_lst.append(
                copy.deepcopy(
                    sorted(
                        ag_next_pos_lst,
                        key=lambda x: ag_idx_lst[ag_next_pos_lst.index(x)],
                    )
                )
            )
            all_racks_path_lst.append(copy.deepcopy(rack_next_pos_lst))
            completed_task_idx_lst += just_completed_task_idx_lst
            t += 1

        if self.timed_out:
            print("Timed out during path search.")
            return None, None
        else:
            return all_ag_path_lst[:-1], all_racks_path_lst[:-1]

    def search_ag_path_reassign_each_step_loading_cost(self):
        """
        Reassign tasks at each step, planning agent paths with loading/unloading priority.
        """
        dependency_relation: Dict[int, List[int]] = {
            i_task: [] for i_task in range(len(self.rk_transport_task_lst))
        }

        for i_task in range(len(self.rk_transport_task_lst)):
            pickup_i, delivery_i = self.rk_transport_task_lst[i_task]

            for j_task in range(i_task):
                pickup_j, delivery_j = self.rk_transport_task_lst[j_task]

                # Dependency 1: Own pickup equals other's pickup/delivery
                if pickup_i == pickup_j or pickup_i == delivery_j:
                    dependency_relation[i_task].append(j_task)

                # Dependency 2: Own delivery equals other's pickup/delivery
                if delivery_i == pickup_j or delivery_i == delivery_j:
                    dependency_relation[i_task].append(j_task)

        completed_task_idx_lst: List[int] = []

        ag_class: List[Dict] = []
        for i_ag in range(len(self.ag_start_pos_lst)):
            ag_class.append(
                {
                    "cur_pos": self.ag_start_pos_lst[i_ag],
                    "assigned_task_idx": None,
                    "assigned_task_pickup_pos": None,
                    "assigned_task_delivery_pos": None,
                    "pre_assigned_task_idx": None,
                    "status": "idle",  # picking, loading, delivering, unloading, idle
                    "pass": [],
                }
            )

        all_ag_path_lst = [
            [
                [self.ag_start_pos_lst[i_ag][0], self.ag_start_pos_lst[i_ag][1], 0]
                for i_ag in range(len(self.ag_start_pos_lst))
            ]
        ]
        all_racks_path_lst = [self.rk_start_pos_lst]

        unassigned_task_idx_lst = [i for i in range(len(self.rk_transport_task_lst))]
        t = 0
        while len(unassigned_task_idx_lst) > 0 or not all(
            [ag["status"] == "idle" for ag in ag_class]
        ):
            if (time.time() - self.time_count_start) > self.max_calc_time:
                self.timed_out = True
                break
            ag_class, unassigned_task_idx_lst = self._assign_tasks(
                dependency_relation,
                ag_class,
                unassigned_task_idx_lst,
                completed_task_idx_lst,
                task_assignment_fixed=False,
            )

            # ---------------------------
            # Plan with PIBT
            # ---------------------------
            grid = np.full(self.dimension, True)
            for obs in self.static_obss:
                grid[obs[0], obs[1]] = False
            ags_start = [ag["cur_pos"] for ag in ag_class]
            priority = [
                (len(self.rk_transport_task_lst) - ag["assigned_task_idx"]) \
                    / len(self.rk_transport_task_lst)
                if ag["assigned_task_idx"] is not None
                else -1
                for ag in ag_class
            ]
            ags_goal = []
            for i_ag, ag in enumerate(ag_class):
                if ag["status"] == "picking":
                    ags_goal.append(ag["assigned_task_pickup_pos"])
                elif ag["status"] == "loading":
                    # loading: highest priority
                    priority[i_ag] += self.size_x * self.size_y * 2
                    ags_goal.append(ag["assigned_task_pickup_pos"])
                elif ag["status"] == "delivering":
                    # delivering: high priority
                    priority[i_ag] += self.size_x * self.size_y
                    ags_goal.append(ag["assigned_task_delivery_pos"])
                elif ag["status"] == "unloading":
                    # unloading: highest priority
                    priority[i_ag] += self.size_x * self.size_y * 2
                    ags_goal.append(ag["cur_pos"])
                else:
                    ags_goal.append(ag["cur_pos"])

            pibt_calc_time = self.max_calc_time - (time.time() - self.time_count_start)
            if pibt_calc_time < 0:
                self.timed_out = True
                break
            pibt = PIBT(grid, ags_start, ags_goal, max_calc_time=pibt_calc_time)
            configs, goal_reached_steps = pibt.run_all_steps(priority=priority, max_timestep=2)
            if configs is None:
                print("Timed out during PIBT search.")
                self.timed_out = True
                break
            if len(configs) == 1:
                next_pos_lst = configs[0]
            else:
                next_pos_lst = configs[1]

            # ---------------------------
            # Update states
            # ---------------------------
            ag_next_pos_lst = []
            rack_next_pos_lst = copy.deepcopy(all_racks_path_lst[-1])
            just_completed_task_idx_lst: List[int] = []
            ag_idx_lst = sorted(
                [i_ag for i_ag in range(len(ag_class))],
                key=lambda i_ag: ag_class[i_ag]["assigned_task_idx"]
                if ag_class[i_ag]["assigned_task_idx"] is not None
                else 999,
            )
            for i_ag in ag_idx_lst:
                ag_class[i_ag]["cur_pos"] = next_pos_lst[i_ag]
                ag_class[i_ag]["pass"].append(next_pos_lst[i_ag])

                if (
                    ag_class[i_ag]["status"] == "loading"
                    and ag_class[i_ag]["cur_pos"]
                    == ag_class[i_ag]["assigned_task_pickup_pos"]
                ):
                    # loading finished --> delivering
                    ag_next_pos_lst.append(
                        [next_pos_lst[i_ag][0], next_pos_lst[i_ag][1], 1]
                    )
                    ag_class[i_ag]["pre_assigned_task_idx"] = ag_class[i_ag][
                        "assigned_task_idx"
                    ]
                    ag_class[i_ag]["status"] = "delivering"
                    unassigned_task_idx_lst.remove(ag_class[i_ag]["assigned_task_idx"])
                elif ag_class[i_ag]["status"] == "delivering":
                    ag_next_pos_lst.append(
                        [next_pos_lst[i_ag][0], next_pos_lst[i_ag][1], 1]
                    )
                    if (
                        ag_class[i_ag]["cur_pos"]
                        == ag_class[i_ag]["assigned_task_delivery_pos"]
                        and all(
                            dep in completed_task_idx_lst
                            for dep in dependency_relation[ag_class[i_ag]["assigned_task_idx"]]
                        )
                    ):
                        # delivering finished --> unloading
                        i_rack = rack_next_pos_lst.index(
                            ag_class[i_ag]["assigned_task_pickup_pos"]
                        )
                        rack_next_pos_lst[i_rack] = next_pos_lst[i_ag]
                        just_completed_task_idx_lst.append(
                            ag_class[i_ag]["assigned_task_idx"]
                        )
                        if ag_class[i_ag]["assigned_task_idx"] in unassigned_task_idx_lst:
                            # If task became delivering without unloading phase, remove here
                            unassigned_task_idx_lst.remove(ag_class[i_ag]["assigned_task_idx"])
                        ag_class[i_ag]["pre_assigned_task_idx"] = ag_class[i_ag][
                            "assigned_task_idx"
                        ]
                        ag_class[i_ag]["status"] = "unloading"
                        ag_class[i_ag]["assigned_task_idx"] = None
                        ag_class[i_ag]["assigned_task_pickup_pos"] = None
                        ag_class[i_ag]["assigned_task_delivery_pos"] = None
                    else:
                        # Assigned delivery failed (still in delivery)
                        if ag_class[i_ag]["assigned_task_idx"] in unassigned_task_idx_lst:
                            # If delivering without unloading, remove from unassigned queue here
                            unassigned_task_idx_lst.remove(ag_class[i_ag]["assigned_task_idx"])
                else:
                    # unloading --> idle, idle --> idle, and other failures
                    ag_next_pos_lst.append(
                        [next_pos_lst[i_ag][0], next_pos_lst[i_ag][1], 0]
                    )
                    ag_class[i_ag]["pre_assigned_task_idx"] = ag_class[i_ag][
                        "assigned_task_idx"
                    ]
                    ag_class[i_ag]["status"] = "idle"
                    ag_class[i_ag]["assigned_task_idx"] = None
                    ag_class[i_ag]["assigned_task_pickup_pos"] = None
                    ag_class[i_ag]["assigned_task_delivery_pos"] = None

            all_ag_path_lst.append(
                copy.deepcopy(
                    sorted(
                        ag_next_pos_lst,
                        key=lambda x: ag_idx_lst[ag_next_pos_lst.index(x)],
                    )
                )
            )
            all_racks_path_lst.append(copy.deepcopy(rack_next_pos_lst))
            completed_task_idx_lst += just_completed_task_idx_lst
            t += 1

        if self.timed_out:
            print("Timed out during path search.")
            return None, None
        else:
            return all_ag_path_lst[:-1], all_racks_path_lst[:-1]

    def run_loop(
        self,
        ag_start_pos_lst: List[Tuple[int, int]],
        rk_start_pos_lst: List[Tuple[int, int]],
        tgt_rk_goal_pos_lst: List[Tuple[int, int]],
        task_assignment_fixed: bool = True,
    ):
        """
        Main entry:
        1) Optimize racks' trajectories; 2) Assign tasks and optimize agents' paths.
        """
        self.ag_start_pos_lst = ag_start_pos_lst
        self.tgt_rk_goal_pos_lst = tgt_rk_goal_pos_lst
        self.tgt_rk_num = len(tgt_rk_goal_pos_lst)
        self.rk_start_pos_lst = rk_start_pos_lst

        # -----------------------------------
        # 1. Optimize racks' trajectories
        # -----------------------------------
        self.run_phans_loop()

        # -----------------------------------
        # 2. Assign tasks & optimize agents
        # -----------------------------------
        if task_assignment_fixed:
            return self.search_ag_path_assign_fixed_ivf_loading_cost()
        else:
            return self.search_ag_path_reassign_each_step_loading_cost()

    def plot_animation(
        self,
        animation_name,
        ag_routes,
        rack_routes,
        slow_factor: int = 2,
    ):
        """
        Animate agent and rack movement.
        """
        # --------------------------
        # Draw animation
        # --------------------------
        import matplotlib.animation as anm

        fig, ax = plt.subplots(figsize=(7, 3.2))

        ax.set_xlim(-0.6, self.size_x - 1 + 0.5)
        ax.set_ylim(-0.55, self.size_y - 1 + 0.55)
        ax.set(xticks=[], yticks=[])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        def update(frame):
            ax.cla()

            tgt_color = ["red", "blue", "green", "steelblue", "deeppink", "brown", "orange", 
                         "purple", "olive", "cyan", "lime", "magenta", "yellow", "sandybrown",
                         "lightcoral", "darkorange", "gold", "orchid", "lightseagreen", 
                         "slateblue", "darkkhaki", "plum", "lightgray", "lightblue", "lightgreen",
                         "lightsalmon", "lightpink", "lightyellow", "lightcyan", "lavender", 
                         "thistle","peachpuff", "powderblue", "rosybrown", "skyblue", "tan", 
                         "wheat", "khaki"]

            # --------------------------
            # Agent
            # --------------------------
            t = int(frame / slow_factor)
            cur_pos_lst = [ag_routes[t][i][:2] for i in range(len(ag_routes[t]))]
            if t < len(ag_routes) - 1:
                next_pos_lst = [ag_routes[t + 1][i][:2] for i in range(len(ag_routes[t + 1]))]
            else:
                next_pos_lst = cur_pos_lst

            # Interpolate between current and next positions for smooth animation
            w = (frame % slow_factor) / slow_factor
            mid_pos_lst = [
                (
                    (1 - w) * cur_pos_lst[i][0] + w * next_pos_lst[i][0],
                    (1 - w) * cur_pos_lst[i][1] + w * next_pos_lst[i][1],
                )
                for i in range(len(cur_pos_lst))
            ]

            # Labels (legend stubs)
            ax.scatter(
                -10, 
                -10, 
                s=30, 
                color="white", 
                edgecolor="black", 
                label="ag (free)", 
                marker="o", 
                alpha=1.0
            )
            ax.scatter(
                -10,
                -10,
                s=30,
                color="black",
                edgecolor="black",
                label="ag (conveying)",
                marker="o",
                alpha=1.0,
            )

            for i in range(len(ag_routes[t])):
                if ag_routes[t][i][2] == 0:
                    ax.scatter(
                        mid_pos_lst[i][0] - 0.05,
                        mid_pos_lst[i][1],
                        s=30,
                        color="white",
                        edgecolor="black",
                        marker="o",
                        alpha=1.0,
                    )
                else:
                    ax.scatter(
                        mid_pos_lst[i][0] - 0.05,
                        mid_pos_lst[i][1],
                        s=30,
                        color="black",
                        edgecolor="black",
                        marker="o",
                        alpha=1.0,
                    )

            # --------------------------
            # Rack
            # --------------------------
            cur_pos_lst = [rack_routes[t][i][:2] for i in range(len(rack_routes[t]))]
            if t < len(rack_routes) - 1:
                next_pos_lst = [
                    rack_routes[t + 1][i][:2] for i in range(len(rack_routes[t + 1]))
                ]
            else:
                next_pos_lst = cur_pos_lst

            mid_pos_lst = [
                (
                    (1 - w) * cur_pos_lst[i][0] + w * next_pos_lst[i][0],
                    (1 - w) * cur_pos_lst[i][1] + w * next_pos_lst[i][1],
                )
                for i in range(len(cur_pos_lst))
            ]

            for i in range(len(rack_routes[t])):
                if i < self.tgt_rk_num:
                    ax.scatter(
                        mid_pos_lst[i][0] - 0.05,
                        mid_pos_lst[i][1],
                        s=50,
                        color=tgt_color[i],
                        label="target rack",
                        marker="s",
                        alpha=0.5,
                    )
                elif i == self.tgt_rk_num + 1:
                    ax.scatter(
                        mid_pos_lst[i][0] - 0.05,
                        mid_pos_lst[i][1],
                        s=50,
                        color="grey",
                        label="obstructing rack",
                        marker="s",
                        alpha=0.5,
                    )
                else:
                    ax.scatter(
                        mid_pos_lst[i][0] - 0.05,
                        mid_pos_lst[i][1],
                        s=50,
                        color="grey",
                        marker="s",
                        alpha=0.5,
                    )

            for i in range(len(self.static_obss)):
                ax.scatter(
                    self.static_obss[i][0], self.static_obss[i][1], s=50, color="black", marker="s"
                )

            for i_target, tgt_path in enumerate(
                self.tgt_rk_path_lst_plot[min(t, len(self.tgt_rk_path_lst_plot) - 1)]
            ):
                for i in range(len(tgt_path) - 1):
                    ax.plot(
                        [tgt_path[i][0], tgt_path[i + 1][0]],
                        [tgt_path[i][1], tgt_path[i + 1][1]],
                        color=tgt_color[i_target],
                        alpha=0.5,
                    )

            # --------------------------
            # Axes and decorations
            # --------------------------
            ax.set_xlim(-0.6, self.size_x - 1 + 0.5)
            ax.set_xticks([i for i in range(self.size_x)])
            ax.tick_params(axis="x", labelsize=plt.rcParams["font.size"] / 2)
            ax.set_ylim(-0.55, self.size_y - 1 + 0.55)
            ax.set_yticks([i for i in range(self.size_y)])
            ax.tick_params(axis="y", labelsize=plt.rcParams["font.size"] / 2)

            ax.set_title(f"t: {t}")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, labelspacing=0.2)

            return ax.plot()

        plt.subplots_adjust(left=0.005, right=0.7, bottom=0.01, top=0.92)
        ani = anm.FuncAnimation(
            fig, update, interval=300, frames=len(ag_routes) * slow_factor, repeat=False
        )
        plt.close()
        ani.save(animation_name, fps=8, dpi=300)
