import time

import numpy as np

from .dist_table import DistTable
from .mapf_utils import Config, Coord, get_neighbors


class PIBT:
    """Priority Inheritance with Backtracking (PIBT) solver."""

    def __init__(self, grid, agents_start, agents_goal, seed=0, max_calc_time=120):
        self.grid = grid
        self.starts = Config(agents_start)
        self.goals = Config(agents_goal)
        self.N = len(self.starts)
        self.priority = []

        # Distance tables to each agent's goal
        self.dist_tables = [DistTable(grid, goal) for goal in self.goals]

        # Cache
        self.NIL = self.N  # sentinel for "bottom"
        self.NIL_COORD: Coord = self.grid.shape  # sentinel for "bottom"
        self.occupied_now = np.full(grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(grid.shape, self.NIL, dtype=int)
        self.requesting_now = np.full(grid.shape, self.NIL, dtype=int)

        # RNG for tie-breaking
        self.rng = np.random.default_rng(seed)

        # Max calculation time control
        self.time_count_start = time.time()
        self.max_calc_time = max_calc_time
        self.timed_out = False

    def funcPIBT(self, i, parent_idx=None) -> bool:
        """PIBT recursion for agent i.
        Returns True if a valid move/request was set for agent i, otherwise False.
        """
        # Timeout guard
        if self.timed_out:
            return False
        if time.time() - self.time_count_start > self.max_calc_time:
            self.timed_out = True
            return False

        # ------------------------------------------------------------
        # If the agent is "requesting", check whether it can move now
        # ------------------------------------------------------------
        if self.agents[i]["mode"] == "requesting":
            tails_except_i = [self.agents[j]["tail"] for j in range(self.N) if j != i]
            if self.agents[i]["head"] not in tails_except_i:
                # Request accepted -> move to extended
                self.agents[i]["mode"] = "extended"
                self.agents[i]["parent"] = None
                self.unsearched_agents_idx_lst.remove(i)
                return True
            # Keep requesting
            self.unsearched_agents_idx_lst.remove(i)
            return True

        # ----------------------------------------------------------------
        # If the agent is contracted, choose next vertex to request or move
        # ----------------------------------------------------------------
        # Candidate next vertices (stay or move to neighbors)
        candidates = [self.agents[i]["tail"]] + get_neighbors(self.grid, self.agents[i]["tail"])
        self.rng.shuffle(candidates)  # tie-breaking by randomization
        candidates = sorted(candidates, key=lambda u: self.dist_tables[i].get(u))

        # Vertex assignment
        for v in candidates:
            heads_except_i = [self.agents[j]["head"] for j in range(self.N) if j != i]
            tails_except_i = [self.agents[j]["tail"] for j in range(self.N) if j != i]

            # v is not in other agents' heads or tails
            if v not in heads_except_i + tails_except_i:
                self.agents[i]["head"] = v
                self.agents[i]["mode"] = "extended"
                self.unsearched_agents_idx_lst.remove(i)
                return True

            # v is in other agents' heads -> cannot use; skip
            elif v in heads_except_i:
                continue

            # v is in other agents' tails
            else:
                obs_idx = [j for j in range(self.N) if self.agents[j]["tail"] == v][0]
                obstacle_agent = self.agents[obs_idx]

                # If that blocking agent has not been explored yet
                if obstacle_agent["head"] is None:
                    # Request the blocking agent to move elsewhere
                    self.agents[i]["head"] = v
                    self.agents[i]["mode"] = "requesting"
                    self.agents[i]["parent"] = parent_idx

                    if self.funcPIBT(obs_idx, parent_idx):
                        self.unsearched_agents_idx_lst.remove(i)
                        return True
                    else:
                        # Backtrack
                        self.agents[i]["head"] = None
                        self.agents[i]["mode"] = "contracted"
                        self.agents[i]["parent"] = None
                        continue
                # If the blocking agent is already explored
                else:
                    continue

        return False

    def step(self):
        """Perform one PIBT step for all agents based on current priorities."""
        # Prepare index list of agents sorted by priority (higher first)
        self.unsearched_agents_idx_lst = list(range(self.N))
        self.unsearched_agents_idx_lst = sorted(
            self.unsearched_agents_idx_lst, key=lambda k: self.priority[k], reverse=True
        )

        # Resolve requests/moves for all agents in priority order
        while self.unsearched_agents_idx_lst:
            if self.timed_out:
                # If already timed out during the loop, stop immediately
                return
            i = self.unsearched_agents_idx_lst[0]
            self.funcPIBT(i)

        # Commit moves: extended -> contracted (head becomes new tail)
        for a_name in self.agents:
            if self.agents[a_name]["mode"] == "extended":
                self.agents[a_name]["tail"] = self.agents[a_name]["head"]
                self.agents[a_name]["head"] = None
                self.agents[a_name]["mode"] = "contracted"
                self.agents[a_name]["parent"] = None
            else:
                pass

        # Update configuration history
        self.configs.append([self.agents[i]["tail"] for i in range(self.N)])

    def run(self, max_timestep: int = 1000):
        """Run exactly one step (after initializing agents/priority) and return the configs."""
        self.agents = {}
        self.configs = [self.starts]
        for i, pos in enumerate(self.starts):
            self.agents[i] = {}
            self.agents[i]["tail"] = pos
            self.agents[i]["head"] = None
            self.agents[i]["mode"] = "contracted"  # contracted, requesting, extended
            self.agents[i]["parent"] = None

        # Priority init (kept as-is to preserve original behavior)
        self.priority = []
        for i in range(self.N):
            self.priority.append(self.dist_tables[i].get(self.agents[i]["tail"]) / self.grid.size)

        # Priority update (overwrites the above; kept to maintain original logic)
        for i in range(self.N):
            self.priority[i] = self.dist_tables[i].get(self.agents[i]["tail"])

        # Obtain new configuration
        self.step()

        return self.configs

    def run_all_steps(self, priority: list = [], max_timestep: int = 1000):
        """Run multiple steps until all agents reach goals or max_timestep is hit.

        Note: The default mutable argument is intentionally kept to preserve the original signature
        and behavior. Callers may pass an explicit list to control initial priorities.
        """
        self.agents = {}
        self.configs = [self.starts]
        for i, pos in enumerate(self.starts):
            self.agents[i] = {}
            self.agents[i]["tail"] = pos
            self.agents[i]["head"] = None
            self.agents[i]["mode"] = "contracted"  # contracted, requesting, extended
            self.agents[i]["parent"] = None

        # Main loop: generate a sequence of configurations
        goal_reached_steps = [None for _ in range(self.N)]
        self.priority = priority

        while len(self.configs) <= max_timestep:
            if self.timed_out:
                # Stop immediately if timed out during the loop
                return None, None

            # Define/update priorities
            num_goal_reached = 0
            for i in range(self.N):
                if self.agents[i]["tail"] == self.goals[i]:
                    if self.priority[i] < self.grid.shape[0] * self.grid.shape[1]:
                        self.priority[i] = 0
                    num_goal_reached += 1
                    if goal_reached_steps[i] is None:
                        goal_reached_steps[i] = len(self.configs)
                else:
                    pass

            if num_goal_reached == self.N:
                break

            # Obtain new configuration
            self.step()

        return self.configs, goal_reached_steps
