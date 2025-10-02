import random

from solver.multi_agent_and_multi_rack_path_finding import PathFindingDenseEnv

def main():                
    random.seed(0)

    # --------------------------
    # problem definition
    # --------------------------
    size_x = 7
    size_y = 5

    static_obss = []

    tgt_rk_goal = [(0, 2), (6, 2)]       

    ag_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss))
    ag_start_pos_lst = random.sample(ag_option_nodes, int(len(tgt_rk_goal)*1.5))

    tgt_rk_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss) - set(tgt_rk_goal))
    tgt_rk_start_pos_lst = random.sample(tgt_rk_option_nodes, len(tgt_rk_goal))

    other_rk_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss+tgt_rk_start_pos_lst))
    other_rk_start_pos_lst = random.sample(other_rk_option_nodes, int(size_x*size_y*0.8)-len(static_obss))

    rk_start_pos_lst = tgt_rk_start_pos_lst + other_rk_start_pos_lst

    # --------------------------
    # Run
    # --------------------------
    problem = PathFindingDenseEnv(size_x, size_y, static_obstacles=static_obss)
    all_agv_path_lst, all_racks_path_lst = problem.run_loop(ag_start_pos_lst, rk_start_pos_lst, tgt_rk_goal, task_assignment_fixed=False)
    print(all_agv_path_lst)

    # --------------------------
    # Save animation
    # --------------------------
    animation_name = f'{size_x}x{size_y}_{len(static_obss)}obs_{len(ag_start_pos_lst)}ags_{len(rk_start_pos_lst)}rks_{len(tgt_rk_goal)}tgts_{len(all_agv_path_lst)}steps.gif'
    problem.plot_animation(animation_name, all_agv_path_lst, all_racks_path_lst)

if __name__ == '__main__':
    main()
