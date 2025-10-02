import random

from solver.multi_agent_and_multi_rack_path_finding import PathFindingDenseEnv

def main():                
    random.seed(0)

    # --------------------------
    # problem definition
    # --------------------------
    size_x = 35
    size_y = 21

    static_obss = [(0, 9), (1, 9), (0, 10), (1, 10), (0, 11), (1, 11),
                   (4,  3), (5,  3), (6,  3), (4,  4), (5,  4), (6,  4), 
                   (4,  5), (5,  5), (6,  5), 
                   (4,  15), (5,  15), (6,  15), (4,  16), (5,  16), (6,  16), 
                   (4,  17), (5,  17), (6,  17), 
                   (9, 9), (10, 9), (11, 9), (12, 9), (9, 10), (10, 10), 
                   (11, 10), (12, 10), (9, 11), (10, 11), (11, 11), (12, 11), 
                   (16, 3), (17, 3), (18, 3), (16, 4), (17, 4), (18, 4), 
                   (16, 5), (17, 5), (18, 5), 
                   (16, 15), (17, 15), (18, 15), (16, 16), (17, 16), (18, 16), 
                   (16, 17), (17, 17), (18, 17),
                   (22, 9), (23, 9), (24, 9), (25, 9), (22, 10), (23, 10), 
                   (24, 10), (25, 10), (22, 11), (23, 11), (24, 11), (25, 11),
                   (28, 3), (29, 3), (30, 3), (28, 4), (29, 4), (30, 4), 
                   (28, 5), (29, 5), (30, 5), 
                   (28, 15), (29, 15), (30, 15), (28, 16), (29, 16), (30, 16), 
                   (28, 17), (29, 17), (30, 17),
                   (33, 9), (34, 9), (33, 10), (34, 10), (33, 11), (34, 11)]

    tgt_rk_goal = [(0, 0), 
                   (size_x-1, 0),
                   (size_x-1, size_y-1), 
                   (0, size_y-1),
                   
                   (int(size_x/3), 0), 
                   (size_x-1, int(size_y/3)),
                   (int(size_x*2/3), size_y-1),
                   (0, int(size_y*2/3)-1),
                   
                   (int(size_x*2/3), 0),
                   (size_x-1, int(size_y*2/3)-1),
                   (int(size_x*1/3), size_y-1), 
                   (0, int(size_y/3))]       

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
