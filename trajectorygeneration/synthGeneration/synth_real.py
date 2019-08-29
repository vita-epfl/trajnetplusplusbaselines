import argparse
import rvo2
import socialforce  
import trajnettools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file',
                        help='trajnet dataset file')
    parser.add_argument('dest_file',
                        help='destination file')
    parser.add_argument('--n', type=int, default=3,
                        help='sample n trajectories')
    parser.add_argument('--id', type=int, nargs='*',
                        help='plot a particular scene')
    parser.add_argument('-o', '--output', default=None,
                        help='specify output prefix')
    parser.add_argument('--random', default=False, action='store_true',
                        help='randomize scenes')
    args = parser.parse_args()

    ## Load Scene
    reader = trajnettools.Reader(args.dataset_file, scene_type='rows')
    if args.id:
        scenes = reader.scenes(ids=args.id, randomize=args.random)
    elif args.n:
        scenes = reader.scenes(limit=args.n, randomize=args.random)
    else:
        scenes = reader.scenes(randomize=args.random)
       
    ## Load pd of scene
    destf = pd.read_csv(args.dest_file)
    # print(destf.head())

    for scene_id, scene_ped, rows in scenes:
        start_frame = rows[0].frame
        first_half = start_frame < 100000
        print("Start frame: ", start_frame)
        paths = reader.track_rows_to_paths(scene_ped, rows)
        xy = np.array(reader.paths_to_xy(paths))

        ## Ped_set
        print(scene_ped)
        # print(rows)
        ped_set = [scene_ped] + [row.pedestrian for row in rows if row.frame == start_frame if row.pedestrian != scene_ped]
        print(ped_set)
        ## Corresponding Path Set
        start_pos = np.invert(np.isnan(xy[0,:,0]))
        ped_paths = xy[:,start_pos,:]     
        assert len(ped_set) == ped_paths.shape[1], "Shortlisting Error"

        # trajectories, positions, goals, speed = initialize(ped_set, ped_paths, destf, sim=None)
        trajectories, goals = generate_sf_trajectory(ped_set, ped_paths, destf)
        simulated = np.array(trajectories)
        # simulated = np.array(trajectories).transpose(1,0,2)
        print("Original: ", np.array(ped_paths).shape)
        print("Sim: ", np.array(simulated).shape)

        # print("Sim: ", np.array(trajectories).transpose(1,0,2).shape)
        plt.plot(simulated[:, 0, 0], simulated[:, 0, 1], label='Sim')
        plt.plot(ped_paths[:, 0, 0], ped_paths[:, 0, 1], label='GT')
        for i in range(1, ped_paths.shape[1]):
            plt.plot(ped_paths[:, i, 0], ped_paths[:, i, 1], label='Neigh')            
        plt.plot(goals[0][0], goals[0][1], 'bo', label='Goal')
        # print(np.array(goals).shape)
        plt.legend()
        plt.show()
        plt.close()

        # print("Sim: ", np.array(trajectories).transpose(1,0,2).shape)
        plt.plot(simulated[:, 0, 0], simulated[:, 0, 1], label='Sim')
        plt.plot(ped_paths[:, 0, 0], ped_paths[:, 0, 1], label='GT')
        for i in range(1, simulated.shape[1]):
            plt.plot(simulated[:, i, 0], simulated[:, i, 1], label='Neigh')            
        plt.plot(goals[0][0], goals[0][1], 'bo', label='Goal')
        # print(np.array(goals).shape)
        plt.legend()
        plt.show()
        plt.close()

        print("ADE: ", np.linalg.norm(simulated[:, 0] - ped_paths[:, 0])/21)
        print("FDE: ", np.linalg.norm(simulated[-1, 0] - ped_paths[-1, 0]))

def initialize(ped_set, ped_paths, destf, sim=None):

    # initialize agents' starting and goal positions
    positions = []
    goals = []
    speed = []
    for i in range(len(ped_set)):
        path = ped_paths[:,i]
        px = path[0,0]
        py = path[0,1]

        gx = destf.loc[destf["ped"] == ped_set[i]].iloc[0]["x"]/1000
        gy = destf.loc[destf["ped"] == ped_set[i]].iloc[0]["y"]/1000
        # print("Gx: ", gx)
        # print("Gy: ", gy)
        positions.append((px, py))
        goals.append((gx, gy))

        if sim is not None:
            sim.addAgent((px, py))
            
        time_instant = 1
        row1, row2 = path[0], path[time_instant]
        if not np.isnan(row2[0]):
            diff = np.array([row2[0] - row1[0], row2[1] - row1[1]])
            theta = np.arctan2(diff[1], diff[0])
            # print(theta*180/np.pi)
            vr = np.linalg.norm(diff) / (time_instant * 0.4)
            # print(vr*np.sin(theta))
            speed.append((vr*np.cos(theta), vr*np.sin(theta)))
        else:
            speed.append((0,0))

    trajectories = [[positions[i]] for i in range(len(ped_set))]
    return trajectories, positions, goals, speed


def generate_sf_trajectory(ped_set, ped_paths, destf, end_range=1):

    num_ped = len(ped_set)
    ##Initiliaze a scene
    trajectories, positions, goals, speed = initialize(ped_set, ped_paths, destf)

    initial_state = np.array([[positions[i][0], positions[i][1], speed[i][0], speed[i][1],
                               goals[i][0], goals[i][1]] for i in range(num_ped)])
    s = socialforce.Simulator(initial_state, tau=5)
    states = np.stack([s.step().state.copy() for _ in range(20)])
    trajectories = np.concatenate((initial_state[np.newaxis,:,:2], states[:, :, 0:2]))
    return trajectories, goals

    # reaching_goal = [False] * num_ped
    # done = False
    # count = 2
    # delta_t = 0.2
    # fps = 0.4
    # sample = int(fps/delta_t)
    # ##Simulate a scene
    # while not done and count < 21*sample:
    #     count += 1
    #     s = socialforce.Simulator(initial_state, delta_t=delta_t)
    #     position = np.stack(s.step().state.copy())
    #     for i in range(len(initial_state)):
    #         if count % sample == 0:
    #             trajectories[i].append((position[i, 0], position[i, 1]))
    #         # check if this agent reaches the goal
    #         if np.linalg.norm(position[i, :2] - np.array(goals[i])) < end_range:
    #             if i == 0:
    #                 print(position[i, :2])
    #                 print("Reached Goal")
    #             reaching_goal[i] = True
    #         else:
    #             initial_state[i, :4] = position[i, :4]

    #     done = all(reaching_goal)

        # control_trajectories, control_goals = generate_sf_control()
        # print(np.array(control_trajectories).shape)
        # control_s = np.array(control_trajectories).transpose(1,0,2)
        # # print("Original: ", np.array(ped_paths).shape)
        # # print("Sim: ", np.array(trajectories).transpose(1,0,2).shape)
        # print(control_s[:, 0, 1])
        # print(control_s[:, 1, 1])
        # plt.plot(control_s[:, 0, 0], control_s[:, 0, 1], label='Sim1')
        # # plt.plot(ped_paths[:, 0, 0], ped_paths[:, 0, 1], label='GT')
        # plt.plot(control_goals[0][0], control_goals[0][1], 'bo', label='Goal1')
        # plt.plot(-control_s[:, 1, 0], control_s[:, 1, 1], label='Sim2')
        # # plt.plot(ped_paths[:, 0, 0], ped_paths[:, 0, 1], label='GT')
        # plt.plot(control_goals[1][0], control_goals[1][1], 'ro', label='Goal2')
        # # print(np.array(goals).shape)
        # plt.legend()
        # plt.xlim(-10, 10)
        # plt.show()
        # plt.close()

# def generate_sf_control(end_range=1):
#     positions = []
#     goals = []
#     speed = []

#     gg = 10
#     positions.append((0,0))
#     positions.append((0,gg))

#     goals.append((0,gg))
#     goals.append((0,0))

#     speed.append((0,  1))
#     speed.append((0, -1))

#     num_ped = len(positions)
#     print("Num Ped: ", num_ped)
#     trajectories = [[positions[i]] for i in range(num_ped)]

#     ##Initiliaze a scene
#     # trajectories, positions, goals, speed = initialize(ped_set, ped_paths, destf)

#     initial_state = np.array([[positions[i][0], positions[i][1], speed[i][0], speed[i][1],
#                                goals[i][0], goals[i][1]] for i in range(num_ped)])
#     print(initial_state)
#     reaching_goal = [False] * num_ped
#     done = False
#     count = 2
#     delta_t = 0.2
#     fps = 0.4
#     sample = int(fps/delta_t)
#     ##Simulate a scene
#     while not done and count < 1000*sample:
#         count += 1
#         sc = socialforce.Simulator(initial_state, delta_t=delta_t, tau=1.0)
#         position = np.stack(sc.step().state.copy())
#         for i in range(len(initial_state)):
#             if count % sample == 0:
#                 trajectories[i].append((position[i, 0], position[i, 1]))
#             # check if this agent reaches the goal
#             if np.linalg.norm(position[i, :2] - np.array(goals[i])) < end_range:
#                 if i == 1:
#                     print(position[i, :2])
#                     print("Reached Goal")
#                 reaching_goal[i] = True
#             else:
#                 initial_state[i, :4] = position[i, :4]

#         done = all(reaching_goal)

#     return trajectories, goals


if __name__ == '__main__':
    main()

# def scenario2_initialize(num_ped, sim=None):
#     # initialize agents' starting and goal positions
#     positions = []
#     goals = []
#     speed = []
#     for i in range(num_ped):
#         norm = 0
#         while norm < 12:
#             px = (np.random.random() - 0.5) * SQUARE_LENGTH
#             py = (np.random.random() - 0.5) * SQUARE_LENGTH
#             gx = (np.random.random() - 0.5) * SQUARE_LENGTH
#             gy = (np.random.random() - 0.5) * SQUARE_LENGTH
#             norm = np.linalg.norm([py - px, gy - gx])
#         positions.append((px, py))
#         goals.append((gx, gy))
#         if sim is not None:
#             sim.addAgent((px, py))

#         rand_speed = random.uniform(0.8, 1.2)
#         perc_x = abs(gx - px) / (abs(gx - px) + abs(gy - py))
#         vx = perc_x * rand_speed * np.sign(gx - px)
#         vy = (1 - perc_x) * rand_speed * np.sign(gy - py)
#         speed.append((vx, vy))

#     trajectories = [[positions[i]] for i in range(num_ped)]
#     return trajectories, positions, goals, speed

# def generate_orca_trajectory(ped_set, ped_paths, destf, end_range=1):
#     sim = rvo2.PyRVOSimulator(1 / 2.5, 3, 10, 1.5, 2, 0.4, 2)

#     ##Initiliaze a scene
#     trajectories, positions, goals, speed = initialize(scenario, num_ped, sim=sim)
#     done = False
#     reaching_goal_by_ped = [False] * num_ped
#     count = 0

#     ##Simulate a scene
#     while not done and count < 21:
#         sim.doStep()
#         reaching_goal = []
#         for i in range(num_ped):
#             if count == 0:
#                 trajectories[i].pop(0)
#             position = sim.getAgentPosition(i)
#             trajectories[i].append(position)

#             # check if this agent reaches the goal
#             if np.linalg.norm(np.array(position) - np.array(goals[i])) < end_range:
#                 reaching_goal.append(True)
#                 sim.setAgentPrefVelocity(i, (0, 0))
#                 reaching_goal_by_ped[i] = True
#             else:
#                 reaching_goal.append(False)
#                 velocity = np.array((goals[i][0] - position[0], goals[i][1] - position[1]))
#                 speed = np.linalg.norm(velocity)
#                 pref_vel = velocity / speed if speed > 1 else velocity
#                 sim.setAgentPrefVelocity(i, tuple(pref_vel.tolist()))
#         count += 1
#         done = all(reaching_goal)

#     return trajectories