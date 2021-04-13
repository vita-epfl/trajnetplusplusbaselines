import random
import argparse
import os
import shutil

random.seed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='trajdata',
                        help='glob expression for data files')
    parser.add_argument('--val_ratio', default=0.2, type=float,
                        help='sample ratio of val set given the train set')
    args = parser.parse_args()

    args.path = 'DATA_BLOCK/' + args.path
    
    ## Prepare destination folder containing dataset with train and val split
    args.dest_path = args.path + '_split'

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path)

    if not os.path.exists('{}/train/'.format(args.dest_path)):
        os.makedirs('{}/train/'.format(args.dest_path))

    if not os.path.exists('{}/val/'.format(args.dest_path)):
        os.makedirs('{}/val/'.format(args.dest_path))

    ## List train file names
    files = [f.split('.')[-2] for f in os.listdir(args.path + '/train/') if f.endswith('.ndjson')]
    print(files)
    # exit()

    for file in files:
        ## Read All Samples
        orig_train_file = args.path + '/train/' + file + '.ndjson'
        with open(orig_train_file, "r") as f:
            lines = f.readlines()

        ## Split scenes into train and val
        train_file = open(args.dest_path + '/train/' + file + ".ndjson", "w")
        val_file = open(args.dest_path + '/val/' + file + ".ndjson", "w")
        for line in lines:
            ## Sample Scenes
            if 'scene' in line:
                if random.random() < args.val_ratio:
                    val_file.write(line)
                else:
                    train_file.write(line)
                continue
            ## Write All tracks
            train_file.write(line)
            val_file.write(line)

        train_file.close()
        val_file.close()

    # ## Assert val folder does not exist
    # if os.path.isdir(args.path + '/val'):
    #     print("Validation folder already exists")
    #     exit()

if __name__ == '__main__':
    main()

#     ## Iterate over file names
#     for file in files:
#         with open("DATA_BLOCK/honda/test/honda_v1.ndjson", "r") as f
#         reader = trajnetplusplustools.Reader(path + '/train/' + file + '.ndjson', scene_type='paths')
#         ## Necessary modification of train scene to add filename
#         scene = [(file, s_id, s) for s_id, s in reader.scenes()]
#         all_scenes += scene

# with open("test_dummy1.ndjson", "w") as f:
#     for line in lines:
#         if 'scene' in line and random.random() < 0.8:
#             continue
#         f.write(line)


# ## read goal files
# all_goals = {}
# all_scenes = []

# ## List file names
# files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
# ## Iterate over file names
# for file in files:
#     reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
#     ## Necessary modification of train scene to add filename
#     scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
#     if goals:
#         goal_dict = pickle.load(open('goal_files/' + subset + file +'.pkl', "rb"))
#         ## Get goals corresponding to train scene
#         all_goals[file] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scene}
#     all_scenes += scene



