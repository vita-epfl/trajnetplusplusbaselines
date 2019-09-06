import argparse
from trajnettools.reader import Reader
from trajnettools import show


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files', nargs='+',
                        help='Trajnet dataset file(s).')
    parser.add_argument('--n', type=int, default=15,
                        help='sample n trajectories')
    parser.add_argument('--id', type=int, nargs='*',
                        help='plot a particular scene')
    parser.add_argument('-o', '--output', default=None,
                        help='specify output prefix')
    parser.add_argument('--random', default=True, action='store_true',
                        help='randomize scenes')
    args = parser.parse_args()

    ## TODO Configure Writing images
    # if args.output is None:
    #     args.output = args.dataset_file

    ## Read GT Scenes
    reader = Reader(args.dataset_files[0], scene_type='paths')
    if args.id:
        scenes = reader.scenes(ids=args.id, randomize=args.random)
    elif args.n:
        scenes = reader.scenes(limit=args.n, randomize=args.random)
    else:
        scenes = reader.scenes(randomize=args.random)

    ## Reader Predictions 
    reader_list = {}
    for dataset_file in args.dataset_files[1:]:
        name = dataset_file.split('/')[-2]
        reader_list[name] = Reader(dataset_file, scene_type='paths')

    ## Visualize
    pred_paths = {}
    for scene_id, paths in scenes:
        for dataset_file in args.dataset_files[1:]:
            name = dataset_file.split('/')[-2]
            scenes_pred = reader_list[name].scenes(ids=[scene_id])
            for scene_id, preds in scenes_pred:
                primary = preds[0]
                primary_path = [t for t in primary if t.scene_id == scene_id]
            pred_paths[name] = primary_path
        ##TODO
        # output = '{}.scene{}.png'.format(args.output, scene_id)
        with show.predicted_paths(paths, pred_paths):
            pass

if __name__ == '__main__':
    main()
