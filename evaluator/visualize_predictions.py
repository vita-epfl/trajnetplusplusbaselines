import argparse
from trajnetplusplustools.reader import Reader
from trajnetplusplustools import show


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
    parser.add_argument('--labels', required=False, nargs='+',
                        help='labels of models')
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
    label_dict = {}
    for i, dataset_file in enumerate(args.dataset_files[1:]):
        name = dataset_file.split('/')[-2]
        label_dict[name] = args.labels[i] if args.labels else name
        reader_list[name] = Reader(dataset_file, scene_type='paths')

    ## Visualize
    pred_paths = {}
    pred_neigh_paths = {}
    for scene_id, paths in scenes:
        print("Scene ID: ", scene_id)
        for dataset_file in args.dataset_files[1:]:
            name = dataset_file.split('/')[-2]
            scenes_pred = reader_list[name].scenes(ids=[scene_id])
            for scene_id, preds in scenes_pred:
                predicted_paths = [[t for t in pred if t.scene_id == scene_id] for pred in preds]
            pred_paths[label_dict[name]] = predicted_paths[0]
            pred_neigh_paths[label_dict[name]] = predicted_paths[1:]

        output_filename = None
        if args.output is not None:
            output_filename = '{}.scene{}.png'.format(args.output, scene_id)
        with show.predicted_paths(paths, pred_paths, output_file=output_filename):
            pass
        # with show.predicted_paths(paths, pred_paths, pred_neigh_paths):
        #     pass

if __name__ == '__main__':
    main()
