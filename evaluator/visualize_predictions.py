from pathlib import Path
import argparse
from trajnetplusplustools.reader import Reader
from trajnetplusplustools import show


def add_gt_observation_to_prediction(gt_observation, model_prediction):
    obs_length = len(gt_observation[0]) - len(model_prediction[0])
    full_predicted_paths = [gt_observation[ped_id][:obs_length] + pred for ped_id, pred in enumerate(model_prediction)]
    return full_predicted_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files', nargs='+',
                        help='Provide the ground-truth file followed by model prediction file(s)')
    parser.add_argument('--n', type=int, default=15,
                        help='sample n trajectories')
    parser.add_argument('--id', type=int, nargs='*',
                        help='plot a particular scene')
    parser.add_argument('--viz_folder', default='./visualizations',
                        help='base folder to store visualizations')
    parser.add_argument('-o', '--output', default=None,
                        help='specify output prefix')
    parser.add_argument('--random', default=True, action='store_true',
                        help='randomize scenes')
    parser.add_argument('--labels', required=False, nargs='+',
                        help='labels of models')
    args = parser.parse_args()

    # Determine and construct appropriate folders to save visualization
    dataset_name = args.dataset_files[0].split('/')[1]
    model_name = args.dataset_files[1].split('/')[-2]
    folder_name = f"{args.viz_folder}/{dataset_name}/{model_name}"
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    single_model = len(args.dataset_files) == 2

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

        # Visualize prediction(s) overlayed on GT scene
        output_filename = f"{folder_name}/single_scene{scene_id}.png" if single_model else \
                            f"{folder_name}/multiple_scene{scene_id}.png"
        with show.predicted_paths(paths, pred_paths, output_file=output_filename):
            pass

        # Used when visualizing only a single model
        if single_model:
            # Visualize GT scene
            gt_filename = f"{folder_name}/gt_scene{scene_id}.png"
            with show.paths(paths, output_file=gt_filename):
                pass
            # Visualize Model Prediction scene
            pred_filename = f"{folder_name}/pred_scene{scene_id}.png"
            full_predicted_paths = add_gt_observation_to_prediction(paths, predicted_paths)
            with show.paths(full_predicted_paths, output_file=pred_filename):
                pass

if __name__ == '__main__':
    main()
