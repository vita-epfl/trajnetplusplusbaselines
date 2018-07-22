import argparse
import trajnettools.show

from . import kalman
from . import lstm


def predict(ground_truth, output_file):
    with trajnettools.show.paths(ground_truth, output_file) as ax:
        # KF prediction
        kf = kalman.predict(ground_truth)
        ax.plot([ground_truth[0][8].x] + [r.x for r in kf],
                [ground_truth[0][8].y] + [r.y for r in kf],
                color='orange', label='KF', marker='o', markersize=2.5)
        ax.plot([kf[-1].x], [kf[-1].y], color='orange', marker='o', linestyle='None')

        # LSTM prediction
        lstm_predictor = lstm.LSTMPredictor.load('output/vanilla_lstm.pkl')
        lstmp = lstm_predictor(ground_truth)
        ax.plot([ground_truth[0][8].x] + [r.x for r in lstmp],
                [ground_truth[0][8].y] + [r.y for r in lstmp],
                color='blue', label='LSTM', marker='o', markersize=2.5)
        ax.plot([lstmp[-1].x], [lstmp[-1].y], color='blue', marker='o', linestyle='None')

        # OLSTM prediction
        olstm_predictor = lstm.LSTMPredictor.load('output/occupancy_lstm.pkl')
        olstmp = olstm_predictor(ground_truth)
        ax.plot([ground_truth[0][8].x] + [r.x for r in olstmp],
                [ground_truth[0][8].y] + [r.y for r in olstmp],
                color='green', label='O-LSTM', marker='o', markersize=2.5)
        ax.plot([olstmp[-1].x], [olstmp[-1].y], color='green', marker='o', linestyle='None')

        # SLSTM prediction
        slstm_predictor = lstm.LSTMPredictor.load('output/social_lstm.pkl')
        slstmp = slstm_predictor(ground_truth)
        ax.plot([ground_truth[0][8].x] + [r.x for r in slstmp],
                [ground_truth[0][8].y] + [r.y for r in slstmp],
                color='red', label='Social LSTM', marker='o', markersize=2.5)
        ax.plot([slstmp[-1].x], [slstmp[-1].y], color='red', marker='o', linestyle='None')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file')
    parser.add_argument('-o', '--output',
                        help='output file prefix')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.dataset_file

    reader = trajnettools.Reader(args.dataset_file, scene_type='paths')
    for scene_id, ground_truth in reader.scenes(limit=5):
        output_file = '{output_prefix}.prediction{scene}.png'.format(
            output_prefix=args.output, scene=scene_id)
        predict(ground_truth, output_file)


if __name__ == '__main__':
    main()
