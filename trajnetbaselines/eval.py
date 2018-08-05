"""Command line tool to create a table of evaluations metrics."""

import pickle

import trajnettools
import trajnetbaselines


class Evaluator(object):
    def __init__(self, scenes, nonlinear_scene_index):
        self.scenes = scenes
        self.nonlinear_scene_index = nonlinear_scene_index

        self.average_l2 = {'N': len(scenes)}
        self.average_l2_nonlinear = {'N': len(nonlinear_scene_index)}
        self.final_l2 = {'N': len(scenes)}

    def aggregate(self, name, predictor):
        print('evaluating', name)

        average = 0.0
        nonlinear = 0.0
        final = 0.0

        for scene_i, paths in enumerate(self.scenes):
            prediction = predictor(paths)
            average_l2 = trajnettools.metrics.average_l2(paths[0], prediction)
            final_l2 = trajnettools.metrics.final_l2(paths[0], prediction)

            # aggregate
            average += average_l2
            final += final_l2
            if scene_i in self.nonlinear_scene_index:
                nonlinear += average_l2

        average /= len(self.scenes)
        nonlinear /= len(self.nonlinear_scene_index)
        final /= len(self.scenes)

        self.average_l2[name] = average
        self.average_l2_nonlinear[name] = nonlinear
        self.final_l2[name] = final

        return self

    def result(self):
        return self.average_l2, self.average_l2_nonlinear, self.final_l2


def eval(input_file):
    print('dataset', input_file)

    sample = 0.05 if 'syi.ndjson' in input_file else None
    reader = trajnettools.Reader(input_file, scene_type='paths')
    scenes = [s for _, s in reader.scenes(sample=sample)]

    # non-linear scenes from high Kalman Average L2
    nonlinear_score = []
    for paths in scenes:
        kalman_prediction = trajnetbaselines.kalman.predict(paths)
        nonlinear_score.append(trajnettools.metrics.average_l2(paths[0], kalman_prediction))
    mean_nonlinear = sum(nonlinear_score) / len(scenes)
    nonlinear_scene_index = {i for i, nl in enumerate(nonlinear_score) if nl > mean_nonlinear}

    evaluator = Evaluator(scenes, nonlinear_scene_index)

    # Kalman Filter (Lin) and non-linear scenes
    evaluator.aggregate('kf', trajnetbaselines.kalman.predict)

    # LSTM
    lstm_predictor = trajnetbaselines.lstm.LSTMPredictor.load('output/vanilla_lstm.pkl')
    evaluator.aggregate('lstm', lstm_predictor)

    # OLSTM
    olstm_predictor = trajnetbaselines.lstm.LSTMPredictor.load('output/occupancy_lstm.pkl')
    evaluator.aggregate('olstm', olstm_predictor)

    # DLSTM
    dlstm_predictor = trajnetbaselines.lstm.LSTMPredictor.load('output/directional_lstm.pkl')
    evaluator.aggregate('dlstm', dlstm_predictor)

    # Social LSTM
    slstm_predictor = trajnetbaselines.lstm.LSTMPredictor.load('output/social_lstm.pkl')
    evaluator.aggregate('slstm', slstm_predictor)

    return evaluator.result()


def main():
    datasets = [
        # 'data/val/biwi_eth.ndjson',
        'data/val/biwi_hotel.ndjson',
        # 'data/val/crowds_zara01.ndjson',
        'data/val/crowds_zara02.ndjson',
        # 'data/val/crowds_uni_examples.ndjson',

        'data/val/crowds_zara03.ndjson',
        'data/val/crowds_students001.ndjson',
        'data/val/crowds_students003.ndjson',
        # 'data/val/mot_pets2009_s2l1.ndjson',

        'data/val/dukemtmc.ndjson',
        # 'data/val/syi.ndjson',
        # 'data/val/wildtrack.ndjson',
    ]
    results = {dataset
               .replace('data/', '')
               .replace('.ndjson', ''): eval(dataset)
               for dataset in datasets}
    with open('output/eval.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('## Average L2 [m]')
    print('{dataset:>30s} |   N  |  Lin | LSTM | O-LSTM | D-LSTM | S-LSTM'.format(dataset=''))
    for dataset, (r, _, _) in results.items():
        print(
            '{dataset:>30s}'
            ' | {r[N]:>4}'
            ' | {r[kf]:.2f}'
            ' | {r[lstm]:.2f}'
            ' |  {r[olstm]:.2f} '
            ' |  {r[dlstm]:.2f} '
            ' |  {r[slstm]:.2f}'.format(dataset=dataset, r=r)
        )

    print('')
    print('## Average L2 (non-linear sequences) [m]')
    print('{dataset:>30s} |   N  |  Lin | LSTM | O-LSTM | D-LSTM | S-LSTM'.format(dataset=''))
    for dataset, (_, r, _) in results.items():
        print(
            '{dataset:>30s}'
            ' | {r[N]:>4}'
            ' | {r[kf]:.2f}'
            ' | {r[lstm]:.2f}'
            ' |  {r[olstm]:.2f} '
            ' |  {r[dlstm]:.2f} '
            ' |  {r[slstm]:.2f}'.format(dataset=dataset, r=r)
        )

    print('')
    print('## Final L2 [m]')
    print('{dataset:>30s} |   N  |  Lin | LSTM | O-LSTM | D-LSTM | S-LSTM'.format(dataset=''))
    for dataset, (_, _, r) in results.items():
        print(
            '{dataset:>30s}'
            ' | {r[N]:>4}'
            ' | {r[kf]:.2f}'
            ' | {r[lstm]:.2f}'
            ' |  {r[olstm]:.2f} '
            ' |  {r[dlstm]:.2f} '
            ' |  {r[slstm]:.2f}'.format(dataset=dataset, r=r)
        )


if __name__ == '__main__':
    main()
