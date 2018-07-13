import pysparkling
import trajnettools


def eval(input_files):
    average_l2 = {}
    average_l2_non_linear = {}
    final_l2 = {}

    sc = pysparkling.Context()
    paths = (sc
             .wholeTextFiles(input_files)
             .mapValues(trajnettools.readers.trajnet)
             .cache())

    # Kalman Filter (Lin)
    kalman_predictions = paths.mapValues(trajnettools.kalman.predict)
    paths_kf = (paths
                .mapValues(lambda paths: paths[0])
                .leftOuterJoin(kalman_predictions)
                .cache())
    average_l2_kf = paths_kf.mapValues(trajnettools.metrics.average_l2).cache()

    # determine non-linear sequences
    mean_kf = average_l2_kf.values().mean()
    nonlinear_sequences = set(average_l2_kf
                              .filter(lambda seq_kf: seq_kf[1] > mean_kf)
                              .keys()
                              .toLocalIterator())

    # N
    average_l2['N'] = paths.count()
    average_l2_non_linear['N'] = len(nonlinear_sequences)
    final_l2['N'] = paths.count()

    # KF
    average_l2['kf'] = mean_kf
    average_l2_non_linear['kf'] = average_l2_kf.filter(lambda seq_v: seq_v[0] in nonlinear_sequences).values().mean()
    final_l2['kf'] = paths_kf.values().map(trajnettools.metrics.final_l2).mean()

    # LSTM
    lstm_predictor = trajnettools.lstm.LSTMPredictor.load('output/vanilla_lstm.pkl')
    lstm_predictions = paths.mapValues(lstm_predictor)
    paths_lstm = (paths
                  .mapValues(lambda paths: paths[0])
                  .leftOuterJoin(lstm_predictions)
                  .cache())
    average_l2_lstm = paths_lstm.mapValues(trajnettools.metrics.average_l2).cache()
    average_l2['lstm'] = average_l2_lstm.values().mean()
    average_l2_non_linear['lstm'] = average_l2_lstm.filter(lambda seq_v: seq_v[0] in nonlinear_sequences).values().mean()
    final_l2['lstm'] = paths_lstm.values().map(trajnettools.metrics.final_l2).mean()

    # OLSTM
    # olstm_predictor = trajnettools.lstm.OLSTMPredictor.load('output/olstm.pkl')
    olstm_predictor = trajnettools.lstm.LSTMPredictor.load('output/occupancy_lstm.pkl')
    olstm_predictions = paths.mapValues(olstm_predictor)
    paths_olstm = (paths
                   .mapValues(lambda paths: paths[0])
                   .leftOuterJoin(olstm_predictions)
                   .cache())
    average_l2_olstm = paths_olstm.mapValues(trajnettools.metrics.average_l2).cache()
    average_l2['olstm'] = average_l2_olstm.values().mean()
    average_l2_non_linear['olstm'] = average_l2_olstm.filter(lambda seq_v: seq_v[0] in nonlinear_sequences).values().mean()
    final_l2['olstm'] = paths_olstm.values().map(trajnettools.metrics.final_l2).mean()

    return average_l2, average_l2_non_linear, final_l2


def main():
    datasets = [
        'output/val/biwi_eth/*.txt',
        'output/val/biwi_hotel/*.txt',
        'output/val/crowds_zara01/*.txt',
        'output/val/crowds_zara02/*.txt',
        'output/val/crowds_uni_examples/*.txt',

        'output/val/crowds_zara03/*.txt',
        'output/val/crowds_students001/*.txt',
        'output/val/crowds_students003/*.txt',
        'output/val/mot_pets2009_s2l1/*.txt',
    ]
    results = {dataset
               .replace('output/', '')
               .replace('.txt', ''): eval(dataset)
               for dataset in datasets}

    print('## Average L2 [m]')
    print('{dataset:>30s} |   N  |  Lin | LSTM | O-LSTM'.format(dataset=''))
    for dataset, (r, _, _) in results.items():
        print(
            '{dataset:>30s}'
            ' | {r[N]:>4}'
            ' | {r[kf]:.2f}'
            ' | {r[lstm]:.2f}'
            ' | {r[olstm]:.2f}'.format(dataset=dataset, r=r)
        )

    print('')
    print('## Average L2 (non-linear sequences) [m]')
    print('{dataset:>30s} |   N  |  Lin | LSTM | O-LSTM'.format(dataset=''))
    for dataset, (_, r, _) in results.items():
        print(
            '{dataset:>30s}'
            ' | {r[N]:>4}'
            ' | {r[kf]:.2f}'
            ' | {r[lstm]:.2f}'
            ' | {r[olstm]:.2f}'.format(dataset=dataset, r=r)
        )

    print('')
    print('## Final L2 [m]')
    print('{dataset:>30s} |   N  |  Lin | LSTM | O-LSTM'.format(dataset=''))
    for dataset, (_, _, r) in results.items():
        print(
            '{dataset:>30s}'
            ' | {r[N]:>4}'
            ' | {r[kf]:.2f}'
            ' | {r[lstm]:.2f}'
            ' | {r[olstm]:.2f}'.format(dataset=dataset, r=r)
        )


if __name__ == '__main__':
    main()
