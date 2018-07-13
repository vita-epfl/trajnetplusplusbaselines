from contextlib import contextmanager
import os

import matplotlib.pyplot as plt
import pysparkling
import trajnettools


@contextmanager
def show(fig_file=None, **kwargs):
    fig, ax = plt.subplots(**kwargs)

    yield ax

    fig.set_tight_layout(True)
    if fig_file:
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        fig.savefig(fig_file, dpi=300)
    fig.show()
    plt.close(fig)


def predict(input_files):
    sc = pysparkling.Context()
    paths = (sc
             .wholeTextFiles(input_files)
             .mapValues(trajnettools.readers.trajnet)
             .cache())
    kalman_predictions = paths.mapValues(trajnettools.kalman.predict)
    lstm_predictor = trajnettools.lstm.LSTMPredictor.load('output/vanilla_lstm.pkl')
    lstm_predictions = paths.mapValues(lstm_predictor)
    olstm_predictor = trajnettools.lstm.LSTMPredictor.load('output/occupancy_lstm.pkl')
    olstm_predictions = paths.mapValues(olstm_predictor)
    # olstm_predictions = paths.mapValues(lambda _: None)
    olstm_others = paths.mapValues(lambda _: None)

    paths = (
        paths
        .leftOuterJoin(kalman_predictions)
        .leftOuterJoin(lstm_predictions).mapValues(lambda v: v[0] + (v[1],))
        .leftOuterJoin(olstm_predictions).mapValues(lambda v: v[0] + (v[1],))
        .leftOuterJoin(olstm_others).mapValues(lambda v: v[0] + (v[1],))
    )
    for scene, (gt, kf, lstm, olstm, olstm_others) in paths.toLocalIterator():
        output_file = (scene
                       .replace('/train/', '/train_plots/')
                       .replace('/test/', '/test_plots/')
                       .replace('.txt', '.png'))
        with show(output_file) as ax:
            # KF prediction
            ax.plot([gt[0][8].x] + [r.x for r in kf],
                    [gt[0][8].y] + [r.y for r in kf], color='orange', label='KF')
            ax.plot([kf[-1].x], [kf[-1].y], color='orange', marker='o', linestyle='None')

            # LSTM prediction
            ax.plot([gt[0][8].x] + [r.x for r in lstm],
                    [gt[0][8].y] + [r.y for r in lstm], color='blue', label='LSTM')
            ax.plot([lstm[-1].x], [lstm[-1].y], color='blue', marker='o', linestyle='None')

            # OLSTM prediction
            if olstm is not None:
                ax.plot([gt[0][8].x] + [r.x for r in olstm],
                        [gt[0][8].y] + [r.y for r in olstm], color='green', label='O-LSTM')
                ax.plot([olstm[-1].x], [olstm[-1].y], color='green', marker='o', linestyle='None')

            # LSTM predictions for OLSTM occupancy
            if olstm_others is not None:
                olstm_others_by_ped = zip(*olstm_others)
                for olstm_other in olstm_others_by_ped:
                    x = [x for x, _ in olstm_other if x is not None]
                    y = [y for _, y in olstm_other if y is not None]
                    if not x or not y:
                        continue
                    ax.plot(x, y, color='gray', linestyle='dotted', marker='o', markersize=3.5)

            # ground truths
            for i_gt, g in enumerate(gt):
                xs = [r.x for r in g]
                ys = [r.y for r in g]

                # markers
                label_start = None
                label_end = None
                if i_gt == 0:
                    label_start = 'start'
                    label_end = 'end'
                ax.plot(xs[0:1], ys[0:1], color='black', marker='x',
                        label=label_start, linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color='black', marker='o',
                        label=label_end, linestyle='None')

                # ground truth lines
                ls = 'dotted' if i_gt > 0 else 'solid'
                label = None
                if i_gt == 0:
                    label = 'primary'
                if i_gt == 1:
                    label = 'others'
                ax.plot(xs, ys, color='black', linestyle=ls, label=label)

            # frame
            ax.legend()
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')


if __name__ == '__main__':
    predict('output/val/biwi_eth/20?.txt')
    # predict('output/val/biwi_eth/98.txt')
    # predict('output/val/biwi_eth/362.txt')
    # predict('output/train/biwi_hotel/?.txt')
