import argparse
import json
import numpy as np
import pysparkling
import trajnettools.show


def read_log(path):
    sc = pysparkling.Context()
    return (sc
            .textFile(path)
            .filter(lambda line: line.startswith(('{', 'json:')))
            .map(lambda line: json.loads(line.strip('json:')))
            .groupBy(lambda data: data.get('type'))
            .collectAsMap())


def plots(log_files, output_prefix, labels=None):
    if not labels:
        labels = log_files
    datas = [read_log(log_file) for log_file in log_files]

    with trajnettools.show.canvas(output_prefix + 'lr.png') as ax:
        for data, label in zip(datas, labels):
            if 'train' in data:
                x = [row.get('epoch') for row in data['train']]
                y = [row.get('lr') for row in data['train']]
                ax.plot(x, y, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('learning rate')
        ax.set_yscale('log', nonposy='clip')
        ax.legend()

    with trajnettools.show.canvas(output_prefix + 'val.png') as ax:
        for data, label in zip(datas, labels):
            if 'val' in data:
                x = [row.get('epoch') for row in data['val']]
                y = [row.get('accuracy', row.get('prec@1')) for row in data['val']]
                ax.plot(x, y, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        ax.legend()

    with trajnettools.show.canvas(output_prefix + 'epoch-loss.png') as ax:
        for data, label in zip(datas, labels):
            val_color = None
            if 'val-epoch' in data:
                x = [row.get('epoch') for row in data['val-epoch']]
                y = [row.get('loss') for row in data['val-epoch']]
                val_line, = ax.plot(x, y, label=label + ' (val)')
                val_color = val_line.get_color()
            if 'train-epoch' in data:
                x = [row.get('epoch') for row in data['train-epoch']]
                y = [row.get('loss') for row in data['train-epoch']]
                ax.plot(x, y, label=label + ' (train)',
                        color=val_color, linestyle='dotted')

        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        if min(y) > -0.1:
            ax.set_yscale('log', nonposy='clip')
        ax.legend()

    with trajnettools.show.canvas(output_prefix + 'time.png') as ax:
        for data, label in zip(datas, labels):
            if 'train' in data:
                x = np.array([row.get('epoch') + row.get('batch') / row.get('n_batches')
                              for row in data['train']])
                y = np.array([row.get('data_time') / row.get('time') * 100.0 for row in data['train']])
                if len(x) / x[-1] > 10:
                    stride = int(len(x) / x[-1] / 3.0)  # 3 per epoch
                    x_binned = np.array([x[i] for i in range(0, len(x), stride)])
                    y_mean = [np.mean(y[np.logical_and(x1 < x, x < x2)])
                              for x1, x2 in zip(x_binned[:-1], x_binned[1:])]
                    y_min = [np.min(y[np.logical_and(x1 < x, x < x2)])
                             for x1, x2 in zip(x_binned[:-1], x_binned[1:])]
                    y_max = [np.max(y[np.logical_and(x1 < x, x < x2)])
                             for x1, x2 in zip(x_binned[:-1], x_binned[1:])]
                    ax.fill_between(x_binned[:-1], y_min, y_max, alpha=0.2)
                    ax.plot(x_binned[:-1], y_mean, label=label)
                else:
                    ax.plot(x, y, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('data preprocessing time [%]')
        ax.set_ylim(0, 100)
        ax.legend()

    with trajnettools.show.canvas(output_prefix + 'train.png') as ax:
        for data, label in zip(datas, labels):
            if 'train' in data:
                x = np.array([row.get('epoch') + row.get('batch') / row.get('n_batches')
                              for row in data['train']])
                y = np.array([row.get('loss') for row in data['train']])
                if len(x) / x[-1] > 10:
                    stride = int(len(x) / x[-1] / 3.0)  # 3 per epoch
                    x_binned = np.array([x[i] for i in range(0, len(x), stride)])
                    y_mean = [np.mean(y[np.logical_and(x1 < x, x < x2)])
                              for x1, x2 in zip(x_binned[:-1], x_binned[1:])]
                    y_min = [np.min(y[np.logical_and(x1 < x, x < x2)])
                             for x1, x2 in zip(x_binned[:-1], x_binned[1:])]
                    y_max = [np.max(y[np.logical_and(x1 < x, x < x2)])
                             for x1, x2 in zip(x_binned[:-1], x_binned[1:])]
                    ax.fill_between(x_binned[:-1], y_min, y_max, alpha=0.2)
                    ax.plot(x_binned[:-1], y_mean, label=label)
                else:
                    ax.plot(x, y, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_ylim(-5, 2)
        if min(y_mean) > -0.1:
            ax.set_yscale('log', nonposy='clip')
        ax.legend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', nargs='+',
                        help='path to log file')
    parser.add_argument('--label', nargs='+',
                        help='labels in the same order as files')
    parser.add_argument('-o', '--output', default=None,
                        help='output prefix (default is log_file + .)')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.log_file[-1] + '.'

    plots(args.log_file, args.output, args.label)


if __name__ == '__main__':
    main()
