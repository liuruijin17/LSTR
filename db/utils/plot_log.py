import os
import re
import argparse
import datetime

import numpy as np
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib.pyplot as plt

ITER_PATTERN = re.compile(
    '^\[([^\]]*)\]\ .*Epoch \[(\d*)/(\d*).*Step\ \[(\d*)/(\d*).*Loss: (\d*\.?\d*)\ \((.*)\).*s/iter:\ -?(\d*\.?\d*).*lr:\ ([^\ ]*)$'  # noqa: E501
)
LOSS_COMP_PATTERN = re.compile('(\w+):\ (\d*\.?\d*)')  # noqa: w605
EPOCH_PATTERN = re.compile('^\[([^\]]*)\]\ .*Epoch \[(\d*)/(\d*).*Val\ loss: (\d*\.?\d*)$')  # noqa: w605
EXPS_DIR = '../data_lane-regression/experiments'

# TODO: refactor this file


def smooth_curve(xs, factor):
    smoothed = [None] * len(xs)
    smoothed[0] = xs[0]
    for i in range(1, len(xs)):
        smoothed[i] = xs[i] * (1 - factor) + smoothed[i - 1] * factor

    return smoothed


def plot_loss(data,
              fig,
              ax,
              label,
              plot_lr=True,
              smoothing=0,
              xaxis='time',
              only_epoch_end=False,
              plot_val=False,
              plot_loss_comps=False):
    iter_data = data['iter_update']
    epoch_data = data['epoch_update']
    now = datetime.datetime.today()
    if xaxis == 'epoch':
        if only_epoch_end:
            iter_data = [d for d in iter_data if d['iter_nb'] == d['total_iters']]
        x = [d['epoch'] + d['iter_nb'] * 1.0 / d['total_iters'] for d in iter_data]
    elif xaxis == 'time':
        d0 = iter_data[0]['date']
        x = [now + (d['date'] - d0) for d in iter_data]
    elif xaxis == 'iter':
        x = [(d['epoch'] - 1) * d['total_iters'] + d['iter_nb'] for d in iter_data]
    loss = [d['loss'] for d in iter_data]
    if plot_loss_comps:
        loss_comps = {comp: [d['loss_comps'][comp] for d in iter_data] for comp in iter_data[0]['loss_comps']}
    if plot_val:
        val_loss = [d['val_loss'] for d in epoch_data]
        if xaxis == 'epoch':
            val_loss_x = [d['epoch'] for d in epoch_data]
        else:
            val_loss_d0 = epoch_data[0]['date']
            val_loss_x = [now + (d['date'] - val_loss_d0) for d in epoch_data]
    loss_smooth = smooth_curve(loss, factor=smoothing)
    if plot_lr:
        lr = [d['lr'] for d in iter_data]
        lr_decays = [(iter_data[i + 1]['epoch'], iter_data[i]['lr'], iter_data[i + 1]['lr'])
                     for i in range(len(iter_data) - 1) if iter_data[i + 1]['lr'] != iter_data[i]['lr']]
        if len(lr_decays) < 10:
            for epoch, old, new in lr_decays:
                ax.axvline(x=epoch, linestyle='--')
        ax.plot(x, lr, label='LR: {}'.format(label))
    ax.set_yscale('log')
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    loss_line = ax.plot(x, loss_smooth)[0]
    loss_line_color = np.array(colors.to_rgba(loss_line.get_color()))
    loss_line_color[-1] = 0.5
    if plot_loss_comps:
        for loss_comp in loss_comps:
            line = ax.plot(x, smooth_curve(loss_comps[loss_comp], smoothing))[0]
            line_color = np.array(colors.to_rgba(line.get_color()))
            line_color[-1] = 0.5
            ax.plot(x, loss_comps[loss_comp], label='{}: {}'.format(loss_comp, label), color=line_color)
    ax.plot(x, loss, label='Train Loss: {}'.format(label), color=loss_line_color)
    if plot_val:
        ax.plot(val_loss_x, val_loss, label='Val Loss: {}'.format(label))
    if xaxis == 'time':
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M"'))


def parse_line(line):
    iter_match = re.match(ITER_PATTERN, line)
    epoch_match = re.match(EPOCH_PATTERN, line)
    data = {}
    if iter_match is not None:
        date, epoch, total_epochs, iter_nb, total_iters, loss, loss_comps, speed, lr = iter_match.groups()
        date, epoch, total_epochs, iter_nb, total_iters, loss, speed, lr = datetime.datetime.strptime(
            date, '%Y-%m-%d %H:%M:%S,%f'), int(epoch), int(total_epochs), int(iter_nb), int(total_iters), float(
                loss), float(speed), float(lr)
        loss_comps = re.findall(LOSS_COMP_PATTERN, loss_comps)
        loss_comps = {d[0]: float(d[1]) for d in loss_comps}
        data['iter_update'] = {
            'date': date,
            'epoch': epoch,
            'total_epochs': total_epochs,
            'iter_nb': iter_nb,
            'total_iters': total_iters,
            'loss': loss,
            'speed': date,
            'loss_comps': loss_comps,
            'lr': lr,
        }
    if epoch_match is not None:
        date, epoch, _, val_loss = epoch_match.groups()
        date, epoch, val_loss = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S,%f'), int(epoch), float(val_loss)
        data['epoch_update'] = {'date': date, 'epoch': epoch, 'val_loss': val_loss}

    return data


def parse_log(log_path):
    with open(log_path, 'r') as log_file:
        lines = [line.rstrip() for line in log_file.readlines()]
    data = {'iter_update': [], 'epoch_update': []}
    for line in lines:
        line_data = parse_line(line)
        for key in line_data:
            data[key].append(line_data[key])
    return data


def get_logfilepath(exp_name):
    return os.path.join(EXPS_DIR, exp_name, 'log.txt')


def parse_args():
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('exp_name', nargs='*', default=None, help='Experiment names')
    parser.add_argument('--smoothing', type=float, default=0.99, help='Experiment name')
    parser.add_argument('--xaxis', default='time', help='X axis (`time`or `epoch`)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for exp_name in args.exp_name:
        log_filepath = get_logfilepath(exp_name)
        data = parse_log(log_filepath)
        plot_loss(data, fig, ax, exp_name, smoothing=args.smoothing, xaxis=args.xaxis)

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.legend()
    plt.show()
