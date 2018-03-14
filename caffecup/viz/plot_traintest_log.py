'''
plot_traintest_log.py in caffecup

author  : cfeng
created : 3/12/17 5:00 PM
modified: 5/16/17
'''

import os
import sys
import glob
import time
import argparse

import numpy as np
# import matplotlib
# matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

import re

def moving_average(data_set, periods):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

def get_training_loss(log, loss_name='loss', smooth_len=-1, smooth_ratio=-1):
    '''
    get training loss from the log buffer
    :param log: a str corresponding to the log file
    :param loss_name: default='loss'
    :return: (iters, losses), a tuple of two np.array
    '''
    loss_pattern = r"Iteration (?P<iter_num>\d+).*, {} = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)".format(
        loss_name)
    losses = []
    loss_iters = []
    for r in re.findall(loss_pattern, log):
        loss_iters.append(int(r[0]))
        losses.append(float(r[1]))
    losses = np.array(losses)
    loss_iters = np.array(loss_iters)

    if len(loss_iters)<=0:
        return None, None

    # smooth
    if smooth_len<0:
        if smooth_ratio<0:
            smooth_len = len(loss_iters)/100
        else:
            smooth_len = int(len(loss_iters)*smooth_ratio)
    if smooth_len>0:
        losses = moving_average(losses, smooth_len)
        loss_iters = loss_iters[len(loss_iters)-len(losses):]

    return loss_iters, losses


def get_testing_accuracy(log, accuracy_name='accuracy'):
    '''
    get testing accuracy from the log buffer
    :param log: a str corresponding to the log file
    :param accuracy_name: default='accuracy'
    :return: (iters, accuracies), a tuple of two np.array
    '''
    acc_pattern = r"Test net output #.*: {} = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)".format(
        accuracy_name)
    acc_iter_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#.\)"
    accuracies = []
    accu_iters = []
    for r in re.findall(acc_pattern, log):
        accuracies.append(float(r[0]))
    for r in re.findall(acc_iter_pattern, log):
        accu_iters.append(int(r))
    accuracies = np.array(accuracies)
    accu_iters = np.array(accu_iters)

    if len(accu_iters)<=0 or len(accu_iters)<len(accuracies):
        return None, None

    if len(accu_iters)==len(accuracies)+1: #for the case where testing is not finished yet
        accu_iters = accu_iters[:-1]

    if len(accu_iters)==0:
        return None, None

    return accu_iters, accuracies


def main_folder(args, logs):
    if args.all_in_one==1:
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax2 = ax1.twinx()
    else:
        fig, ax1, ax2 = None, None, None

    colors = plt.cm.ScalarMappable(cmap='jet').to_rgba(list(xrange(len(logs))))
    for l,clr in zip(logs,colors):
        args.input = l
        main_file(args, fig, ax1, ax2, clr=clr)
        print('processed: '+l)

    if args.all_in_one==1:
        output_folder = args.output_folder if args.output_folder else os.path.dirname(args.input)
        output_prefix = os.path.join(
            output_folder,
            args.output_name
        )
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()
        # figManager.window.showMaximized()
        if args.output_name!='none':
            plt.savefig(output_prefix+'.png', bbox_inches='tight')
        plt.show()
    print('Done!')


def main_file(args, fig=None, ax1=None, ax2=None, clr=None):
    output_folder = args.output_folder if args.output_folder else os.path.dirname(args.input)
    output_prefix = os.path.join(
        output_folder,
        os.path.splitext(os.path.basename(args.input))[0]
    )

    log = open(args.input, 'r').read()
    assert(len(log)>0)

    it_l, loss = get_training_loss(log, args.loss_name, args.smooth_len, args.smooth_ratio)
    it_a, accu = get_testing_accuracy(log, args.accuracy_name)

    do_plot_loss = it_l is not None
    do_plot_accu = it_a is not None

    if not (do_plot_accu or do_plot_loss):
        print('Nothing to plot!')
        return

    if do_plot_accu:
        max_accu_pos = np.argmax(accu)

    if args.all_in_one==0:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        loss_color = 'r'
        accu_color = 'b'
        train_label = 'train loss'
        if do_plot_accu:
            test_label = 'max test accuracy {:.3f} @ {}'.format(
                accu[max_accu_pos], it_a[max_accu_pos])
        plt.title(os.path.basename(args.input))
    else:
        loss_color = np.random.rand(3,1) if clr is None else clr
        accu_color = loss_color.copy()
        train_label = 'train loss: '+os.path.splitext(os.path.basename(args.input))[0]
        if do_plot_accu:
            test_label = 'max test accuracy {:.3f} @ {}\n{}'.format(
                accu[max_accu_pos], it_a[max_accu_pos], os.path.splitext(os.path.basename(args.input))[0])
        if args.output_name!='none':
            plt.title(args.output_name)

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax1.plot(it_l, loss, '-', label=train_label, color=loss_color)
    if isinstance(args.axis_left, list):
        ax1.set_ylim(args.axis_left)
    ax1.legend(loc='upper left', framealpha=0.8)

    if do_plot_accu:
        ax2.set_ylabel('accuracy')
        ax2.plot(it_a, accu, 'o--', label=test_label, color=accu_color)
        if isinstance(args.axis_right, list):
            ax2.set_ylim(args.axis_right)
        ax2.legend(loc='lower right', framealpha=0.8)

    if args.all_in_one==0:
        if do_plot_accu:
            ax2.text(0.5, 0.5, 'max accuracy\n{:.3f} @ {}'.format(
                accu[max_accu_pos], it_a[max_accu_pos]), fontsize=12, transform=ax2.transAxes)
        plt.savefig(output_prefix+'.png', bbox_inches='tight')
        plt.show()


def main(args):
    if os.path.isdir(args.input):
        print('process all .logs in folder: '+args.input)
        args.root = args.input
        args.output_folder = args.output_folder if args.output_folder else args.root
        args.output_name = args.output_name if args.output_name else 'all'

        logs = sorted(os.listdir(args.root))
        if args.include_filter!='':
            re_inc = re.compile(args.include_filter)
            logs = [f for f in logs if re_inc.search(f) is not None]
        if args.exclude_filter!='':
            re_exc = re.compile(args.exclude_filter)
            logs = [f for f in logs if re_exc.search(f) is None]
        logs = [os.path.join(args.root,f)
                for f in logs
                if os.path.isfile(os.path.join(args.root,f)) and (f.rfind('.log')>=0 or f.rfind('.txt')>=0)]

        main_folder(args, logs)
        return

    args.all_in_one=0 #input is not folder, so disable all_in_one plot
    main_file(args)

def get_args_gui():
    from Tkinter import Tk
    from tkFileDialog import askopenfilename
    tkroot = Tk()
    tkroot.withdraw()
    filename = askopenfilename(title='Choose a log file',
                               initialdir='logs',
                               filetypes=(('logs', '*.log'),('all','*x*')))
    tkroot.destroy()
    if not filename:
        sys.exit(0)

    args = type('args',(),{})
    args.input = filename
    args.loss_name = 'loss'
    args.accuracy_name = 'accuracy'
    args.output_folder = ''
    args.smooth_len = -1
    args.smooth_ratio = -1.
    args.all_in_one = 0
    args.output_name = ''
    args.exclude_filter = ''
    args.include_filter = ''
    args.axis_left=[0,4]
    args.axis_right=[0.5,1]
    return args

def get_args(argv):
    class ThrowingArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            raise ValueError()

    parser = ThrowingArgumentParser(argv[0])
    parser.add_argument('input', type=str,
                        help="path/to/log/file or path/to/logs/folder")
    parser.add_argument('-l', '--loss_name', type=str, default='loss',
                        help="loss name")
    parser.add_argument('-a', '--accuracy_name', type=str, default='accuracy',
                        help="accuracy name")
    parser.add_argument('-o', '--output_folder', type=str, default='',
                        help="output folder (default to the log file folder)")
    parser.add_argument('-s', '--smooth_len', type=int, default=-1,
                        help="whether to smooth loss (<0 or >0), or not (==0)")
    parser.add_argument('-r', '--smooth_ratio', type=float, default=-1,
                        help="whether to smooth loss by a ratio of whole sequence length (<0 or >0), or not (==0)")
    parser.add_argument('-1', '--all_in_one', type=int, default=0,
                        help="whether to plot all curves in one figure (1) or not (0)")
    parser.add_argument('-n', '--output_name', type=str, default='',
                        help="output file name for all_in_one mode")
    parser.add_argument('-x', '--exclude_filter', type=str, default='',
                        help="do not include log files filtered by this")
    parser.add_argument('-i', '--include_filter', type=str, default='',
                        help="only include log files filtered by this")
    parser.add_argument('-axl', '--axis_left', type=lambda v: eval(v), default=None,
                        help="left axis limit")
    parser.add_argument('-axr', '--axis_right', type=lambda v: eval(v), default=None,
                        help="right axis limit")

    args = parser.parse_args(argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    return args

if __name__ == '__main__':
    args = get_args(sys.argv)
    main(args)