'''
solver.py in caffecup

author  : cfeng
created : 10/22/17 3:34 PM
'''

import os
import sys
import argparse
from string import Template

import numpy as np

from ..design.builder import safe_create_file
import errno

__all__ = [
    'TrainTestSolver'
]

def safe_create_folder(fpath):
    try:
        if not os.path.exists(fpath):
            os.makedirs(fpath)
            print('created:'+fpath)
            sys.stdout.flush()
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class TrainTestSolver(object):

    def __init__(self, fp, add_gen_log=True):
        self.fp, self.filepath, self.filedir = safe_create_file(fp, add_gen_log)

    def build(
            self,
            traintest_net,

            n_train_data,
            train_batch_size,

            n_test_data,
            test_batch_size,
            test_interval,

            base_lr,
            lr_policy='poly',

            solver_type='SGD',
            solver_mode='GPU',

            iter_size=1,
            n_epoch=100,
            momentum=0.9,
            momentum2=0.999, #for Adam
            weight_decay=1e-5,
            average_loss=100,
            display=1,
            test_initialization=False,
            snapshot_interval=-1,   #default test_interval
            snapshot_folder='snapshots',
            snapshot_prefix='',     #default basename(net)_lr_wd
            lr_stepsize=-1,         #default int(0.5*max_iter)
            lr_gamma=0.1,
            lr_multistepsize=None,  #default [int(0.5*max_iter), int(0.75*max_iter)]
    ):

        max_iter = n_train_data*n_epoch/(train_batch_size*iter_size)
        test_iter = int(np.ceil(float(n_test_data)/test_batch_size))
        if not test_initialization:
            test_initialization = 'false'
        else:
            test_initialization = 'true'
        if snapshot_interval<0:
            snapshot_interval = test_interval #int(0.01*max_iter)
        if not snapshot_prefix:
            snapshot_prefix = os.path.splitext(os.path.basename(traintest_net))[0]
            if snapshot_prefix.startswith('traintest_'):
                snapshot_prefix = snapshot_prefix.replace('traintest_','',1)
            lr_exponent = int(np.floor(np.log10(np.abs(base_lr))))
            lr_significand = int(np.power(10, -lr_exponent)*base_lr)
            snapshot_prefix+= '_{:s}{:s}l{:d}e{:d}'.format(solver_type[0].upper(), lr_policy[0],
                                                           lr_significand, lr_exponent)
            if weight_decay!=1e-5:
                snapshot_prefix+='_w{:1.0e}'.format(weight_decay)
            snapshot_prefix=os.path.join(snapshot_folder,snapshot_prefix)
        safe_create_folder(os.path.dirname(snapshot_prefix))

        if lr_policy=='poly':
            lr_policy_block = \
'''
lr_policy: "poly"
power: 1
'''
        elif lr_policy=='step':
            if lr_stepsize<0:
                lr_stepsize=int(0.5*max_iter)
            lr_policy_block = \
'''
lr_policy: "step"
stepsize: {:d}
gamma:    {:f}
'''.format(lr_stepsize, lr_gamma)
        elif lr_policy=='multistep':
            if lr_multistepsize is None:
                lr_multistepsize=[int(0.5*max_iter), int(0.75*max_iter)]
            lr_multistepsize_str = '\n'.join(['stepvalue: {:d}'.format(r) for r in lr_multistepsize])
            lr_policy_block = \
'''
lr_policy: "multistep"
{}
'''.format(lr_multistepsize_str)
        elif lr_policy=='fixed':
            lr_policy_block = \
'''
lr_policy: "fixed"
'''
        else:
            raise NotImplementedError('TODO: '+lr_policy)

        s=Template(
'''
net: "$traintest_net"
#------
# n_train_data*n_epoch/(train_batch_size*iter_size) = $n_train_data*$n_epoch/($train_batch_size*$iter_size) = $max_iter
iter_size: $iter_size
max_iter: $max_iter
#------
# n_test_data/test_batch_size = $n_test_data/$test_batch_size = $test_iter
test_iter: $test_iter
test_interval: $test_interval
test_initialization: $test_initialization

#------
base_lr: $base_lr
$lr_policy_block
momentum: $momentum
momentum2: $momentum2
weight_decay: $weight_decay

#------
average_loss: $average_loss
display: $display
#------
snapshot_prefix: "$snapshot_prefix"
snapshot: $snapshot_interval

#------
type: "$solver_type"
solver_mode: $solver_mode
#random_seed: 13598234
'''
        )

        self.fp.write(s.substitute(locals()))
        self.fp.close()


def main(args):
    TrainTestSolver('models/solver_test.ptt').build(
        'models/traintest_test.ptt',
        60000, 100,
        10000, 100, 1000,
        2e-3
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)