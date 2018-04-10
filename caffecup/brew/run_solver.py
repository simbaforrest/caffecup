'''
run_solver.py in caffecup

author  : cfeng
created : 3/4/18 12:05 PM
'''

import os
import sys
import time
import argparse
from collections import OrderedDict
import pprint as pp
import glog as logger
import subprocess

def solver2dict(fpath):
    lines = open(fpath,'r').readlines()
    lst = [l.strip().split(':',1) for l in lines
           if len(l.strip())>0 and l.strip()[0]!='#']
    ret = OrderedDict()
    for it in lst:
        val = it[1].strip()
        val = val.replace('\"','')
        ret[it[0].strip()] = val
    return ret


def create_if_not_exist(path, name=''):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(name+' created: '+path)
    else:
        logger.info(name+' existed: '+path)


def make_cmd(args):
    if args.srun:
        srun = 'srun{} -X -D $PWD --gres gpu:1 '.format(' -p '+args.cluster if args.cluster else '')
        if args.jobname=='':
            args.jobname = os.path.basename(args.sdict['snapshot_prefix']) #os.path.splitext(os.path.basename(args.solver))[0]
        if args.jobname!='none':
            srun += '--job-name='+args.jobname+' '
    else:
        srun=''
    cmd = srun

    caffe = args.caffe+' train -solver '+args.solver
    logger.info('Training solver: '+args.solver)
    is_resuming = False
    if os.path.exists(args.snapshot) and args.snapshot.endswith('.solverstate'):
        is_resuming = True
        caffe += ' -snapshot '+args.snapshot
        logger.info('Resume training from: '+args.snapshot)
    cmd += caffe

    if args.logname=='':
        args.logname = args.jobname
    if args.logname!='none':
        log = ' 2>&1 | tee {}logs/{}.log'.format('-a ' if is_resuming else '', args.logname)
    else:
        log = ''
    cmd += log
    return cmd


def fastprint(msg):
    print(msg)
    sys.stdout.flush()


def run(args):
    cmd = make_cmd(args)
    args.cmd=cmd
    logger.info('to run:')
    fastprint(cmd)
    if args.dryrun:
        logger.info('dryrun: sleep 20')
        cmd = 'echo PYTHONPATH=$PYTHONPATH; for i in {1..20}; do echo $i; sleep 1; done;'

    my_env = os.environ.copy()
    if my_env.has_key('PYTHONPATH'):
        my_env['PYTHONPATH'] = ':'.join(args.additional_pythonpath)+':'+my_env['PYTHONPATH']
    else:
        my_env['PYTHONPATH'] = ':'.join(args.additional_pythonpath)
    THE_JOB = subprocess.Popen(cmd, shell=True, cwd=args.cwd, env=my_env)

    while True:
        retcode=THE_JOB.poll()
        if retcode is not None:
            logger.info('job({}) finished!'.format(args.jobname))
            break

        try:
            what = raw_input('{}: press CTRL-C to kill training>\n'.format(args.jobname))
            logger.info('continue ({:s})...'.format(what))
        except KeyboardInterrupt:
            THE_JOB.kill()
            logger.info('job({}) killed by CTRL-C!'.format(args.jobname))
            break


def main(args):
    sdict = solver2dict(args.solver)
    pp.pprint(sdict.items())
    args.sdict = sdict

    # check if valid
    if not os.path.exists(sdict['net']):
        assert(os.path.exists(
            os.path.join(args.cwd,sdict['net'])
        ))

    # snapshot_folder
    snapshot_prefix = sdict['snapshot_prefix']
    snapshot_folder = os.path.dirname(snapshot_prefix) \
        if snapshot_prefix[-1] not in [os.path.sep, os.path.altsep] \
        else snapshot_prefix
    create_if_not_exist(snapshot_folder, 'snapshot_folder')

    # logs
    create_if_not_exist(os.path.join(args.cwd,'logs'), 'logs')

    # enter loop
    run(args)
    logger.info('Done!')

def get_args(argv):
    parser = argparse.ArgumentParser(argv[0])

    parser.add_argument('-s','--solver',type=str,
                        help='path to solver')
    parser.add_argument('--caffe',type=str,default='../caffe/install/bin/caffe',
                        help='path to caffe executable')
    parser.add_argument('--cluster',type=str,default='',
                        help='which cluster')
    parser.add_argument('--snapshot',type=str,default='',
                        help='if resume previous job, must provide snapshot file (*.solverstate)')
    parser.add_argument('--jobname',type=str,default='',
                        help='cluster job name')
    parser.add_argument('--logname',type=str,default='',
                        help='log file name')

    parser.add_argument('--srun',dest='srun',action='store_true',default=True,
                        help='use srun')
    parser.add_argument('--no-srun',dest='srun',action='store_false',
                        help='DO NOT use srun')

    parser.add_argument('--dryrun',dest='dryrun',action='store_true',default=False,
                        help='sleep 20 seconds')

    args = parser.parse_args(argv[1:])
    args.cwd = os.getcwd()
    args.raw_argv = ' '.join(argv)
    args.additional_pythonpath = ['./']
    return args

if __name__ == '__main__':
    args = get_args(sys.argv)
    main(args)