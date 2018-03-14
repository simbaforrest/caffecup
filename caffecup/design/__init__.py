import builder
import operator
from learable import *

__all__=[
    'builder',
    'operator',
    'learable',
    'Designer'
]

class Designer(operator.TensorOp,
               operator.NormalizationOp,
               operator.FinalOp,
               learable.Learable):
    '''
    An easily extendable class for generating Caffe Network specification file (*.ptt/*.prototxt)
    '''

    def __init__(self, fp, add_gen_log=True):
        super(Designer, self).__init__(fp, add_gen_log=add_gen_log)