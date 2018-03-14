'''
operator.py in caffecup

author  : cfeng
created : 3/8/18 4:49 AM
'''

from string import Template
import re

import numpy as np

from builder import BaseBuilder

__all__ = [
    'TensorOp',
    'NormalizationOp',
    'AggregationOp',
    'EltOp',
    'FinalOp'
]

class TensorOp(BaseBuilder):

    def _check_matmul_shape(self, input_X, input_Y, output_Z, mode='XY'):
        shape_X = self.shape_of(input_X)
        shape_Y = self.shape_of(input_Y)
        assert(len(shape_X)>=2)
        assert(len(shape_Y)>=2)

        if mode=='XY':
            assert(shape_X[-1]==shape_Y[-2])
            shape_Z = [shape_X[-2],shape_Y[-1]]
        elif mode=='XtY':
            assert(shape_X[-2]==shape_Y[-2])
            shape_Z = [shape_X[-1],shape_Y[-1]]
        elif mode=='XYt':
            assert(shape_X[-1]==shape_Y[-1])
            shape_Z = [shape_X[-2],shape_Y[-2]]
        else:
            raise ValueError()

        if len(shape_X)>2 and len(shape_Y)>2:
            assert(len(shape_X)==len(shape_Y))

        if len(shape_X)>2:
            shape_Z = shape_X[:-2] + shape_Z
        elif len(shape_Y)>2:
            shape_Z = shape_Y[:-2] + shape_Z

        assert(self.register_new_blob(output_Z, shape_Z))

    def matmul(
            self,
            input_X,
            input_Y,
            output_Z,
            name=''
    ):
        self._check_matmul_shape(input_X, input_Y, output_Z, 'XY')
        if name=='':
            name = 'matmul_{}_{}'.format(input_X, input_Y)
        s=Template(
'''layer {
  name: "$name" type: "MatrixMultiplication"
  bottom: "$input_X"
  bottom: "$input_Y"
  top:    "$output_Z"
}
'''
        )
        self.fp.write(s.substitute(locals()))
        return self

    def matmulXtY(
            self,
            input_X,
            input_Y,
            output_Z,
            name=''
    ):
        self._check_matmul_shape(input_X, input_Y, output_Z, 'XtY')
        if name=='':
            name = 'matmulXtY_{}_{}'.format(input_X, input_Y)
        s=Template(
'''layer {
  name: "$name" type: "MatrixMultiplicationXt"
  bottom: "$input_X"
  bottom: "$input_Y"
  top:    "$output_Z"
}
'''
        )
        self.fp.write(s.substitute(locals()))
        return self

    def matmulXYt(
            self,
            input_X,
            input_Y,
            output_Z,
            name=''
    ):
        self._check_matmul_shape(input_X, input_Y, output_Z, 'XYt')
        if name=='':
            name = 'matmulXYt_{}_{}'.format(input_X, input_Y)
        s=Template(
'''layer {
  name: "$name" type: "MatrixMultiplicationYt"
  bottom: "$input_X"
  bottom: "$input_Y"
  top:    "$output_Z"
}
'''
        )
        self.fp.write(s.substitute(locals()))
        return self

    def matT(
            self,
            input,
            output,
            name=''
    ):
        assert(input!=output)
        shape = self.shape_of(input)
        shape[-2:]=shape[-2:][::-1]
        assert(self.register_new_blob(output, shape))
        if name=='':
            name = 'matT_{}'.format(input)
        s=Template(
'''layer {
  name: "$name" type: "MatrixTranspose"
  bottom: "$input"
  top:    "$output"
}
'''
        )
        self.fp.write(s.substitute(locals()))
        return self

    def tensorT(
            self,
            input,
            output,
            orders, #Ouput to Input
            name=''
    ):
        assert(input!=output)
        shape = np.asarray(self.shape_of(input))
        assert(len(shape)==len(orders))
        shape = shape[orders].tolist()
        assert(self.register_new_blob(output, shape))
        if name=='':
            name = 'tensorT_{}'.format(input)

        order_str = ' '.join(['order: {:d}'.format(o) for o in orders])

        s=Template(
'''layer {
  name: "$name" type: "TensorTranspose"
  bottom: "$input"
  top:    "$output"
  tensor_transpose_param {
    $order_str
  }
}
'''
        )
        self.fp.write(s.substitute(locals()))
        return self

#########################################################

class NormalizationOp(BaseBuilder):

    def l2normalization(
            self,
            input,
            output,
            is_global=True,
            axis=0,
            eps=1e-6,
            name=''
    ):
        assert(self.register_new_blob(output, self.shape_of(input)))
        if name=='':
            name = 'l2normalize{}_{}'.format(('_global' if is_global else ''), input)
        s=Template(
'''layer {
  name: "$name" type: "L2Normalization"
  bottom: "$input"
  top: "$output"
  l2n_param { eps: $eps %s }
}
'''%('is_global: false axis: $axis' if not is_global else '')
        )
        self.fp.write(s.substitute(locals()))
        return self

#     def sumnormalization(
#             self,
#             input,
#             output,
#             is_global=True,
#             axis=0,
#             eps=1e-6,
#             name=''
#     ):
#         if name=='':
#             name = 'sumnormalize{}_{}'.format(('_global' if is_global else ''), input)
#         s=Template(
# '''layer {
#   name: "$name"
#   type: "SumNormalization"
#   bottom: "$input"
#   top: "$output"
#   l2n_param { eps: $eps %s }
# }
# '''%('is_global: false axis: $axis' if not is_global else '')
#         )
#         self.fp.write(s.substitute(locals()))
#         return self

#########################################################

class AggregationOp(BaseBuilder):

    def reduce(
            self,
            input,
            output,
            axis,
            reduce_op='SUM',
            loss_weight=0,
            coeff=1.0,
            name=''
    ):
        shape = self.shape_of(input)
        assert(axis<len(shape))
        shape = shape[:axis]
        assert(self.register_new_blob(output, shape))
        if name=='':
            name=input+'_'+reduce_op

        if loss_weight!=0:
            loss_weight_str = '\n  loss_weight: $loss_weight'
        else:
            loss_weight_str = ''

        s=Template(
'''layer {
  bottom: "${input}" top: "${output}"
  name: "${name}" type: "Reduction"
  reduction_param { axis: $axis coeff: $coeff operation: $reduce_op } %s
}
''' % loss_weight_str
        )
        self.fp.write(s.substitute(locals()))
        return self

    def pool(
            self,
            input,
            output,
            kernel_h,
            kernel_w,
            stride_h=1,
            stride_w=1,
            pool_op='MAX',
            name=''
    ):
        shape = self.shape_of(input)
        assert(len(shape)==4)
        shape[-2] = int((shape[-2]-kernel_h)/stride_h)+1
        shape[-1] = int((shape[-1]-kernel_w)/stride_w)+1
        assert(self.register_new_blob(output, shape))
        if name=='':
            name= pool_op + '['+input+']'

        stride_str = ''
        if stride_h!=1:
            stride_str+='stride_h:{}'.format(stride_h)
        if stride_w!=1:
            if stride_str:
                stride_str+=', '
            stride_str+='stride_w:{}'.format(stride_w)

        s=Template(
'''layer {
  name: "$name" type: "Pooling"
  pooling_param { pool: $pool_op kernel_h:$kernel_h kernel_w:$kernel_w $stride_str }
  bottom: "$input"
  top: "$output"
}
''')
        self.fp.write(s.substitute(locals()))
        return self

    def global_average_pooling(
            self,
            input,
            output,
            name=''
    ):
        shape = self.shape_of(input)
        assert(len(shape)==4)
        shape[-2:]=1
        assert(self.register_new_blob(output, shape))
        if name=='':
            name='global_ave_pool[{:s}]'.format(input)

        s=Template(
'''layer {
  bottom: "$input" top: "$output" name: "$name" type: "Pooling"
  pooling_param { pool: AVE global_pooling: true }
}
''')
        self.fp.write(s.substitute(locals()))
        return self

#########################################################

class EltOp(BaseBuilder):

    def eltprod(
            self,
            inputs,
            output,
            name=''
    ):
        assert(isinstance(inputs,list))
        shapes = [self.shape_of(it) for it in inputs]
        assert(all([shapes[0]==it for it in shapes]))
        assert(self.register_new_blob(output, shapes[0]))
        if name=='':
            name = '_'.join(inputs)
            name = 'eltprod_'+name

        bottoms_str='\n'.join(['  bottom: "{}"'.format(r) for r in inputs])
        s=Template(
'''layer {
  name: "$name" type: "Eltwise"
$bottoms_str
  top: "$output"
  eltwise_param { operation: PROD }
}
'''
        )
        self.fp.write(s.substitute(locals()))
        return self

    def eltsub(
            self,
            input_A,
            input_B,
            output,
            name=''
    ):
        shape = self.shape_of(input_A)
        assert(shape==self.shape_of(input_B))
        assert(self.register_new_blob(output, shape))
        if name=='':
            name = 'eltsub_'+input_A+'_'+input_B

        s=Template(
'''layer {
  name: "$name" type: "Eltwise"
  bottom: "$input_A"
  bottom: "$input_B"
  top:    "$output"
  eltwise_param { operation: SUM coeff: 1 coeff: -1 }
}
'''
        )
        self.fp.write(s.substitute(locals()))
        return self

    def eltsum(
            self,
            inputs,
            output,
            coeffs=None,
            loss_weight=0,
            name=''
    ):
        assert(isinstance(inputs,list))
        shapes = [self.shape_of(it) for it in inputs]
        assert(all([shapes[0]==it for it in shapes]))
        assert(self.register_new_blob(output, shapes[0]))
        if name=='':
            name = 'eltsum_'+'_'.join(inputs)

        bottoms_str= '\n'.join(['  bottom: "%s"'%r for r in inputs])
        if coeffs is None:
            coeffs = [1 for r in inputs]
        else:
            assert(len(coeffs)==len(inputs))
        coeffs_str = ' '.join(['coeff: '+str(r) for r in coeffs])

        if loss_weight!=0:
            loss_str='\n  loss_weight: {:f}'.format(loss_weight)
        else:
            loss_str=''

        s=Template(
'''layer {
  name: "$name" type: "Eltwise"
$bottoms_str
  top: "$output"
  eltwise_param { operation: SUM $coeffs_str } $loss_str
}
'''
        )
        self.fp.write(s.substitute(locals()))
        return self

    def pow(
            self,
            input,
            output,
            power=1,
            scale=1,
            shift=0,
            loss_weight=0,
            name=''
    ):
        assert(self.register_new_blob(output, self.shape_of(input)))
        if name=='':
            name = 'pow'+str(power)+'_'+input

        if loss_weight!=0:
            loss_str='\n  loss_weight: {:f}'.format(loss_weight)
        else:
            loss_str=''

        s=Template(
'''layer {
  bottom: "${input}" top: "${output}" name: "${name}" type: "Power" power_param { power: $power scale: $scale shift: $shift }$loss_str
}
'''
        )
        self.fp.write(s.substitute(locals()))
        return self

    def tile(
            self,
            input,
            output,
            axis=1,
            tiles=1,
            name=''
    ):
        shape = self.shape_of(input)
        shape[axis] *= tiles
        assert(self.register_new_blob(output, shape))
        if name=='':
            name = 'tile_'+input

        s=Template(
'''layer {
  name: "$name" type: "Tile"
  bottom: "$input" top: "$output"
  tile_param { axis: $axis tiles: $tiles }
}
'''
        )
        self.fp.write(s.substitute(locals()))
        return self

    def abs(
            self,
            input,
            output,
            loss_weight=0,
            name=''
    ):
        assert(self.register_new_blob(output, self.shape_of(input)))
        if name=='':
            name='|'+input+'|'

        if loss_weight!=0:
            loss_weight_str = ' loss_weight: '+str(loss_weight)
        else:
            loss_weight_str = ''

        s=Template(
'''layer { bottom: "$input" top: "$output" name: "$name" type: "AbsVal"$loss_weight_str }
''')
        self.fp.write(s.substitute(locals()))
        return self

    def dropout(
            self,
            input,
            output='',
            dropout_ratio=0.5,
            name=''
    ):
        shape = self.shape_of(input)
        if name=='':
            name=input+'_drop'
        if output=='':
            output = input
        else:
            assert(self.register_new_blob(output, shape))

        s=Template(
'''layer { bottom: "$input" top: "$output" name: "$name" type: "Dropout" dropout_param { dropout_ratio: $dropout_ratio } }
''')
        self.fp.write(s.substitute(locals()))
        return self

    def relu(
            self,
            input,
            output='',
            negative_slope=0,
            name=''
    ):
        shape = self.shape_of(input)
        if output=='':
            output = input
        else:
            assert(self.register_new_blob(output, shape))
        if name=='':
            name=input+'_relu'


        s=Template(
'''layer { bottom: "$input" top: "$output" name: "$name" type: "ReLU" %s }
''' % ('' if negative_slope==0 else 'relu_param { negative_slope: %f }' % (negative_slope)) )
        self.fp.write(s.substitute(locals()))
        return self

    def elu(
            self,
            input,
            output='',
            name=''
    ):
        shape = self.shape_of(input)
        if output=='':
            output = input
        else:
            assert(self.register_new_blob(output, shape))
        if name=='':
            name=input+'_elu'

        s=Template(
'''layer { bottom: "$input" top: "$output" name: "$name" type: "ELU" }
''')
        self.fp.write(s.substitute(locals()))
        return self

    def softmax(
            self,
            input,
            output,
            name='',
            axis=0,
    ):
        assert(self.register_new_blob(output, self.shape_of(input)))
        if name=='':
            name = '{}_softmax'.format(input)

        s=Template(
'''layer {
  name: "$name" type: "Softmax"
  softmax_param { axis: $axis }
  bottom: "$input"
  top: "$output"
}
''')
        self.fp.write(s.substitute(locals()))
        return self

#########################################################

class FinalOp(EltOp, AggregationOp):

    def l1_sparsity_loss(
            self,
            input,
            loss_weight,
            axis=1,
            coeff=1,
            output=''
    ):
        if output=='':
            output=input+'_l1loss_SIMPLE_OUTPUT'
        self.abs(input, input+'_abs')
        self.reduce(input+'_abs', output, axis=axis, loss_weight=loss_weight, coeff=coeff)
        return self

    def softmax_loss(
            self,
            predict,
            label,
            loss_name='loss',
            name='loss_layer',
            axis=1,
            loss_weight=1,
            ignore_label=None,
            phase=''
    ):
        shape = self.shape_of(predict)
        assert(shape[0]==self.shape_of(label)[0])
        assert(self.register_new_blob(loss_name, shape[0]))

        s=Template(
'''layer {
  name: "$name" type: "SoftmaxWithLoss" %s
  bottom: "$predict"
  bottom: "$label"
  top: "$loss_name"
  loss_weight: $loss_weight
  %s %s
}
''' % ('''
  softmax_param { axis: $axis }''' if axis!=1 else '',
       phase,
       ('ignore_label:'+str(ignore_label) if ignore_label is not None else '')))
        s = re.sub('\n\s*\n', '\n', s.substitute(locals())) #remove blank lines
        self.fp.write(s)
        return self

    def accuracy(
            self,
            predict,
            label,
            accu_name='accuracy',
            name='accuracy_layer',
            axis=1,
            ignore_label=None
    ):
        shape = self.shape_of(predict)
        assert(shape[0]==self.shape_of(label)[0])
        assert(self.register_new_blob(accu_name, shape[0]))

        s=Template(
'''layer {
  name: "$name" type: "Accuracy" %s
  bottom: "$predict"
  bottom: "$label"
  top: "$accu_name" %s
}
''' % ('''
  accuracy_param { axis: %d }'''%(axis) if axis!=1 else '',
       ('\n  ignore_label:'+str(ignore_label) if ignore_label is not None else '')))
        self.fp.write(s.substitute(locals()))
        return self