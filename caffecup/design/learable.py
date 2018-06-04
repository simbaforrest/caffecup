'''
learable.py in caffecup

author  : cfeng
created : 3/8/18 4:48 AM
'''

from string import Template
import re

from builder import BaseBuilder

__all__ = [
    'filler_uniform',
    'filler_constant',
    'filler_xavier',
    'filler_gaussian',
    'filler_msra',
    'learning_param',
    'Learable'
]

###########################################################

def filler_xavier():
    return 'filler { type: "xavier" }'

def filler_constant(val=0.):
    return 'filler {{ type: "constant" value: {} }}'.format(val)

def filler_uniform(min, max):
    return 'filler {{ type: "uniform" min: {} max: {} }}'.format(min, max)

def filler_gaussian(mean, std):
    return 'filler {{ type: "gaussian" mean: {} std: {} }}'.format(mean, std)

def filler_msra():
    return 'filler { type: "msra" }'

###########################################################

def learning_param(lr_mult=1., decay_mult=1., name=''):
    return 'param {{{} lr_mult: {}, decay_mult: {} }}'.format(
        'name: "%s"'%name if name else '',
        lr_mult, decay_mult
    )

###########################################################

class Learable(BaseBuilder):

    def parameter(
            self,
            output,
            shape,
            filler_str=filler_xavier(),
            param_str='',
            name=''
    ):
        assert(self.register_new_blob(output, shape))
        if name=='':
            name='param_'+output

        s=Template(
'''layer {
  top: "$output" type: "Parameter" name: "$name"
  parameter_param {
    shape { %s }
    $filler_str
  }
  $param_str
}
''' % (' '.join(['dim: {:d}'.format(it) for it in shape]))
        )
        s = re.sub('\n\s*\n', '\n', s.substitute(locals())) #remove blank lines
        self.fp.write(s)
        return self

    def batchnorm(
            self,
            input,
            output='',
            name=''
    ):
        shape = self.shape_of(input)
        if output=='':
            output=input
        else:
            assert(self.register_new_blob(output, shape))
        if name=='':
            name=input+'_bn'

        s=Template(
'''layer { bottom: "$input" top: "$output" name: "$name" type: "BatchNorm" }
''')
        self.fp.write(s.substitute(locals()))
        return self

    def scale(
            self,
            input,
            output='',
            bias_term='true',
            param_str='',
            name=''
    ):
        shape = self.shape_of(input)
        if output=='':
            output=input
        else:
            assert(self.register_new_blob(output, shape))
        if name=='':
            name=input+'_sc'

        s=Template(
'''layer { bottom: "$input" top: "$output" name: "$name" type: "Scale" scale_param { bias_term: $bias_term } }
''')
        self.fp.write(s.substitute(locals()))
        return self

    def inplace_batchnorm_with_scale(
            self,
            input,
    ):
        self.batchnorm(input, input)
        self.scale(input, input)
        return self

    def scalar_scale(
            self,
            input,
            output='',
            filler_str=filler_constant(1.0),
            param_str=learning_param(),
            name=''
    ):
        shape = self.shape_of(input)
        if output=='':
            output=input
        else:
            assert(self.register_new_blob(output, shape))
        if name=='':
            name=input+'_sc'

        s=Template(
'''layer { #scalar_scale
  bottom: "$input" top: "$output" name: "$name" type: "Scale"
  scale_param { num_axes: 0 bias_term: false }
  $filler_str
  $param_str
}
''')
        s = re.sub('\n\s*\n', '\n', s.substitute(locals())) #remove blank lines
        self.fp.write(s)
        return self

    def elm_bias(
            self,
            input,
            output,
            axis = 1,
            filler_str=filler_uniform(-0.5,0.5),
            param_str=learning_param(1.0, 0.0),
            name=''
    ):
        assert(self.register_new_blob(output, self.shape_of(input)))
        if name=='':
            name = '{}_elm_bias'.format(input)

        s=Template(
'''layer {
  name: "$name" type: "Bias"
  bias_param { axis: $axis num_axes: -1 $filler_str }
  $param_str
  bottom: "$input"
  top: "$output"
}
''')
        s = re.sub('\n\s*\n', '\n', s.substitute(locals())) #remove blank lines
        self.fp.write(s)
        return self

    def scalar_bias(
            self,
            input,
            output='',
            filler_str=filler_constant(1.0),
            param_str=learning_param(1.0, 0.0),
            name=''
    ):
        shape = self.shape_of(input)
        if output=='':
            output=input
        else:
            assert(self.register_new_blob(output, shape))
        if name=='':
            name=input+'_bias'

        s=Template(
'''layer { #scalar_bias
  bottom: "$input" top: "$output" name: "$name" type: "Bias"
  bias_param { num_axes: 0 }
  $filler_str
  $param_str
}
''')
        s = re.sub('\n\s*\n', '\n', s.substitute(locals())) #remove blank lines
        self.fp.write(s)
        return self

    def fc(
            self,
            input,
            output,
            num_output,
            axis=1,
            weight_filler_str=filler_xavier(),
            weight_param_str=learning_param(1.0, 1.0),
            bias_term=True,
            bias_filler_str='',
            bias_param_str=learning_param(1.0, 0.0),
            name=''
    ):
        '''
        this is used for classifier
        '''
        shape = self.shape_of(input)
        if axis<0:
            axis += len(shape)
        assert(0<=axis<len(shape))
        shape = shape[:axis+1]
        shape[-1] = num_output
        assert(self.register_new_blob(output, shape))

        if name=='':
            if not hasattr(self, 'n_fc'):
                self.n_fc = 1
            else:
                self.n_fc += 1
            name = 'FC{:d}'.format(self.n_fc)

        axis_str = ''
        if axis!=1:
            axis_str = \
'''
    axis: {:d}'''.format(axis)

        if bias_filler_str:
            bias_filler_str = 'bias_' + bias_filler_str

        if bias_term:
            bias_str=''
            filler_str ='weight_%s %s' % (weight_filler_str, bias_filler_str)
        else:
            bias_str='bias_term: false'
            filler_str ='weight_%s' % (weight_filler_str)
            bias_param_str=''

        s=Template(
'''layer {
  bottom: "$input"
  top: "$output"
  name: "$name" type: "InnerProduct"
  inner_product_param {
    num_output: $num_output
    $axis_str
    $bias_str
    $filler_str
  }
  $weight_param_str
  $bias_param_str
}
'''
        )
        return self.write_no_blankline(s.substitute(locals()))

    def XK(
            self,
            input,
            output,
            num_output,
            axis=1,
            weight_filler_str=filler_xavier(),
            weight_param_str=learning_param(1.0, 1.0),
            bias_term=True,
            bias_filler_str=filler_constant(),
            bias_param_str=learning_param(2.0, 0.0),
            name=''
    ):
        if name=='':
            if not hasattr(self, 'n_XK'):
                self.n_XK = 1
            else:
                self.n_XK += 1
            name='XK{:d}'.format(self.n_XK)

        return self.fc(input, output, num_output, axis=axis,
                       weight_filler_str=weight_filler_str, weight_param_str=weight_param_str,
                       bias_term=bias_term,bias_filler_str=bias_filler_str,bias_param_str=bias_param_str,
                       name=name)

