'''
builder.py in caffecup

author  : cfeng
created : 3/8/18 3:12 AM
'''

import os
import sys
import time
from string import Template
from collections import OrderedDict

import numpy as np
import re
from .. import version

__all__ = [
    'draw_net_by_caffe',
    'safe_create_file',
    'BaseBuilder',
    'Template'
]


def draw_net_by_caffe(draw_net, filepath, drawpath='', phase='TRAIN', direction='TB'):
    sys.argv.insert(1, filepath)
    if drawpath=='':
        drawpath = os.path.splitext(filepath)[0] + '.png'
    sys.argv.insert(2, drawpath)
    sys.argv.insert(3, '--rankdir')
    sys.argv.insert(4, direction)
    sys.argv.insert(5, '--phase')
    sys.argv.insert(6, phase)
    sys.argv = sys.argv[:7]
    try:
        draw_net.main()
        # print('[draw_net_by_caffe] done!')
    except:
        print('[draw_net_by_caffe] ignore drawing error.')
        pass


def safe_create_file(fp, add_gen_log=True, check_overwrite=False):
    if isinstance(fp, file):
        if add_gen_log:
            fp.write(
'''#caffecup generated at %s
''' % (time.asctime()))
        return fp, None, None
    assert(isinstance(fp, str))

    file_path = fp

    if check_overwrite and os.path.exists(file_path):
        overwrite = raw_input('overwrite existing {} ? [yes(1)]/no(0)'.format(file_path))
        if overwrite=='0':
            sys.exit(0)

    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        print('created: '+file_dir)
        sys.stdout.flush()

    fp = open(fp, 'w')

    if add_gen_log:
            fp.write(
'''#caffecup(%s) generated at %s
''' % (version.__version__, time.asctime()))

    return fp, file_path, file_dir


class BaseBuilder(object):

    def __init__(self, fp, add_gen_log=True):

        self.fp, self.filepath, self.filedir = safe_create_file(fp, add_gen_log)
        self.blob2shape = OrderedDict()

    def done(self, draw_net=None, phase='TRAIN', direction='TB'):
        self.fp.close()
        if draw_net is not None and hasattr(self, 'filepath'):
            draw_net_by_caffe(draw_net=draw_net, filepath=self.filepath, phase=phase, direction=direction)

    def name(self, name):
        self.fp.write('name: "%s"\n' % name)
        return self

    def comment_bar(self, name=''):
        self.fp.write(
'''###################### %s #####################
''' % (name))
        return self

    def comment(self, msg=''):
        self.fp.write(
'''# %s
''' % (msg))
        return self

    def space(self):
        self.fp.write('\n')
        return self

    def silence(
            self,
            input_name
    ):
        if type(input_name)==str:
            self.shape_of(input_name)
            s=Template('layer { bottom: "$input_name" name: "silence" type: "Silence" }\n')
            self.fp.write(s.substitute(locals()))
        elif type(input_name)==list or type(input_name)==tuple:
            shapes = [self.shape_of(it) for it in input_name]
            s='layer {%s name: "silence" type: "Silence" }\n'%(
                ' '.join(['bottom: "'+r+'"' for r in input_name]))
            self.fp.write(s)
        else:
            raise NotImplementedError()
        return self

    def reshape(
            self,
            input,
            output,
            shape=None, #default reshape to batched-vector; use [1,-1] to reshape to single vector
            name='',
            phase=''
    ):
        if shape is None:
            shape = [0,-1]

        shape_in = self.shape_of(input)
        shape_out = []
        shape_out_cnt = 1
        for ith,si in enumerate(shape):
            if si==0:
                shape_out.append(shape_in[ith])
            else:
                shape_out.append(si)
            if shape_out[-1]>0:
                shape_out_cnt*=shape_out[-1]
        infer_cnt = 0
        for ith,so in enumerate(shape_out):
            if so==-1:
                assert(infer_cnt==0)
                infer_cnt += 1
                shape_out[ith] = np.prod(shape_in)/shape_out_cnt
        assert(self.register_new_blob(output, shape_out))

        if name=='':
            name = '{}_reshape'.format(input)

        s=Template(
'''layer { bottom: "$input" top: "$output" name: "$name" type: "Reshape" reshape_param { shape { %s } } %s }
''' % (' '.join(['dim: {:d}'.format(it) for it in shape]), phase))
        self.fp.write(s.substitute(locals()))
        return self

    def concat(
            self,
            inputs,
            output,
            axis=1,
            name=''
    ):
        assert(isinstance(inputs,list))
        shapes = [self.shape_of(it) for it in inputs]
        shp0=shapes[0]
        ax=(len(shp0)+axis) if axis<0 else axis
        assert(all([shp0[0:ax]==it[0:ax] and shp0[ax+1:]==it[ax+1:] for it in shapes]))
        shape = shapes[0]
        shape[axis] = sum([it[axis] for it in shapes])
        assert(self.register_new_blob(output, shape))
        if name=='':
            name = '^'.join(inputs)

        bottoms_str='\n'.join(['  bottom: "{}"'.format(r) for r in inputs])
        s=Template(
'''layer {
  name: "$name" type: "Concat"
$bottoms_str
  top: "$output"
  concat_param { axis: $axis }
}
''')
        self.fp.write(s.substitute(locals()))
        return self

    def pydata(
            self,
            outputs,
            shapes,
            module,
            layer,
            param_str,
            phase,
            name='',
            check_shape=False
    ):
        if isinstance(outputs, str):
            outputs = [outputs]
            shapes = [shapes]
        assert(isinstance(outputs, list))
        assert(isinstance(outputs[0], str))
        assert(isinstance(shapes, list))
        assert(isinstance(shapes[0], list))
        assert(len(outputs)==len(shapes))
        assert(phase.upper() in ['TRAIN', 'TEST', ''])

        if check_shape:
            for o in outputs:
                self.shape_of(o)
        else:
            for o,s in zip(outputs, shapes):
                self.register_new_blob(o, s)

        if name=='':
            name ='{}_'.format(phase) if phase else ''
            name+='pydata('+','.join(outputs)+')'

        output_str = '\n'.join(['  top:"{}"'.format(it) for it in outputs])
        if param_str:
            param_str = 'param_str: "{}"'.format(param_str.replace('\"','\''))

        phase_str = ''
        if phase:
            phase_str = 'include { phase: %s }' % (phase.upper())

        s=Template(
'''layer {
  name: "$name" type: "Python"
$output_str
  python_param {
    module: "$module"
    layer: "$layer"
    $param_str
  }
  $phase_str
}
'''
        )
        return self.write_no_blankline(s.substitute(locals()))

    def write_no_blankline(self, s):
        self.fp.write(re.sub('\n\s*\n', '\n', s)) #remove blank lines before write
        return self

    def register_new_blob(self, name, shape):
        '''return False if blob already exist'''
        if self.blob2shape.has_key(name):
            return False
        else:
            self.blob2shape[name]=shape
            return True

    def shape_of(self, blob):
        if self.blob2shape.has_key(blob):
            return list(self.blob2shape[blob]) #make a copy
        else:
            raise ValueError()

    def comment_blob_shape(self):
        self.comment_bar('blob shapes')
        for name, shape in self.blob2shape.items():
            self.comment('{}: {}'.format(name, str(shape)))

    def print_blob_shape(self):
        print('blob shapes')
        for name, shape in self.blob2shape.items():
            print('{}: {}'.format(name, str(shape)))