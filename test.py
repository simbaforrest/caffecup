'''
test.py in caffecup

author  : cfeng
created : 3/11/18 5:16 AM
'''

import os
import sys
import argparse

import caffecup
sys.path.insert(0, '../caffe/install/python')
import caffecup.viz.draw_net as ccdraw
from string import Template

def global_max_pooling(self, input, input_offset, output, name='GlobalMaxPool'):
    assert(isinstance(self, caffecup.Designer))
    shape = self.shape_of(input)
    shape[0] = 1
    assert(self.register_new_blob(output, shape))

    s = Template(
'''layer {
	name: "$name" type: "GraphPooling"
	graph_pooling_param { mode: MAX	}
	bottom: "$input"
	bottom: "$input_offset"
	propagate_down: true
	propagate_down: false
	top: "$output"
}
'''
    )
    self.fp.write(s.substitute(locals()))
    return self

def main(args):
    cc = caffecup.Designer('./tmp/test_network.ptt')
    cc.name('test_network')
    cc.comment_bar('Data')
    cc.pydata(
        ['X', 'Y', 'n_offset'],
        [[1024,3], [1,], [2,]],
        module='py_graph_net',
        layer='ModelNetGraphDataLayer',
        param_str="{'source':'data/train_test_data/modelNet_train_data.npy', 'batch_size':1, 'modes':[]}",
        phase='TRAIN'
    )
    cc.pydata(
        ['X', 'Y', 'n_offset'],
        [[1024,3], [1,], [2,]],
        module='py_graph_net',
        layer='ModelNetGraphDataLayer',
        param_str="{'source':'data/train_test_data/modelNet_test_data.npy', 'batch_size':1, 'modes':[]}",
        phase='TEST'
    )

    cc.comment_bar('Feature')
    XKrelu = lambda i,o,n: cc.XK(i,o,n).relu(o).space()
    XKrelu('X', 'X1', 64)
    XKrelu('X1','X2', 64)
    XKrelu('X2','X3', 64)
    XKrelu('X3','X4',128)
    cc.XK( 'X4','P',1024).space()
    global_max_pooling(cc,'P','n_offset','F')

    cc.comment_bar('Classifier')
    fcreludrop = lambda i,o,n,d: cc.fc(i,o,n).relu(o).dropout(o,dropout_ratio=d).space()
    fcreludrop('F','F1',512,0.3)
    fcreludrop('F1','F2',256,0.3)
    cc.fc('F2','Yp',40)

    cc.comment_bar('Final')
    cc.softmax_loss('Yp','Y')
    cc.accuracy('Yp','Y')

    cc.comment_blob_shape()
    cc.done(draw_net=ccdraw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)