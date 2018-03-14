"""
Caffe network visualization: draw the NetParameter protobuffer.


.. note::

    This requires pydot>=1.0.2, which is not included in requirements.txt since
    it requires graphviz and other prerequisites outside the scope of the
    Caffe.
"""

from caffe.proto import caffe_pb2

"""
pydot is not supported under python 3 and pydot2 doesn't work properly.
pydotplus works nicely (pip install pydotplus)
"""
try:
    # Try to load pydotplus
    import pydotplus as pydot
except ImportError:
    import pydot

# Internal layer and blob styles.
LAYER_STYLE_DEFAULT = {'shape': 'record',
                       'fillcolor': '#6495ED',
                       'style': 'filled'}
NEURON_LAYER_STYLE = {'shape': 'record',
                      'fillcolor': '#90EE90',
                      'style': 'filled'}
BLOB_STYLE = {'shape': 'octagon',
              'fillcolor': '#E0E0E0',
              'style': 'filled'}
ACTIVATED_BLOB_STYLE = {'fillcolor': '#90EE90'}


def get_pooling_types_dict():
    """Get dictionary mapping pooling type number to type name
    """
    desc = caffe_pb2.PoolingParameter.PoolMethod.DESCRIPTOR
    d = {}
    for k, v in desc.values_by_name.items():
        d[v.number] = k
    return d


def get_edge_label(layer):
    """Define edge label based on layer type.
    """

    if layer.type == 'Data':
        edge_label = 'Batch ' + str(layer.data_param.batch_size)
    elif layer.type == 'Convolution' or layer.type == 'Deconvolution':
        edge_label = str(layer.convolution_param.num_output)
    elif layer.type == 'InnerProduct':
        edge_label = str(layer.inner_product_param.num_output)
    else:
        edge_label = '""'

    return edge_label


def get_layer_label(layer, rankdir):
    """Define node label based on layer type.

    Parameters
    ----------
    layer : ?
    rankdir : {'LR', 'TB', 'BT'}
        Direction of graph layout.

    Returns
    -------
    string :
        A label for the current layer
    """

    if rankdir in ('TB', 'BT'):
        # If graph orientation is vertical, horizontal space is free and
        # vertical space is not; separate words with spaces
        separator = ' '
    else:
        # If graph orientation is horizontal, vertical space is free and
        # horizontal space is not; separate words with newlines
        separator = '\\n'

    if layer.type == 'Convolution' or layer.type == 'Deconvolution':
        # Outer double quotes needed or else colon characters don't parse
        # properly
        node_label = '"%s%s(%s)%skernel size: %d%sstride: %d%spad: %d"' %\
                     (layer.name,
                      separator,
                      layer.type,
                      separator,
                      layer.convolution_param.kernel_size[0] if len(layer.convolution_param.kernel_size) else 1,
                      separator,
                      layer.convolution_param.stride[0] if len(layer.convolution_param.stride) else 1,
                      separator,
                      layer.convolution_param.pad[0] if len(layer.convolution_param.pad) else 0)
    elif layer.type == 'Pooling':
        pooling_types_dict = get_pooling_types_dict()
        kernel_str=stride_str=pad_str=''
        if layer.pooling_param.kernel_size!=0:
            kernel_str = 'kernel size:%d' % layer.pooling_param.kernel_size
        if layer.pooling_param.kernel_h!=0:
            kernel_str = 'kernel(%dx%d)' % (layer.pooling_param.kernel_h, layer.pooling_param.kernel_w)
        if layer.pooling_param.stride!=1:
            stride_str = 'stride:%d' % layer.pooling_param.stride
        if layer.pooling_param.pad!=0:
            pad_str = 'pad:%d' % layer.pooling_param.pad
        node_label = '"%s%s(%s %s)%s\n%s%s%s%s%s"' %\
                     (layer.name,
                      separator,
                      pooling_types_dict[layer.pooling_param.pool],
                      layer.type,
                      separator,
                      kernel_str,
                      separator,
                      stride_str,
                      separator,
                      pad_str)

    else:
        node_label = '"%s%s(%s)"' % (layer.name, separator, layer.type)
    return node_label


def choose_color_by_layertype(layertype):
    """Define colors for nodes based on the layer type.
    """
    color = '#6495ED'  # Default
    if layertype == 'Convolution' or layertype == 'Deconvolution':
        color = '#FF5050'
    elif layertype == 'Pooling':
        color = '#FF9900'
    elif layertype == 'InnerProduct':
        color = '#CC33FF'
    return color


def get_pydot_graph(caffe_net, rankdir, label_edges=True, phase=None, is_simplified=True):
    """Create a data structure which represents the `caffe_net`.

    Parameters
    ----------
    caffe_net : object
    rankdir : {'LR', 'TB', 'BT'}
        Direction of graph layout.
    label_edges : boolean, optional
        Label the edges (default is True).
    phase : {caffe_pb2.Phase.TRAIN, caffe_pb2.Phase.TEST, None} optional
        Include layers from this network phase.  If None, include all layers.
        (the default is None)
    is_simplified: {True, False} optonal
        Whether to ignore inplace layer (the default is True).

    Returns
    -------
    pydot graph object
    """
    pydot_graph = pydot.Dot(caffe_net.name if caffe_net.name else 'Net',
                            graph_type='digraph',
                            rankdir=rankdir)
    pydot_nodes = {}
    pydot_edges = []
    is_blob_inplace = {}
    for layer in caffe_net.layer:
        if phase is not None:
          included = False
          if len(layer.include) == 0:
            included = True
          if len(layer.include) > 0 and len(layer.exclude) > 0:
            raise ValueError('layer ' + layer.name + ' has both include '
                             'and exclude specified.')
          for layer_phase in layer.include:
            included = included or layer_phase.phase == phase
          for layer_phase in layer.exclude:
            included = included and not layer_phase.phase == phase
          if not included:
            continue
        node_label = get_layer_label(layer, rankdir)
        node_name = "%s_%s" % (layer.name, layer.type)
        if (len(layer.bottom) == 1 and len(layer.top) == 1 and
           layer.bottom[0] == layer.top[0]):
            # We have an in-place neuron layer.
            is_inplace = True
            is_blob_inplace[layer.bottom[0]] = True
            if not is_simplified:
                pydot_nodes[node_name] = pydot.Node(node_label,
                                                    **NEURON_LAYER_STYLE)
        else:
            is_inplace = False
            layer_style = LAYER_STYLE_DEFAULT
            layer_style['fillcolor'] = choose_color_by_layertype(layer.type)
            pydot_nodes[node_name] = pydot.Node(node_label, **layer_style)
        if is_simplified and is_inplace:
            def inplace_blob_name():
                if layer.type=='Dropout':
                    return 'Dropout{}'.format(layer.dropout_param.dropout_ratio)
                return layer.type
            bottom_blob = layer.bottom[0]
            blob_key = bottom_blob + '_blob'
            blob_style = dict(BLOB_STYLE)
            blob_style.update(ACTIVATED_BLOB_STYLE)
            assert(pydot_nodes.has_key(blob_key)) #inplace blob must have been created before
            pydot_nodes[blob_key].obj_dict['attributes'] = blob_style
            pydot_nodes[blob_key].set_name(
                pydot_nodes[blob_key].get_name()+'>'+inplace_blob_name()
            )
            continue
        for bottom_blob in layer.bottom:
            blob_key = bottom_blob + '_blob'
            if not pydot_nodes.has_key(blob_key) or (
                    not is_blob_inplace.has_key(bottom_blob) or not is_blob_inplace[bottom_blob]):
                pydot_nodes[blob_key] = pydot.Node('%s' % bottom_blob,
                                                   **BLOB_STYLE)
            edge_label = '""'
            pydot_edges.append({'src': bottom_blob + '_blob',
                                'dst': node_name,
                                'label': edge_label})
        for top_blob in layer.top:
            blob_key = top_blob + '_blob'
            pydot_nodes[blob_key] = pydot.Node('%s' % (top_blob))
            if label_edges:
                edge_label = get_edge_label(layer)
            else:
                edge_label = '""'
            pydot_edges.append({'src': node_name,
                                'dst': top_blob + '_blob',
                                'label': edge_label})
    # Now, add the nodes and edges to the graph.
    for node in pydot_nodes.values():
        pydot_graph.add_node(node)
    for edge in pydot_edges:
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[edge['src']],
                       pydot_nodes[edge['dst']],
                       label=edge['label']))
    return pydot_graph


def draw_net(caffe_net, rankdir, ext='png', phase=None):
    """Draws a caffe net and returns the image string encoded using the given
    extension.

    Parameters
    ----------
    caffe_net : a caffe.proto.caffe_pb2.NetParameter protocol buffer.
    ext : string, optional
        The image extension (the default is 'png').
    phase : {caffe_pb2.Phase.TRAIN, caffe_pb2.Phase.TEST, None} optional
        Include layers from this network phase.  If None, include all layers.
        (the default is None)

    Returns
    -------
    string :
        Postscript representation of the graph.
    """
    return get_pydot_graph(caffe_net, rankdir, phase=phase).create(format=ext)


def draw_net_to_file(caffe_net, filename, rankdir='LR', phase=None):
    """Draws a caffe net, and saves it to file using the format given as the
    file extension. Use '.raw' to output raw text that you can manually feed
    to graphviz to draw graphs.

    Parameters
    ----------
    caffe_net : a caffe.proto.caffe_pb2.NetParameter protocol buffer.
    filename : string
        The path to a file where the networks visualization will be stored.
    rankdir : {'LR', 'TB', 'BT'}
        Direction of graph layout.
    phase : {caffe_pb2.Phase.TRAIN, caffe_pb2.Phase.TEST, None} optional
        Include layers from this network phase.  If None, include all layers.
        (the default is None)
    """
    ext = filename[filename.rfind('.')+1:]
    with open(filename, 'wb') as fid:
        fid.write(draw_net(caffe_net, rankdir, ext, phase))

###############################################################################

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

import caffe
from caffe.proto import caffe_pb2


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_net_proto_file',
                        help='Input network prototxt file')
    parser.add_argument('output_image_file',
                        help='Output image file')
    parser.add_argument('--rankdir',
                        help=('One of TB (top-bottom, i.e., vertical), '
                              'RL (right-left, i.e., horizontal), or another '
                              'valid dot option; see '
                              'http://www.graphviz.org/doc/info/'
                              'attrs.html#k:rankdir'),
                        default='LR')
    parser.add_argument('--phase',
                        help=('Which network phase to draw: can be TRAIN, '
                              'TEST, or ALL.  If ALL, then all layers are drawn '
                              'regardless of phase.'),
                        default="ALL")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(args.input_net_proto_file).read(), net)
    print('Drawing net to %s' % args.output_image_file)
    phase=None
    if args.phase == "TRAIN":
        phase = caffe.TRAIN
    elif args.phase == "TEST":
        phase = caffe.TEST
    elif args.phase != "ALL":
        raise ValueError("Unknown phase: " + args.phase)
    draw_net_to_file(net, args.output_image_file, args.rankdir,
                                phase)


if __name__ == '__main__':
    main()
