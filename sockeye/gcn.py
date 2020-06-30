"""
Implementation of Graph Convolutional Network.
Trying to follow the structure of rnn_cell.py in the mxnet code.
"""

import mxnet as mx

import sockeye.constants as C
from sockeye.config import Config


import logging
logger = logging.getLogger(__name__)


#def get_gcn(input_dim: int, output_dim: int, 
#            tensor_dim: int, use_gcn_gating: bool, 
#            dropout: float, prefix: str):
def get_gcn(config, prefix):
    gcn = GCNCell(config.input_dim,
                  config.output_dim,
                  config.tensor_dim,
                  add_gate=config.add_gate,
                  dropout=config.dropout,
                  prefix=prefix)
    return gcn
   

class GCNParams(object):
    """Container to hold GCN variables.
    Used for parameter sharing.

    Parameters
    ----------
    prefix : str
        All variables' name created by this container will
        be prepended with prefix.
    """

    def __init__(self, prefix=''):
        self._prefix = prefix
        self._params = {}

    def get(self, name, **kwargs):
        """Get a variable with name or create a new one if missing.

        Parameters
        ----------
        name : str
            name of the variable
        **kwargs :
            more arguments that's passed to symbol.Variable
        """
        name = self._prefix + name
        if name not in self._params:
            self._params[name] = mx.sym.Variable(name, **kwargs)
        return self._params[name]


class GCNConfig(Config):
    """
    GCN configuration.

    :param input_dim: Dimensionality for input vectors.
    :param output_dim: Dimensionality for output vectors.
    :param tensor_dim: Edge label space dimensionality.
    :param activation: Non-linear function used inside the GCN updates.
    :param add_gate: Add edge-wise gating (Marcheggiani & Titov, 2017).
    :param dropout: Add dropout to the output vectors.
    :param residual: Add residual connections between multi-layered GCNs.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 tensor_dim: int,
                 activation: str = 'relu',
                 add_gate: bool = False,
                 dropout: float = 0.0,
                 residual: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tensor_dim = tensor_dim
        self.activation = activation
        self.add_gate = add_gate
        self.dropout = dropout
        self.residual = residual
        
                 
class GCNCell(object):
    """GCN cell
    """
    def __init__(self, input_dim, output_dim, tensor_dim,
                 add_gate=False,
                 prefix='gcn_', params=None, 
                 activation='relu',
                 dropout=0.0):
        #if params is None:
        #    params = GCNParams(prefix)
        #    self._own_params = True
        #else:
        #    self._own_params = False
        self._own_params = True
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._tensor_dim = tensor_dim
        self._add_gate = add_gate        
        self._prefix = prefix
        self._params = params
        self._modified = False
        self.reset()
        self._activation = activation
        self._dropout = dropout
        self._W = [mx.symbol.Variable(self._prefix + str(i) + '_weight',
                                      shape=(input_dim, output_dim))
                                      for i in range(tensor_dim)]
        self._b = [mx.symbol.Variable(self._prefix + str(i) + '_bias',
                                      shape=(output_dim,))
                                      for i in range(tensor_dim)]
        # Gate parameters
        if self._add_gate:
            self._gate_W = [mx.symbol.Variable(self._prefix + str(i) + '_gate_weight',
                                               shape=(input_dim, 1))
                                               for i in range(tensor_dim)]
            self._gate_b = [mx.symbol.Variable(self._prefix + str(i) + '_gate_bias',
                                               shape=(1, 1))
                                               for i in range(tensor_dim)]

    def convolve(self, adj, inputs, seq_len):
        """
        IMPORTANT: when retrieving the original adj matrix for an
        edge label we add one to "i" because the edge ids stored
        in the matrix start at 1. 0 corresponds to lack of edges.
        """
        output_list = []
        for i in range(self._tensor_dim):
            # linear transformation
            Wi = self._W[i]
            bi = self._b[i]            
            output = mx.symbol.dot(inputs, Wi)
            output = mx.symbol.broadcast_add(output, bi)
            # optional edge gating
            if self._add_gate:
                gate_Wi = self._gate_W[i]
                gate_bi = self._gate_b[i]
                gate_val = mx.symbol.dot(inputs, gate_Wi)
                gate_val = mx.symbol.broadcast_add(gate_val, gate_bi)
                gate_val = mx.symbol.Activation(gate_val, act_type='sigmoid')
                output = mx.symbol.broadcast_mul(output, gate_val)
            # convolution
            label_id = i + 1
            mask = mx.symbol.ones_like(adj) * label_id
            adji = (mask == adj)
            #adji = mx.symbol.slice_axis(adj, axis=1, begin=i, end=i+1)
            #adji = mx.symbol.reshape(adji, shape=(-1, seq_len, seq_len))
            output = mx.symbol.batch_dot(adji, output)
            output = mx.symbol.expand_dims(output, axis=1)
            output_list.append(output)
        outputs = mx.symbol.concat(*output_list, dim=1)
        outputs = mx.symbol.sum(outputs, axis=1)
        final_output = mx.symbol.Activation(outputs, act_type=self._activation)
        final_output = mx.symbol.Dropout(final_output, p=self._dropout)
        return final_output

    def reset(self):
        pass
