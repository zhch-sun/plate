# pylint:skip-file
import sys
#sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
import resnet
from convnet import convstage, convstage_withpool, convstage_withshortcut, convstage_without_bn
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
SRUParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])
SRUState = namedtuple('SRUState', ['c', 'h'])
def sru(num_hidden, indata, prev_state, param, seqidx, layeridx):
    i2x = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 2,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    x = mx.sym.FullyConnected(data=indata,
                                weight=param.h2h_weight,
                                num_hidden=num_hidden,
                                name="t%d_l%d_h2h" % (seqidx, layeridx),
                                no_bias=True)

    slice_gates = mx.sym.SliceChannel(i2x, num_outputs=2,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))

    f_gate = mx.sym.Activation(slice_gates[0],act_type='sigmoid')
    r_gate = mx.sym.Activation(slice_gates[1],act_type='sigmoid')

    next_c = f_gate * prev_state.c + (1 - f_gate) * x
    c_transform = mx.sym.Activation(next_c, act_type='tanh')
    next_h = r_gate * c_transform + (1 - r_gate) * indata
    return SRUState(c=next_c, h=next_h)
def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx):
    """LSTM Cell symbol"""
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    #next_c = mx.sym.element_mask(next_c, mask)
    #next_h = mx.sym.element_mask(next_h, mask)
    return LSTMState(c=next_c, h=next_h)

def lenet(data):
    conv1 = mx.symbol.Convolution(name='conv1',data=data, kernel=(3,3), num_filter=64,pad=(1,1))
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    conv2 = mx.symbol.Convolution(name='conv2',data=relu1, kernel=(3,3), num_filter=64,pad=(1,1))
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    bn1 = mx.sym.BatchNorm(name='batchnorm1',data=relu2, fix_gamma=False)
    pool1 = mx.symbol.Pooling(data=bn1, pool_type="max", kernel=(2,2), stride=(2, 2))

    conv3 = mx.symbol.Convolution(name='conv3',data=pool1, kernel=(3,3), num_filter=128,pad=(1,1))
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(name='conv4',data=relu3, kernel=(3,3), num_filter=128,pad=(1,1))
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    bn2 = mx.sym.BatchNorm(name='batchnorm2',data=relu4, fix_gamma=False)
    pool2 = mx.symbol.Pooling(data=bn2, pool_type="max", kernel=(2,2), stride=(2, 2))

    conv5 = mx.symbol.Convolution(name='conv5',data=pool2, kernel=(3,3), num_filter=256,pad=(1,1))
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    conv6 = mx.symbol.Convolution(name='conv6',data=relu5, kernel=(3,3), num_filter=256,pad=(1,1))
    relu6 = mx.symbol.Activation(data=conv6, act_type="relu")
    bn3 = mx.sym.BatchNorm(data=relu6, fix_gamma=False)
    pool3 = mx.symbol.Pooling(data=relu6, pool_type="max", kernel=(2,2), stride=(2, 2))
    conv7 = mx.symbol.Convolution(name='conv7',data=pool3, kernel=(1,1), num_filter=512,pad=(0,0))
    return conv7
def sru_unroll(num_lstm_layer, seq_len,
                num_hidden, num_label,dropout=0):
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert (len(last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    conv = lenet(data)
    column_features = mx.sym.SliceChannel(data=data, num_outputs=seq_len, axis=3, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        for i in range(num_lstm_layer):
            next_state = sru(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i)
            hidden = next_state.h
            last_states[i] = next_state
        if dropout > 0:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=80)
    print (pred.infer_shape(data=(32, 3, 32, 320)))
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm = mx.sym.WarpCTC(data=pred, label=label, label_length=num_label, input_length=seq_len)
    return sm
def lstm_unroll(num_lstm_layer, seq_len,
                num_hidden, num_label,dropout=0):
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                      h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    conv=lenet(data)
    column_features = mx.sym.SliceChannel(data=data, num_outputs=seq_len,axis=3, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden =mx.sym.Flatten(data=column_features[seqidx])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i)
            hidden = next_state.h
            last_states[i] = next_state
        if dropout > 0:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=80)
    print (pred.infer_shape(data=(32,3,32,320)))
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data = label, dtype = 'int32')
    sm = mx.sym.WarpCTC(data=pred, label=label, label_length = num_label, input_length = seq_len)
    return sm


def bi_sru_unroll(seq_len,
                   num_hidden, num_label, dropout=0, istrain=True):
    last_states = []
    last_states.append(SRUState(c=mx.sym.Variable("l0_init_c"), h=mx.sym.Variable("l0_init_h")))
    last_states.append(SRUState(c=mx.sym.Variable("l1_init_c"), h=mx.sym.Variable("l1_init_h")))
    forward_param = SRUParam(i2h_weight=mx.sym.Variable("l0_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l0_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l0_h2h_weight")
                              )
    backward_param = SRUParam(i2h_weight=mx.sym.Variable("l1_i2h_weight"),
                               i2h_bias=mx.sym.Variable("l1_i2h_bias"),
                               h2h_weight=mx.sym.Variable("l1_h2h_weight")
                               )
    weight = mx.sym.Variable('fc_weight')
    bias = mx.sym.Variable('fc_bias')

    assert (len(last_states) == 2)

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    from importlib import import_module
    resnet = import_module('resnet')
    conv = resnet.get_symbol(2, 50, '3,32,' + str(seq_len * 8))

#    print ('seq_len : ', seq_len)
#    data_shape = {'data':(400,3,32,160)}
#    arg_shape,_,_ = conv.infer_shape(**data_shape)
#    print(conv.infer_shape(**data_shape))
    column_features = mx.sym.SliceChannel(data=conv, num_outputs=seq_len, axis=3, squeeze_axis=1)
    
    hidden_all = []
    forward_hidden = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden = mx.sym.FullyConnected(hidden, weight=weight, bias=bias, num_hidden=num_hidden)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        next_state = sru(num_hidden, indata=hidden,
                          prev_state=last_states[0],
                          param=forward_param,
                          seqidx=seqidx, layeridx=0)
        hidden = next_state.h
        last_states[0] = next_state
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)

        forward_hidden.append(hidden)

    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        hidden = mx.sym.Flatten(data=column_features[k])
        hidden = mx.sym.FullyConnected(hidden, weight=weight, bias=bias, num_hidden=num_hidden)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        next_state = sru(num_hidden, indata=hidden,
                          prev_state=last_states[1],
                          param=backward_param,
                          seqidx=k, layeridx=1)
        hidden = next_state.h
        last_states[1] = next_state
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        backward_hidden.insert(0, hidden)

    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim=1))
    # mx.symbol.contrib.ctc_loss()
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(name='fc1', data=hidden_concat, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    # does Warp-CTC support bucketing?
    sm = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    # mx.viz.plot_network(sm).view()
    # shape = {"data" : (30,512,32,4)}
    # mx.viz.plot_network(pred,shape=shape).view()
    sm1 = mx.sym.softmax(data=pred, axis=1)
#    shape = {"data": (30, 3, 32, 128)}
#    shape = {"data": (30, 3, 32, 160)}
#    
#    for i in range(2):
#        shape['l%d_init_c' % i] = (30, num_hidden)
#        #shape['l%d_init_h' % i] = (30, 256)
#    mx.viz.plot_network(pred, shape=shape).view()

    if istrain:
        return sm
    else:
        return sm1
def bi_lstm_unroll(seq_len,
                num_hidden, num_label,dropout=0, is_train=True):
    last_states = []
    last_states.append(LSTMState(c = mx.sym.Variable("l0_init_c"), h = mx.sym.Variable("l0_init_h")))
    last_states.append(LSTMState(c = mx.sym.Variable("l1_init_c"), h = mx.sym.Variable("l1_init_h")))
    forward_param = LSTMParam(i2h_weight=mx.sym.Variable("l0_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l0_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l0_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l0_h2h_bias"))
    backward_param = LSTMParam(i2h_weight=mx.sym.Variable("l1_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l1_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l1_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l1_h2h_bias"))
    assert(len(last_states) == 2)

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    from importlib import import_module
    resnet = import_module('resnet')
    conv=resnet.get_symbol(2, 50, '3,32,'+str(seq_len*8))

    print ('seq_len : ',seq_len)

    column_features = mx.sym.SliceChannel(data=conv, num_outputs=seq_len, axis=3, squeeze_axis=1) ##(seq_len,(bs,C,H))

    hidden_all = []
    forward_hidden = []
    for seqidx in range(seq_len):
        hidden =mx.sym.Flatten(data=column_features[seqidx])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[0],
                          param=forward_param,
                          seqidx=seqidx, layeridx=0)
        hidden = next_state.h
        last_states[0] = next_state
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
  
        forward_hidden.append(hidden)

    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        hidden =mx.sym.Flatten(data=column_features[k])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[1],
                          param=backward_param,
                          seqidx=k, layeridx=1)
        hidden = next_state.h
        last_states[1] = next_state
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        backward_hidden.insert(0, hidden)
        
    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim=1))
    #mx.symbol.contrib.ctc_loss()
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(name='fc1',data=hidden_concat, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data = label, dtype = 'int32')
    #does Warp-CTC support bucketing?
    sm = mx.sym.WarpCTC(name='ctc-loss',data=pred, label=label, label_length = num_label, input_length = seq_len)
    #mx.viz.plot_network(sm).view()
    # shape = {"data" : (30,512,32,4)}
    # mx.viz.plot_network(pred,shape=shape).view()
    sm1 = mx.sym.softmax(data=pred, axis=1)
    shape = {"data" : (30, 3, 32, 128)}
    '''
    for i in range(2):
        shape['l%d_init_c' % i] = (30,256)
        shape['l%d_init_h' % i] = (30,256)
    mx.viz.plot_network(pred,shape=shape).view()
    '''
    if is_train:
        return sm
    else:
        return sm1


def small_bi_lstm_unroll(seq_len,
                   num_hidden, num_label, dropout=0.75, kernel_size = 5):
    last_states = []
    last_states.append(LSTMState(c=mx.sym.Variable("l0_init_c"), h=mx.sym.Variable("l0_init_h")))
    last_states.append(LSTMState(c=mx.sym.Variable("l1_init_c"), h=mx.sym.Variable("l1_init_h")))
    forward_param = LSTMParam(i2h_weight=mx.sym.Variable("l0_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l0_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l0_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l0_h2h_bias"))
    backward_param = LSTMParam(i2h_weight=mx.sym.Variable("l1_i2h_weight"),
                               i2h_bias=mx.sym.Variable("l1_i2h_bias"),
                               h2h_weight=mx.sym.Variable("l1_h2h_weight"),
                               h2h_bias=mx.sym.Variable("l1_h2h_bias"))
    assert (len(last_states) == 2)
    start_states = last_states
    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    from importlib import import_module
    resnet = import_module('resnet')
    conv = resnet.get_symbol(2, 18, '3,32,' + str(seq_len * 8))

    print ('seq_len : ', seq_len)
    column_features = mx.sym.SliceChannel(data=conv, num_outputs=seq_len, axis=3, squeeze_axis=1)
    hidden_all = []
    forward_hidden = []
    backward_hidden = []
    for strdidx in range(seq_len/(kernel_size-1)):
        last_states = start_states
        if(strdidx == 0):
            num_loop = kernel_size - 1
        else:
            num_loop = kernel_size

        for small_seqidx in range(num_loop):
            if strdidx != 0:
                small_seqidx = small_seqidx - 1
            seqidx = strdidx*(kernel_size-1)+ small_seqidx
            hidden = mx.sym.Flatten(data=column_features[seqidx])
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[0],
                              param=forward_param,
                              seqidx=seqidx, layeridx=0)
            hidden = next_state.h
            last_states[0] = next_state
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)
            if(small_seqidx != -1):
                forward_hidden.append(hidden)

        for small_seqidx in range(num_loop):
            if strdidx != 0:
                small_seqidx = small_seqidx - 1
            seqidx = strdidx * (kernel_size - 1) + small_seqidx
            k = seq_len - seqidx - 1
            hidden = mx.sym.Flatten(data=column_features[k])
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[1],
                              param=backward_param,
                              seqidx=k, layeridx=1)
            hidden = next_state.h
            last_states[1] = next_state
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)
            if (small_seqidx != -1):
                backward_hidden.insert(0, hidden)

    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim=1))
    # mx.symbol.contrib.ctc_loss()
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(name='fc1', data=hidden_concat, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    # does Warp-CTC support bucketing?
    sm = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    # mx.viz.plot_network(sm).view()
    # shape = {"data" : (30,512,32,4)}
    # mx.viz.plot_network(pred,shape=shape).view()
    return sm

def sep_small_bi_lstm_unroll(seq_len,
                   num_hidden, num_label, dropout=0.75, kernel_size = 4):
    print('creating sep_small_bi_lstm_unroll...')
    last_states = []
    last_states.append(LSTMState(c=mx.sym.Variable("l0_init_c"), h=mx.sym.Variable("l0_init_h")))
    last_states.append(LSTMState(c=mx.sym.Variable("l1_init_c"), h=mx.sym.Variable("l1_init_h")))
    forward_param = LSTMParam(i2h_weight=mx.sym.Variable("l0_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l0_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l0_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l0_h2h_bias"))
    backward_param = LSTMParam(i2h_weight=mx.sym.Variable("l1_i2h_weight"),
                               i2h_bias=mx.sym.Variable("l1_i2h_bias"),
                               h2h_weight=mx.sym.Variable("l1_h2h_weight"),
                               h2h_bias=mx.sym.Variable("l1_h2h_bias"))
    assert (len(last_states) == 2)
    start_states = last_states
    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    from importlib import import_module
    resnet = import_module('resnet')
    conv = resnet.get_symbol(2, 18, '3,32,' + str(seq_len * 8))

    print ('seq_len : ', seq_len)
    column_features = mx.sym.SliceChannel(data=conv, num_outputs=seq_len, axis=3, squeeze_axis=1)
    hidden_all = []
    forward_hidden = []
    backward_hidden = []
    for strdidx in range(seq_len/(kernel_size)):
        last_states = start_states
        num_loop = kernel_size
        for small_seqidx in range(num_loop):
            seqidx = strdidx*(kernel_size)+ small_seqidx
            hidden = mx.sym.Flatten(data=column_features[seqidx])
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[0],
                              param=forward_param,
                              seqidx=seqidx, layeridx=0)
            hidden = next_state.h
            last_states[0] = next_state
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)

            forward_hidden.append(hidden)

        for small_seqidx in range(num_loop):
            seqidx = strdidx * (kernel_size) + small_seqidx
            k = seq_len - seqidx - 1
            hidden = mx.sym.Flatten(data=column_features[k])
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[1],
                              param=backward_param,
                              seqidx=k, layeridx=1)
            hidden = next_state.h
            last_states[1] = next_state
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)
            backward_hidden.insert(0, hidden)

    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim=1))
    # mx.symbol.contrib.ctc_loss()
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(name='fc1', data=hidden_concat, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    # does Warp-CTC support bucketing?
    sm = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    # mx.viz.plot_network(sm).view()
    # shape = {"data" : (30,512,32,4)}
    # mx.viz.plot_network(pred,shape=shape).view()
    return sm


def conv(seq_len, num_label,dropout=0):
    from importlib import import_module
    label = mx.sym.Variable('label')
    resnet = import_module('resnet')
    input = resnet.get_symbol(2, 18, '3,32,' + str(seq_len * 8))
    column_features = mx.sym.SliceChannel(data=input, num_outputs=seq_len, axis=3, squeeze_axis=1)
    hidden_all = []
    for seqidx in range(seq_len):
        hidden =mx.sym.Flatten(data=column_features[seqidx])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    '''
    swap = mx.sym.swapaxes(input,dim1=1,dim2=3)
    flat = mx.sym.flatten(swap)
    data = mx.sym.Variable('data')
    '''
    #conv = mx.sym.Convolution(data=flat, num_filter=1, kernel=(32,3), stride=(1,1), pad=(0,1),no_bias=True, workspace=256, name='final_conv')
    pred = mx.sym.FullyConnected(name='fc1', data=hidden_concat, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)

    #shape = {"data" : (30,512,32,4)}
    #mx.viz.plot_network(pred,shape=shape).view()
    return sm2

def conv2(seq_len, num_label,dropout=0):
    from importlib import import_module
    label = mx.sym.Variable('label')
    resnet = import_module('resnet')
    input = resnet.get_symbol(2, 50, '3,32,' + str(seq_len * 8))
    conv = mx.sym.Convolution(data=input, num_filter=96, kernel=(4, 3), stride=(1, 1), pad=(0, 1), no_bias=True,
                              workspace=256, name='final_conv')
    column_features = mx.sym.SliceChannel(data=conv, num_outputs=seq_len, axis=3, squeeze_axis=1)
    hidden_all = []
    for seqidx in range(seq_len):
        hidden =mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = hidden_concat
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    return sm2
def Jugg_net(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    #num_hidden = int(num_hidden*net_size)
    from importlib import import_module
    workspace = 256
    label = mx.sym.Variable('label')
    power_dilation = 3
    middle_dilation = 2
    speed_dialation = 0
    resnet = import_module('resnet')
    conv = resnet.get_symbol(2, 18, '3,32,' + str(seq_len * 8), net_size=net_size)
    pad = int((5 + (power_dilation-1)*4)-1)/2
    power = mx.sym.Convolution(dilate=(1, power_dilation), data=conv, num_filter=num_hidden, kernel=(5, 5), stride=(1, 1), pad=(2,pad),
                               no_bias=True, workspace=workspace, name='power_conv')
    pad = int((5 + (middle_dilation - 1) * 4) - 1) / 2
    middle = mx.sym.Convolution(dilate=(1, middle_dilation), data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 2),
                               no_bias=True, workspace=workspace, name='middle_conv')
    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')
    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)

    speed = mx.sym.FullyConnected(data=speed,num_hidden=num_hidden,name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle,num_hidden=num_hidden,name='middle_fc')
    power = mx.sym.FullyConnected(data=power,num_hidden=num_hidden,name='power_fc')
    feature = speed+middle
    feature = mx.sym.Dropout(data=feature,p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context,num_hidden=num_hidden*4, name= 'gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm = mx.sym.softmax(data=pred, axis=1)
    shape = {"data" : (30, 3, 32, 128)}
    #mx.viz.plot_network(pred,shape=shape).view()
    if is_train:
        return sm2
    else:
        return sm

def Jugg_net_for_doublerows(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    #num_hidden = int(num_hidden*net_size)
    from importlib import import_module
    workspace = 256
    label = mx.sym.Variable('label')
    labels = mx.sym.split(data=label, axis=1, num_outputs=2)
    power_dilation = 3
    middle_dilation = 2
    speed_dialation = 0
    resnet = import_module('resnet')
    conv = resnet.get_symbol(2, 18, '3,32,' + str(seq_len * 8), net_size=net_size)
    pad = int((5 + (power_dilation-1)*4)-1)/2
    power = mx.sym.Convolution(dilate=(1, power_dilation), data=conv, num_filter=num_hidden, kernel=(5, 5), stride=(1, 1), pad=(2,pad),
                               no_bias=True, workspace=workspace, name='power_conv')
    pad = int((5 + (middle_dilation - 1) * 4) - 1) / 2
    middle = mx.sym.Convolution(dilate=(1, middle_dilation), data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 2),
                               no_bias=True, workspace=workspace, name='middle_conv')
    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')
    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)

    speed = mx.sym.FullyConnected(data=speed,num_hidden=num_hidden,name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle,num_hidden=num_hidden,name='middle_fc')
    power = mx.sym.FullyConnected(data=power,num_hidden=num_hidden,name='power_fc')
    feature = speed+middle
    feature = mx.sym.Dropout(data=feature,p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context,num_hidden=num_hidden*4, name= 'gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label0 = mx.sym.Reshape(data=labels[0], shape=(-1,))
    label0 = mx.sym.Cast(data=label0, dtype='int32')
    sm1 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label0, label_length=num_label, input_length=seq_len)
    sm2 = mx.sym.softmax(data=pred, axis=1)
    shape = {"data" : (30, 3, 32, 128)}
    #mx.viz.plot_network(pred,shape=shape).view()

    speed = mx.sym.FullyConnected(data=speed, num_hidden=num_hidden, name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle, num_hidden=num_hidden, name='middle_fc')
    power = mx.sym.FullyConnected(data=power, num_hidden=num_hidden, name='power_fc')
    feature = speed + middle
    feature = mx.sym.Dropout(data=feature, p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context, num_hidden=num_hidden * 4, name='gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label1 = mx.sym.Reshape(data=labels[1], shape=(-1,))
    label1 = mx.sym.Cast(data=label1, dtype='int32')
    sm3 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label1, label_length=num_label, input_length=seq_len)
    sm4 = mx.sym.softmax(data=pred, axis=1)

    if is_train:
        return (sm1, sm3)
    else:
        return (sm2, sm4)

def Jugg_net_1(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    power_dilation = 3
    middle_dilation = 2
    bn_mom = 0.9
    workspace = 256
    data = mx.sym.Variable(name='data')

    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (5, 5), (3, 3)]
    num_filterlist = [16, 32, 32, 64, 64, 64, 64, 64, 128, 128]
    padlist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]

    for i in range(len(num_filterlist)):
        num_filterlist[i] = num_filterlist[i] * net_size
    conv = convstage(data, stagename, len(kernellist), num_filterlist, workspace, bn_mom, kernellist, stridelist,
                     padlist)
    pad = int((5 + (power_dilation - 1) * 4) - 1) / 2
    power = mx.sym.Convolution(dilate=(1, power_dilation), data=conv, num_filter=num_hidden, kernel=(5, 5),
                               stride=(1, 1), pad=(2, pad),
                               no_bias=True, workspace=workspace, name='power_conv')

    middle = mx.sym.Convolution(dilate=(1, middle_dilation), data=conv, num_filter=num_hidden, kernel=(3, 3),
                                stride=(1, 1), pad=(1, 2),
                                no_bias=True, workspace=workspace, name='middle_conv')

    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')

    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)

    speed = mx.sym.FullyConnected(data=speed, num_hidden=num_hidden, name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle, num_hidden=num_hidden, name='middle_fc')
    power = mx.sym.FullyConnected(data=power, num_hidden=num_hidden, name='power_fc')
    feature = speed + middle
    feature = mx.sym.Dropout(data=feature, p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context, num_hidden=num_hidden * 4, name='gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm = mx.sym.softmax(data=pred, axis=1)
    #shape = {"data" : (30, 3, 32, 128)}
    #mx.viz.plot_network(pred,shape=shape).view()
    if is_train:
        return sm2
    else:
        return sm

def Jugg_Huskar_net(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    power_dilation = 3
    bn_mom = 0.9
    workspace = 256
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (5, 5), (3, 3)]
    num_filterlist = [16, 32, 32, 64, 64, 64, 64, 64, 128, 128]
    padlist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]

    for i in range(len(num_filterlist)):
        num_filterlist[i] = num_filterlist[i] * net_size
    conv = convstage_without_bn(data, stagename, len(kernellist), num_filterlist, workspace, bn_mom, kernellist,
                                stridelist,
                                padlist)
    pad = int((5 + (power_dilation - 1) * 4) - 1) / 2
    power = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(5, 13),
                               stride=(1, 1), pad=(2, pad),
                               no_bias=True, workspace=workspace, name='power_conv')

    middle = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 5),
                                stride=(1, 1), pad=(1, 2),
                                no_bias=True, workspace=workspace, name='middle_conv')

    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')

    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)

    speed = mx.sym.FullyConnected(data=speed, num_hidden=num_hidden, name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle, num_hidden=num_hidden, name='middle_fc')
    power = mx.sym.FullyConnected(data=power, num_hidden=num_hidden, name='power_fc')
    feature = speed + middle
    feature = mx.sym.Dropout(data=feature, p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context, num_hidden=num_hidden * 4, name='gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm = mx.sym.softmax(data=pred, axis=1)
    if is_train:
        return sm2
    else:
        return sm

def Jugg_net_1_withoutbn(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    power_dilation = 3
    middle_dilation = 2
    bn_mom = 0.9
    workspace = 256
    data = mx.sym.Variable(name='data')

    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (5, 5), (3, 3)]
    num_filterlist = [16, 32, 32, 64, 64, 64, 64, 64, 128, 128]
    padlist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]

    for i in range(len(num_filterlist)):
        num_filterlist[i] = num_filterlist[i] * net_size
    conv = convstage_without_bn(data, stagename, len(kernellist), num_filterlist, workspace, bn_mom, kernellist, stridelist,
                     padlist)
    pad = int((5 + (power_dilation - 1) * 4) - 1) / 2
    power = mx.sym.Convolution(dilate=(1, power_dilation), data=conv, num_filter=num_hidden, kernel=(5, 5),
                               stride=(1, 1), pad=(2, pad),
                               no_bias=True, workspace=workspace, name='power_conv')

    middle = mx.sym.Convolution(dilate=(1, middle_dilation), data=conv, num_filter=num_hidden, kernel=(3, 3),
                                stride=(1, 1), pad=(1, 2),
                                no_bias=True, workspace=workspace, name='middle_conv')

    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')

    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)

    speed = mx.sym.FullyConnected(data=speed, num_hidden=num_hidden, name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle, num_hidden=num_hidden, name='middle_fc')
    power = mx.sym.FullyConnected(data=power, num_hidden=num_hidden, name='power_fc')
    feature = speed + middle
    feature = mx.sym.Dropout(data=feature, p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context, num_hidden=num_hidden * 4, name='gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm = mx.sym.softmax(data=pred, axis=1)
    #shape = {"data" : (30, 3, 32, 128)}
    #mx.viz.plot_network(pred,shape=shape).view()
    if is_train:
        return sm2
    else:
        return sm

def Jugg_net_1_NNIE(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    power_dilation = 3
    middle_dilation = 2
    bn_mom = 0.9
    workspace = 256
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (5, 5), (3, 3)]
    num_filterlist = [16, 32, 32, 64, 64, 64, 64, 64, 128, 128]

    padlist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]

    for i in range(len(num_filterlist)):
        num_filterlist[i] = int(num_filterlist[i] * net_size)
    conv = convstage(data, stagename, len(kernellist), num_filterlist, workspace, bn_mom, kernellist, stridelist,
                     padlist)
    pad = int((5 + (power_dilation - 1) * 4) - 1) / 2
    power = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(5, 13),
                               stride=(1, 1), pad=(2, pad),
                               no_bias=True, workspace=workspace, name='power_conv')

    middle = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 5),
                                stride=(1, 1), pad=(1, 2),
                                no_bias=True, workspace=workspace, name='middle_conv')

    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')

    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)

    speed = mx.sym.FullyConnected(data=speed, num_hidden=num_hidden, name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle, num_hidden=num_hidden, name='middle_fc')
    power = mx.sym.FullyConnected(data=power, num_hidden=num_hidden, name='power_fc')
    feature = speed + middle
    feature = mx.sym.Dropout(data=feature, p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context, num_hidden=num_hidden * 4, name='gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm = mx.sym.softmax(data=pred, axis=1)
    if is_train:
        return sm2
    else:
        return sm


def Jugg_net_1_NNIE_middle(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    power_dilation = 3
    middle_dilation = 2
    bn_mom = 0.9
    workspace = 256
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (5, 5), (3, 3)]
    num_filterlist = [16, 32, 32, 64, 64, 64, 64, 64, 128, 128]
    padlist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]

    for i in range(len(num_filterlist)):
        num_filterlist[i] = num_filterlist[i] * net_size
    conv = convstage_without_bn(data, stagename, len(kernellist), num_filterlist, workspace, bn_mom, kernellist, stridelist,
                     padlist)
    pad = int((5 + (power_dilation - 1) * 4) - 1) / 2
    power = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(5, 13),
                               stride=(1, 1), pad=(2, pad),
                               no_bias=True, workspace=workspace, name='power_conv')

    middle = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 5),
                                stride=(1, 1), pad=(1, 2),
                                no_bias=True, workspace=workspace, name='middle_conv')

    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')

    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)

    ret = mx.sym.Group([speed, middle, power])
    return ret

def Jugg_net_1_TRT(seq_len,
                   num_hidden, num_label, batchisize, dropout=0.75, is_train=True, net_size=1):
    power_dilation = 3
    bn_mom = 0.9
    workspace = 256
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (5, 5), (3, 3)]
    num_filterlist = [16, 32, 32, 64, 64, 64, 64, 64, 128, 128]
    padlist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]

    for i in range(len(num_filterlist)):
        num_filterlist[i] = num_filterlist[i] * net_size
    conv = convstage_without_bn(data, stagename, len(kernellist), num_filterlist, workspace, bn_mom, kernellist, stridelist,
                     padlist)
    pad = int((5 + (power_dilation - 1) * 4) - 1) / 2
    power = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(5, 13),
                               stride=(1, 1), pad=(2, pad),
                               no_bias=True, workspace=workspace, name='power_conv')

    middle = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 5),
                                stride=(1, 1), pad=(1, 2),
                                no_bias=True, workspace=workspace, name='middle_conv')

    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')

    power = mx.sym.reshape(data=power, shape=(batchsize*seq_len, 0, 0, 1))
    power = mx.sym.reshape(data=power, shape=(batchsize*seq_len, 0, 0, 1))
    power = mx.sym.flatten(data=power)

    middle = mx.sym.reshape(data=middle, shape=(batchsize*seq_len, 0, 0, 1))
    middle = mx.sym.flatten(data=middle)

    speed = mx.sym.reshape(data=speed, shape=(batchsize*seq_len, 0, 0, 1))
    speed = mx.sym.reshape(data=speed)

    speed = mx.sym.FullyConnected(data=speed, num_hidden=num_hidden, name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle, num_hidden=num_hidden, name='middle_fc')
    power = mx.sym.FullyConnected(data=power, num_hidden=num_hidden, name='power_fc')
    feature = speed + middle
    feature = mx.sym.Dropout(data=feature, p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context, num_hidden=num_hidden * 4, name='gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm = mx.sym.softmax(data=pred, axis=1)
    if is_train:
        return sm2
    else:
        return sm

def Jugg_net_2(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):

    power_dilation = 3
    middle_dilation = 2
    bn_mom = 0.9
    workspace = 256
    data = mx.sym.Variable(name='data')

    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (3, 3), (5, 5), (3, 3), (3, 3), (5, 5), (3, 3), (3, 3)]
    num_filterlist = [16, 32, 32, 32, 64, 64, 64, 128, 128, 128]
    padlist = [(1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1)]

    for i in range(len(num_filterlist)):
        num_filterlist[i] = num_filterlist[i]*net_size
    conv = convstage(data, stagename, len(kernellist), num_filterlist, workspace, bn_mom, kernellist, stridelist,
                       padlist)
    pad = int((5 + (power_dilation-1)*4)-1)/2
    power = mx.sym.Convolution(dilate=(1, power_dilation), data=conv, num_filter=num_hidden, kernel=(5, 5), stride=(1, 1), pad=(2,pad),
                               no_bias=True, workspace=workspace, name='power_conv')

    middle = mx.sym.Convolution(dilate=(1, middle_dilation), data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 2),
                               no_bias=True, workspace=workspace, name='middle_conv')

    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')

    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)

    speed = mx.sym.FullyConnected(data=speed,num_hidden=num_hidden,name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle,num_hidden=num_hidden,name='middle_fc')
    power = mx.sym.FullyConnected(data=power,num_hidden=num_hidden,name='power_fc')
    feature = speed+middle
    feature = mx.sym.Dropout(data=feature,p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context,num_hidden=num_hidden*4, name= 'gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm = mx.sym.softmax(data=pred, axis=1)
    #shape = {"data" : (30, 3, 32, 128)}
    #mx.viz.plot_network(pred,shape=shape).view()
    if is_train:
        return sm2
    else:
        return sm

def Jugg_net_1_split_1(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    power_dilation = 3
    middle_dilation = 2
    bn_mom = 0.9
    workspace = 256
    data = mx.sym.Variable(name='data')

    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (5, 5), (3, 3)]
    num_filterlist = [16, 32, 32, 64, 64, 64, 64, 64, 128, 128]
    padlist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]

    for i in range(len(num_filterlist)):
        num_filterlist[i] = num_filterlist[i] * net_size
    conv = convstage(data, stagename, len(kernellist), num_filterlist, workspace, bn_mom, kernellist, stridelist,
                     padlist)
    pad = int((5 + (power_dilation - 1) * 4) - 1) / 2
    power = mx.sym.Convolution(dilate=(1, power_dilation), data=conv, num_filter=num_hidden, kernel=(5, 5),
                               stride=(1, 1), pad=(2, pad),
                               no_bias=True, workspace=workspace, name='power_conv')

    middle = mx.sym.Convolution(dilate=(1, middle_dilation), data=conv, num_filter=num_hidden, kernel=(3, 3),
                                stride=(1, 1), pad=(1, 2),
                                no_bias=True, workspace=workspace, name='middle_conv')

    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')
    return (power, middle, speed)


def Jugg_net_1_split_2(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    speed = mx.sym.Variable(name='speed')
    power = mx.sym.Variable(name='power')
    middle = mx.sym.Variable(name='middle')
    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)
    return (power, middle, speed)

def Jugg_net_1_split_3(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    label = mx.sym.Variable(name='label')
    speed = mx.sym.Variable(name='speed')
    power = mx.sym.Variable(name='power')
    middle = mx.sym.Variable(name='middle')
    speed = mx.sym.FullyConnected(data=speed, num_hidden=num_hidden, name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle, num_hidden=num_hidden, name='middle_fc')
    power = mx.sym.FullyConnected(data=power, num_hidden=num_hidden, name='power_fc')
    feature = speed + middle
    feature = mx.sym.Dropout(data=feature, p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context, num_hidden=num_hidden * 4, name='gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm = mx.sym.softmax(data=pred, axis=1)
    #shape = {"data" : (30, 3, 32, 128)}
    #mx.viz.plot_network(pred,shape=shape).view()
    if is_train:
        return sm2
    else:
        return sm

def Big_Jugg_net(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    #num_hidden = int(num_hidden*net_size)
    from importlib import import_module
    workspace = 256
    label = mx.sym.Variable('label')
    power_dilation = 3
    middle_dilation = 2
    speed_dialation = 0
    resnet = import_module('resnet')
    conv = resnet.get_symbol(2, 50, '3,32,' + str(seq_len * 8), net_size=net_size)
    pad = int((5 + (power_dilation-1)*4)-1)/2
    power = mx.sym.Convolution(dilate=(1, power_dilation), data=conv, num_filter=num_hidden, kernel=(5, 5), stride=(1, 1), pad=(2,pad),
                               no_bias=True, workspace=workspace, name='power_conv')
    pad = int((5 + (middle_dilation - 1) * 4) - 1) / 2
    middle = mx.sym.Convolution(dilate=(1, middle_dilation), data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 2),
                               no_bias=True, workspace=workspace, name='middle_conv')
    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')
    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)

    speed = mx.sym.FullyConnected(data=speed,num_hidden=num_hidden,name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle,num_hidden=num_hidden,name='middle_fc')
    power = mx.sym.FullyConnected(data=power,num_hidden=num_hidden,name='power_fc')
    feature = speed+middle
    feature = mx.sym.Dropout(data=feature,p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context,num_hidden=num_hidden*4, name= 'gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm = mx.sym.softmax(data=pred, axis=1)
#    shape = {"data" : (30, 3, 32, 128)}
    shape = {"data" : (30, 3, 32, 256)}
    #mx.viz.plot_network(pred,shape=shape).view()
    if is_train:
        return sm2
    else:
        return sm
def Jugg_net_1_with_STF(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    power_dilation = 3
    middle_dilation = 2
    bn_mom = 0.9
    workspace = 256
    data = mx.sym.Variable(name='data')

    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (5, 5), (3, 3)]
    num_filterlist = [16, 32, 32, 64, 64, 64, 64, 64, 128, 128]
    padlist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]

    for i in range(len(num_filterlist)):
        num_filterlist[i] = num_filterlist[i] * net_size
    conv = convstage(data, stagename, len(kernellist), num_filterlist, workspace, bn_mom, kernellist, stridelist,
                     padlist)
    pad = int((5 + (power_dilation - 1) * 4) - 1) / 2
    power = mx.sym.Convolution(dilate=(1, power_dilation), data=conv, num_filter=num_hidden, kernel=(5, 5),
                               stride=(1, 1), pad=(2, pad),
                               no_bias=True, workspace=workspace, name='power_conv')

    middle = mx.sym.Convolution(dilate=(1, middle_dilation), data=conv, num_filter=num_hidden, kernel=(3, 3),
                                stride=(1, 1), pad=(1, 2),
                                no_bias=True, workspace=workspace, name='middle_conv')

    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')

    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)

    speed = mx.sym.FullyConnected(data=speed, num_hidden=num_hidden, name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle, num_hidden=num_hidden, name='middle_fc')
    power = mx.sym.FullyConnected(data=power, num_hidden=num_hidden, name='power_fc')
    feature = speed + middle
    feature = mx.sym.Dropout(data=feature, p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context, num_hidden=num_hidden * 4, name='gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm = mx.sym.softmax(data=pred, axis=1)
    #shape = {"data" : (30, 3, 32, 128)}
    #mx.viz.plot_network(pred,shape=shape).view()
    if is_train:
        return sm2
    else:
        return sm

def Jugg_net_1_doubleline(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    power_dilation = 3
    middle_dilation = 2
    bn_mom = 0.9
    workspace = 256
    data = mx.sym.Variable(name='data')

    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (5, 5), (3, 3)]
    num_filterlist = [16, 32, 32, 64, 64, 64, 64, 64, 128, 128]
    padlist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]

    for i in range(len(num_filterlist)):
        num_filterlist[i] = num_filterlist[i] * net_size
    conv = convstage(data, stagename, len(kernellist), num_filterlist, workspace, bn_mom, kernellist, stridelist,
                     padlist)
    pad = int((5 + (power_dilation - 1) * 4) - 1) // 2
    power = mx.sym.Convolution(dilate=(1, power_dilation), data=conv, num_filter=num_hidden, kernel=(5, 5),
                               stride=(1, 1), pad=(2, pad),
                               no_bias=True, workspace=workspace, name='power_conv')

    middle = mx.sym.Convolution(dilate=(1, middle_dilation), data=conv, num_filter=num_hidden, kernel=(3, 3),
                                stride=(1, 1), pad=(1, 2),
                                no_bias=True, workspace=workspace, name='middle_conv')

    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')

    power = new_slice(power, name="power_new_arc")
    middle = new_slice(middle, name="middle_new_arc")
    speed = new_slice(speed, name="speed_new_arc")

    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)

    speed = mx.sym.FullyConnected(data=speed, num_hidden=num_hidden, name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle, num_hidden=num_hidden, name='middle_fc')
    power = mx.sym.FullyConnected(data=power, num_hidden=num_hidden, name='power_fc')
    feature = speed + middle
    feature = mx.sym.Dropout(data=feature, p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context, num_hidden=num_hidden * 4, name='gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    #sm2 = mx.sym.contrib.ctc_loss(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm = mx.sym.softmax(data=pred, axis=1)
    #shape = {"data" : (30, 3, 32, 128)}
    #mx.viz.plot_network(pred,shape=shape).view()
    if is_train:
        return sm2
    else:
        return sm

if __name__ == '__main__':
    #sm = bi_sru_unroll(16, num_hidden=256, num_label=10, dropout=0.75, istrain=False)
    sm = Jugg_net_1_NNIE(20, num_hidden=256, num_label=10, dropout=0.75, is_train=False, net_size=1)
    sm.save('chrecog_new-symbol.json')

    batchsize = 8
    data = np.random.uniform(0, 1, (batchsize, 3, 32, 192))

    #init_c = [('l%d_init_c'%l, (1, 256)) for l in range(2)]
    #init_h = [('l%d_init_h'%l, (1, 256)) for l in range(2)]
    #init_states = init_c + init_hls
    #data_all = data+[mx.nd.zeros(x[1]) for x in init_states]
    data_all = data
    mod = mx.mod.Module(sm,context=[mx.gpu(7)])
    mod.bind(data_shapes=[('data', (batchsize, 3, 32, 192))])
    mod.init_params()
    data = mx.io.DataBatch(data=[mx.nd.array(data_all)])
    t1 = time.time()
    for i in range(1000):
        y = mod.forward(data)
        mx.nd.waitall()
    t2 = time.time()
    print(t2 - t1) / 1000



def Jugg_net_1_all_bn(seq_len,
                   num_hidden, num_label, dropout=0.75, is_train=True, net_size=1):
    power_dilation = 3
    middle_dilation = 2
    bn_mom = 0.9
    workspace = 256
    data = mx.sym.Variable(name='data')

    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (5, 5), (3, 3)]
    num_filterlist = [16, 32, 32, 64, 64, 64, 64, 64, 128, 128]
    padlist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1)]

    for i in range(len(num_filterlist)):
        num_filterlist[i] = num_filterlist[i] * net_size
    conv = convstage(data, stagename, len(kernellist), num_filterlist, workspace, bn_mom, kernellist, stridelist,
                     padlist)
    pad = int((5 + (power_dilation - 1) * 4) - 1) / 2
    power = mx.sym.Convolution(dilate=(1, power_dilation), data=conv, num_filter=num_hidden, kernel=(5, 5),
                               stride=(1, 1), pad=(2, pad),
                               no_bias=True, workspace=workspace, name='power_conv')
    power = mx.sym.BatchNorm(data=power, fix_gamma=False, eps=2e-5, momentum=0.9, name='power_bn')
    
    middle = mx.sym.Convolution(dilate=(1, middle_dilation), data=conv, num_filter=num_hidden, kernel=(3, 3),
                                stride=(1, 1), pad=(1, 2),
                                no_bias=True, workspace=workspace, name='middle_conv')
    middle = mx.sym.BatchNorm(data=middle, fix_gamma=False, eps=2e-5, momentum=0.9, name='middle_bn')

    speed = mx.sym.Convolution(data=conv, num_filter=num_hidden, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name='speed_conv')
    speed = mx.sym.BatchNorm(data=speed, fix_gamma=False, eps=2e-5, momentum=0.9, name='speed_bn')

    column_features = mx.sym.SliceChannel(data=power, num_outputs=seq_len, axis=3, squeeze_axis=1, name='power_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    power = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=middle, num_outputs=seq_len, axis=3, squeeze_axis=1, name='middle_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    middle = mx.sym.Concat(*hidden_all, dim=0)

    column_features = mx.sym.SliceChannel(data=speed, num_outputs=seq_len, axis=3, squeeze_axis=1, name='speed_slice')
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        hidden_all.append(hidden)
    speed = mx.sym.Concat(*hidden_all, dim=0)

    speed = mx.sym.FullyConnected(data=speed, num_hidden=num_hidden, name='speed_fc')
    middle = mx.sym.FullyConnected(data=middle, num_hidden=num_hidden, name='middle_fc')
    power = mx.sym.FullyConnected(data=power, num_hidden=num_hidden, name='power_fc')
    feature = speed + middle
    feature = mx.sym.Dropout(data=feature, p=dropout)
    energy = mx.sym.Activation(data=feature, act_type='tanh')
    attention = mx.sym.softmax(data=energy, axis=1)
    context = middle * attention
    gates = mx.sym.FullyConnected(data=context, num_hidden=num_hidden * 4, name='gates_fc')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="slice_gates")
    in_gate = mx.sym.Dropout(data=slice_gates[0], p=dropout)
    in_gate = mx.sym.Activation(in_gate, act_type="sigmoid")
    in_transform = mx.sym.Dropout(data=slice_gates[1], p=dropout)
    bn = mx.sym.BatchNorm(data=in_transform, fix_gamma=False, eps=2e-5, momentum=0.9, name='2_bn')
    in_transform = mx.sym.Activation(bn, act_type="tanh")
    forget_gate = mx.sym.Dropout(data=slice_gates[2], p=dropout)
    forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid")
    out_gate = mx.sym.Dropout(data=slice_gates[3], p=dropout)
    out_gate = mx.sym.Activation(out_gate, act_type="sigmoid")
    next_c = (forget_gate * power) + (in_gate * in_transform)
    bn = mx.sym.BatchNorm(data=next_c, fix_gamma=False, eps=2e-5, momentum=0.9, name='final_bn')
    pred = out_gate * mx.sym.Activation(bn, act_type="tanh")
    pred = mx.sym.FullyConnected(name='final_fc', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm = mx.sym.softmax(data=pred, axis=1)
    #shape = {"data" : (30, 3, 32, 128)}
    #mx.viz.plot_network(pred,shape=shape).view()
    if is_train:
        return sm2
    else:
        return sm