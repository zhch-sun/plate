import mxnet as mx
import time
import numpy as np
def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def convstage(data, name, num_layer, num_filterlist, workspace, bn_mom, kernellist, stridelist, padlist):
    input = data
    for i in range(num_layer):
        conv = mx.sym.Convolution(data=input, num_filter=num_filterlist[i], kernel=kernellist[i], stride=stridelist[i], pad=padlist[i],
                                   no_bias=True, workspace=workspace, name=name + '_conv'+str(i))
        bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn'+str(i))
        act = mx.sym.Activation(data=bn, act_type='relu', name=name + '_relu'+str(i))
        input = act

    return input

def convstage_with_STN(data, name, num_layer, num_filterlist, workspace, bn_mom, kernellist, stridelist, padlist):
    input = data
    for i in range(num_layer):
        conv = mx.sym.Convolution(data=input, num_filter=num_filterlist[i], kernel=kernellist[i], stride=stridelist[i], pad=padlist[i],
                                   no_bias=True, workspace=workspace, name=name + '_conv'+str(i))
        bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn'+str(i))
        act = mx.sym.Activation(data=bn, act_type='relu', name=name + '_relu'+str(i))
        input = act
        loc = mx.sym.FullyConnected(data=input,num_hidden=6)
        input = mx.sym.SpatialTransformer(data=input, loc=loc, transform_type="affine", name=name + '_stn'+str(i))
    return input

def convstage_without_bn(data, name, num_layer, num_filterlist, workspace, bn_mom, kernellist, stridelist, padlist):
    input = data
    for i in range(num_layer):
        conv = mx.sym.Convolution(data=input, num_filter=num_filterlist[i], kernel=kernellist[i], stride=stridelist[i], pad=padlist[i],
                                    workspace=workspace, name=name + '_conv'+str(i))
        act = mx.sym.Activation(data=conv, act_type='relu', name=name + '_relu'+str(i))
        input = act
    return input

def convstage_withpool(data, name, num_layer, num_filterlist, workspace, bn_mom, kernellist, stridelist, padlist):
    input = data
    for i in range(num_layer):
        conv = mx.sym.Convolution(data=input, num_filter=num_filterlist[i], kernel=kernellist[i], stride=stridelist[i], pad=padlist[i],
                                   no_bias=True, workspace=workspace, name=name + '_conv'+str(i))
        bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn'+str(i))
        act = mx.sym.Activation(data=bn, act_type='relu', name=name + '_relu'+str(i))
        input = act
    output = mx.sym.Pooling(data=input,kernel=(2, 2), pool_type='max', stride=(2, 2))
    return output

def convstage_withshortcut(data, name, num_layer, num_filterlist, workspace, bn_mom, kernellist, stridelist, padlist):
    input = data
    sc = mx.sym.Convolution(data=input, num_filter=num_filterlist[-1], kernel=(1, 1), stride=stridelist[-1],
                              no_bias=True, workspace=workspace, name=name + '_sc')
    for i in range(num_layer):
        conv = mx.sym.Convolution(data=input, num_filter=num_filterlist[i], kernel=kernellist[i], stride=stridelist[i], pad=padlist[i],
                                   no_bias=True, workspace=workspace, name=name + '_conv'+str(i))
        bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn'+str(i))
        act = mx.sym.Activation(data=bn, act_type='relu', name=name + '_relu'+str(i))
        input = act
    return (input+sc)

def convnet(seq_len, num_label, dropout, bn_mom=0.9):
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable('label')
    stage1name = 'Conv1'
    num_filter = 32
    workspace = 256

    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2),
                               no_bias=True, workspace=workspace, name=stage1name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=stage1name + '_relu1')
    conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter , kernel=(5, 5), stride=(1, 1), pad=(2, 2),
                               no_bias=True, workspace=workspace, name=stage1name + '_conv2')
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=stage1name + '_relu2')
    conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2), no_bias=True,
                               workspace=workspace, name=stage1name + '_conv3')
    bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=stage1name + '_relu3')
    conv4 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2), no_bias=True,
                               workspace=workspace, name=stage1name + '_conv4')
    bn4 = mx.sym.BatchNorm(data=conv4, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn4')
    act4 = mx.sym.Activation(data=bn4, act_type='relu', name=stage1name + '_relu4')
    conv5 = mx.sym.Convolution(data=act4, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2),
                               no_bias=True, workspace=workspace, name=stage1name + '_conv5')
    bn5 = mx.sym.BatchNorm(data=conv5, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn5')
    act5 = mx.sym.Activation(data=bn5, act_type='relu', name=stage1name + '_relu5')
    conv6 = mx.sym.Convolution(data=act5, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2), no_bias=True,
                               workspace=workspace, name=stage1name + '_conv6')
    bn6 = mx.sym.BatchNorm(data=conv6, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn6')
    act6 = mx.sym.Activation(data=bn6, act_type='relu', name=stage1name + '_relu6')
    conv7 = mx.sym.Convolution(data=act6, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2), no_bias=True,
                               workspace=workspace, name=stage1name + '_conv7')
    bn7 = mx.sym.BatchNorm(data=conv7, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn7')
    act7 = mx.sym.Activation(data=bn7, act_type='relu', name=stage1name + '_relu7')
    #shortcut1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=(1,1), no_bias=True,
    #                               workspace=workspace, name=stage1name + '_sc')
    #act7 = act7 + shortcut1
    column_features = []
    for idx in range(seq_len):
        slice = mx.sym.slice_axis(data=act7, axis=3, begin=(idx*4), end=(idx*4+4))
        column_features.append(slice)
    #column_features = mx.sym.SliceChannel(data=act4, num_outputs=seq_len, axis=3, squeeze_axis=1)

    hidden_all =[]
    for idx in range(seq_len):
        hidden_all.append(column_features[idx])
    hidden_concat = mx.sym.concat(*hidden_all, dim=0)
    input = hidden_concat
    stage2name = 'Conv2'
    kernellist = [(3,3),(3,3),(3,3),(3,2)]
    num_filterlist = [32,32,32,32]
    padlist = [(0,1),(0,1),(0,0),(0,0)]
    stridelist = [(1,1),(1,1),(1,1),(1,1)]
    output = convstage(input,stage2name,4,num_filterlist,workspace,bn_mom,kernellist,stridelist,padlist)
    flat = mx.sym.Flatten(data=output)
    if dropout > 0.:
        hidden = mx.sym.Dropout(data=flat, p=dropout)
    pred = mx.sym.FullyConnected(name='fc1', data=hidden, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    shape = {"data" : (30, 1, 32, 64)}
    mx.viz.plot_network(pred,shape=shape).view()
    return sm

def convnet2(seq_len, num_label, dropout, bn_mom=0.9):
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable('label')
    stage1name = 'Conv1'
    num_filter = 32
    workspace = 256

    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2),
                               no_bias=True, workspace=workspace, name=stage1name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=stage1name + '_relu1')
    conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter , kernel=(5, 5), stride=(1, 1), pad=(2, 2),
                               no_bias=True, workspace=workspace, name=stage1name + '_conv2')
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=stage1name + '_relu2')
    conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2), no_bias=True,
                               workspace=workspace, name=stage1name + '_conv3')
    bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=stage1name + '_relu3')
    conv4 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2), no_bias=True,
                               workspace=workspace, name=stage1name + '_conv4')
    bn4 = mx.sym.BatchNorm(data=conv4, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn4')
    act4 = mx.sym.Activation(data=bn4, act_type='relu', name=stage1name + '_relu4')
    conv5 = mx.sym.Convolution(data=act4, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2),
                               no_bias=True, workspace=workspace, name=stage1name + '_conv5')
    bn5 = mx.sym.BatchNorm(data=conv5, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn5')
    act5 = mx.sym.Activation(data=bn5, act_type='relu', name=stage1name + '_relu5')
    conv6 = mx.sym.Convolution(data=act5, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2), no_bias=True,
                               workspace=workspace, name=stage1name + '_conv6')
    bn6 = mx.sym.BatchNorm(data=conv6, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn6')
    act6 = mx.sym.Activation(data=bn6, act_type='relu', name=stage1name + '_relu6')
    conv7 = mx.sym.Convolution(data=act6, num_filter=num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2), no_bias=True,
                               workspace=workspace, name=stage1name + '_conv7')
    bn7 = mx.sym.BatchNorm(data=conv7, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=stage1name + '_bn7')
    act7 = mx.sym.Activation(data=bn7, act_type='relu', name=stage1name + '_relu7')
    #shortcut1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=(1,1), no_bias=True,
    #                               workspace=workspace, name=stage1name + '_sc')
    #act7 = act7 + shortcut1
    column_features = []
    for idx in range(seq_len):
        slice = mx.sym.slice_axis(data=act7, axis=3, begin=(idx*8), end=(idx*8+8))
        column_features.append(slice)
    #column_features = mx.sym.SliceChannel(data=act4, num_outputs=seq_len, axis=3, squeeze_axis=1)

    hidden_all =[]
    for idx in range(seq_len):
        hidden_all.append(column_features[idx])
    hidden_concat = mx.sym.concat(*hidden_all, dim=3)
    sm = mx.sym.Dropout(hidden_concat, p=0.75)
    shape = {"data" : (30, 3, 32, 128)}
    mx.viz.plot_network(sm,shape=shape).view()
    return hidden_concat

def no_slice_convnet(seq_len, num_label, dropout, bn_mom=0.9, istrain = True):
    print ('seq_len : ', seq_len)
    data = mx.sym.Variable(name='data')
    workspace = 256
    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (3, 3), (5, 5), (3, 3), (3, 3), (5, 5), (3, 3), (3, 3)]
    num_filterlist = [16, 32, 32, 32, 64, 64, 64, 128, 128, 128]
    padlist = [(1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1)]
    output = convstage(data, stagename,len(kernellist),num_filterlist, workspace,bn_mom,kernellist,stridelist,padlist)
    column_features = mx.sym.SliceChannel(data=output, num_outputs=seq_len, axis=3, squeeze_axis=1)
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.sym.Flatten(data=column_features[seqidx])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.Activation(data=hidden_concat, act_type='tanh')
    pred = mx.sym.FullyConnected(name='fc1', data=pred, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm2 = mx.sym.WarpCTC(name='ctc-loss', data=pred, label=label, label_length=num_label, input_length=seq_len)
    sm3 = mx.sym.contrib.ctc_loss(name='ctc-loss', data=pred, label=label)
    sm = mx.sym.softmax(data=pred,axis=1)
    #shape = {"data" : (30, 3, 32, 128)}
    #mx.viz.plot_network(pred,shape=shape).view()
    if istrain:
        return sm2
    else:
        return sm
def baolinet(dropout, bn_mom=0.9, istrain = True):
    data = mx.sym.Variable(name='data')
    workspace = 256
    label = mx.sym.Variable('label')
    stagename = 'conv1'
    kernellist = [(3, 3), (5, 5), (3, 3), (3, 3), (5, 5), (3, 3), (3, 3), (5, 5), (3, 3), (3, 3)]
    num_filterlist = [16, 32, 32, 32, 64, 64, 64, 32, 32, 10]
    padlist = [(1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1)]
    stridelist = [(1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1)]
    output = convstage(data, stagename,len(kernellist),num_filterlist, workspace,bn_mom,kernellist,stridelist,padlist)
    flat = mx.sym.flatten(output)
    if dropout > 0:
        flat = mx.sym.Dropout(data=flat, p=dropout)
    hidden1= mx.sym.FullyConnected(name="fc1",data=flat,num_hidden=98)
    activation = mx.sym.Activation(data=hidden1,act_type='relu', name='actfn')
    if dropout > 0.:
        activation = mx.sym.Dropout(data=activation, p=dropout)
    hidden2 = mx.sym.FullyConnected(name="fc2", data=activation, num_hidden=98)

    sm2 = mx.sym.SoftmaxOutput(data=hidden2,label=label,name='cross_entropy')
    mx.sym.softmax_cross_entropy()
    sm = mx.sym.softmax(data=hidden2,axis=1)
    shape = {"data" : (30, 3, 32, 96)}
    mx.viz.plot_network(hidden2,shape=shape).view()
    if istrain:
        return sm2
    else:
        return sm
if __name__ == '__main__':
    sm =convnet(8, 10, 0.75)
    '''
    sm = no_slice_convnet(16, 10, 0.75, istrain=False)
    data = np.random.uniform(0, 1, (1, 3, 32, 128))

    mod = mx.mod.Module(sm,context=[mx.gpu(3)])
    mod.bind(data_shapes=[('data', (1, 3, 32, 128))])
    mod.init_params()
    data = mx.io.DataBatch(data=[mx.nd.array(data)])
    t1 = time.time()
    for i in range(100):
        y = mod.forward(data)

    mx.nd.waitall()
    t2 = time.time()
    print(t2 - t1)/100
    
    sm = convnet2(16, 10, 0.75)
    '''