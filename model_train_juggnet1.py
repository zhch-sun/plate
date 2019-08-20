# -*- coding: utf-8 -*-
'''
若输出的 ctcloss 精度为负数，说明出现梯度爆炸现象。
有两个方法可以改进：
0、重新计算一遍，可能是初始化的时候参数随机的不好
1、改小学习率。下降大概10倍
2、改clip_gradient，改小梯度模长的限制
'''
# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys, random
import numpy as np
import mxnet as mx
import os
import logging

def mkdir(path):
    path = path.strip() ##删除空格回车符啥的
    path = path.rstrip("/")##去掉右边的‘/’
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print (path , ' create successfully!') #!!
        return True

cur_path = os.path.abspath(os.curdir)
print cur_path
#cur_path = '/data/huzechen/model_yyh/BeijingPk'

name1 = 'Big_Jugg_net'
name2 = 'resnet18+lstm'
class configure(object):
    def __init__(self, f='train', name='Jugg_net_1_Clean1_finetuning'):
        self.makeset = False
        self.net_size = 0.5 ##?应该和后面的cnn_size 一致，表示 channel 数量的折减倍率。
        self.lr_scheduler = mx.lr_scheduler.FactorScheduler(step=450, factor=0.99) ##学习计划，每隔450，变为原来的0.99
        self.epochload = 316
        
        self.withequdata = 'dataNoEqu'
        self.doubleSingleFlag=False ##单双行是否区分
        
        if self.withequdata == 'dataEqu':
            if self.doubleSingleFlag==False:
#                self.trainpath = '/data/huzechen/work_yyh/BeijingPk/data-Clean1/PProvinceDivide_76wan'
                self.trainpath = '/data/huzechen/work_yyh/BeijingPk/data-Clean1/ProvinceDivide_SinDoubline/Doublelineplate'
            else:                
                self.trainpath = '/data/huzechen/work_yyh/BeijingPk/data-Clean1/ProvinceDivide_SinDoubline'
        else:
            self.trainpath = '/data/huzechen/work_yyh/BeijingPk/data-Clean1/trainData_all_Clean1_76w.lst'
#            self.trainpath = '/data/huzechen/work_yyh/BeijingPk/data-Clean1/trainData_Big_Jugg_net_Clean1.lst'
#            self.trainpath = '/data/huzechen/work_yyh/BeijingPk/data-Clean1/trainData_all_Clean1_updoubline.lst'
            
#        self.testpath = '/data/huzechen/work_yyh/BeijingPk/data-Clean1/testData_all_Clean1_updoubline.lst'
        self.testpath = '/data/huzechen/work_yyh/BeijingPk/data-Clean1/testData_all_Clean1.lst'
        
        
        self.function = f ## ‘train’ 或者 ‘test’ 操纵。
        self.configname = name ##网络结构的名字。sym_gen中根据名字来做相应的操作
        self.lr = 0.01 ##初始学习速率
        self.clip_gradient = 0.01 ##梯度模长限制。错误样本的梯度较大，通过比例回调来限制其梯度。类似地可以限制梯度在一定的区间内
        self.optimizer = 'sgd' ##优化器 sgd 或者 adam
        self.num_epoch = 1000 ##训练迭代的次数
        self.cnn_size = 1 ##cnn_size用于限制cnn channel 的数量。channel * cnn_size 为最终channel数量
        
        
        if name==name2: ## 'resnet18+lstm' 网络
            self.path_test = 'nosep_Testset.lst'
            
        if self.function == 'test':
            self.batchsize = 20 ##一个batch的大小
            self.contexts=[mx.gpu(2),mx.gpu(3)] ##采用2号和3号gpu
            
        if self.function == 'train':
            self.batchsize = 1000
            self.test_batchsize=100
            if self.withequdata == 'dataEqu':            
                self.contexts = [mx.gpu(3),mx.gpu(4),mx.gpu(5), mx.gpu(6), mx.gpu(7)]
#                self.contexts = [mx.gpu(0),mx.gpu(1),mx.gpu(2), mx.gpu(3), mx.gpu(4)]
            else:
                self.contexts = [mx.gpu(0),mx.gpu(1),mx.gpu(2), mx.gpu(3), mx.gpu(4)]

from text_lstm_resnet50 import  Jugg_net_1
from convnet import  convnet, no_slice_convnet
import text_bucketing_iter_dataEqu_doubleSingle as dataEqu
import text_bucketing_iter_blstm as dataNoEqu

import time
import cv2, random
from io import BytesIO ##中文字符计数。一个中文字符3位

config = configure() ##默认是 f='train'

##############################################################################################
##实现打印日志的基础配置
logging.basicConfig(level=logging.DEBUG, ##设置日志级别，默认为logging.WARNING
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                ##asctime：打印日志的时间；filename：打印当前执行程序名；lineno打印日志的当前行号；
                ##levelname：打印日志级别名称；message：打印日志信息
                datefmt='%a, %d %b %Y %H:%M:%S', ##时间格式
                filename=config.configname+'_'+config.withequdata+'.log', ##日志文件格式
                filemode='a+') ##模式是写入

##############################################################################################
console = logging.StreamHandler() ##日志输出。将日志信息输出到sys.stdout, sys.stderr 或者类文件对象
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
#################################################################################################

logging.debug('This is debug message') ##显示信息
logging.info(config.configname) ##题目是网络名字


BATCH_SIZE = config.batchsize
#contexts = [mx.gpu(0)}

def ctc_label(p): ##B函数
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i+1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret

def remove_blank(l): ##删除空格
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret

##仅用于 fit
def Accuracy(label, pred): ##定义精度。label和pred中的条目一一对应
    global BATCH_SIZE ##全局变量 batch 大小
    global SEQ_LENGTH ##车牌长度

    hit = 0. ##正确车牌数量
    total = 0. ##车牌数量
    for i in range(len(label)):
        l = remove_blank(label[i]) ##移除第i个标签中的空格
        p = []

        for k in range(len(pred)/len(label)): ##k的取值其实就0和1。0是单行车牌，1是双行车牌
            p.append(np.argmax(pred[k * (len(label)) + i])) ##返回概率最大的字母索引值
        p = ctc_label(p)

        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False ##存在字母错误。
                    break
            if match:
                hit += 1.0 
        total += 1.0
    return hit / total


if __name__ == '__main__':
    import logging  ##日志文件
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    num_hidden = 256
    num_lstm_layer = 2
    config.num_epoch = 1000
    num_label = 10  ##标签长度。就是车牌号位数
    
    
    def sym_gen(seq_len, config = config): ##?返回一个神经网络        
        if 'Jugg_net_1_Clean1' in config.configname:
            return Jugg_net_1(seq_len,num_hidden=num_hidden,num_label=num_label,dropout=0.75), ('data',), ('label',)
    
    def norm_stat(d):
        return mx.nd.norm(d)/np.sqrt(d.size) ##标准差。平均模长        
   
    
    init_c = [('l%d_init_c'%l, (config.batchsize, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (config.batchsize, num_hidden)) for l in range(num_lstm_layer)]
    
    if config.configname == 'bi_lstm_unroll': ##blstm需要定义h和c的初始状态
        init_states = init_c + init_h 
    else:
        init_states = init_c
    
    testinit_c = [('l%d_init_c' % l, (30, num_hidden)) for l in range(num_lstm_layer)]
    testinit_h = [('l%d_init_h' % l, (30, num_hidden)) for l in range(num_lstm_layer)]
    
    if config.configname == 'bi_lstm_unroll':
        testinit_states = testinit_c + testinit_h
    else:
        testinit_states = testinit_c
        

    data_root='/home/zechenhu/data/Plate/data_all'
    test_root='/home/zechenhu/data/Plate/data_all'
    buckets=[4*i for i in range(1,num_label+1)] ##[4,8,...,40] ##长宽比
    
    if config.withequdata=='dataEqu': 
        print 'dataEqu is in process!'
        data_train = dataEqu.TextIter(config.trainpath, data_root, config.batchsize, num_label, init_states, config,doubleSingleFlag=config.doubleSingleFlag)
    else:
        print 'dataNoEqu!'
        data_train = dataNoEqu.TextIter(config.trainpath, data_root, config.batchsize, num_label, init_states, config,)
    
    data_val = dataNoEqu.TextIter(config.testpath, test_root, config.test_batchsize, num_label, testinit_states, config,) ##30是测试集大小
    model = mx.mod.BucketingModule( ##变长度输入的模型
        sym_gen             = sym_gen, 
        default_bucket_key  = 20, ##The key for the default bucket. ##这里必须 = 4 * max_ratio.(iter文件中的参数)
        context             = config.contexts)

    prefix = config.configname+'_'+config.withequdata+'_model/param' ##前缀
    mkdir (cur_path+'/'+config.configname+'_'+config.withequdata+'_model')
    
    if(config.epochload != 0):
        if config.withequdata == 'dataEqu': 
            n_epoch_load = config.epochload
            sym, arg_params, aux_params = \
               mx.model.load_checkpoint('Jugg_net_1_Clean1_doubSing_dataEqu_model/param',n_epoch_load) ##加载模型。第epochload步的模型
        else:
            n_epoch_load = config.epochload
            sym, arg_params, aux_params = \
               mx.model.load_checkpoint('Jugg_net_1_Clean1_finetuning_dataNoEqu_model/param',n_epoch_load) ##加载模型。第epochload步的模型
    else:
        arg_params = None
        aux_params = None    
    
    if(config.optimizer=='sgd'):
        optimizer = 'sgd'
        optimizer_params = {#'rescale_grad': 1.0/(config.batchsize*32),
                            'learning_rate': config.lr,
                            'momentum': 0.9,
                            'wd': 0.0005,
                            'clip_gradient': config.clip_gradient,
                            'lr_scheduler': config.lr_scheduler}
    else:
        optimizer = mx.optimizer.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, clip_gradient=0.01)
        optimizer_params = None
            #{'learning_rate':config.lr, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-08, 'lazy_update':True }
    
    print config.trainpath
    print config.testpath
    print 'if DoubleSingle? ',config.doubleSingleFlag
    '''训练部分'''   
    if(config.function == 'train' or config.function == 'test'):
        model.fit(
                train_data          = data_train,
                eval_data           = data_val, ##
                eval_metric         = mx.metric.np(Accuracy), ##默认是“accuracy”，显示训练过程中的模型表现 ##每完成一个batch的训练就会显示一组精度
                optimizer           = optimizer, ##优化器，默认是‘sgd’
                optimizer_params    = optimizer_params, ##设置优化参数的。默认是(('learning_rate', 0.01),)
                arg_params          = arg_params,
                aux_params          = aux_params,
                kvstore             = 'device', ## 默认是‘local’，存储key value，控制在多台设备上数值的同步
                initializer         = mx.init.Xavier(factor_type="in", magnitude=0.1),  ##保证各层梯度的方差大致相同。
                num_epoch           = config.num_epoch,
                epoch_end_callback  = mx.callback.do_checkpoint(prefix, 2), ## 每10次保存一次模型
                batch_end_callback  = mx.callback.Speedometer(config.batchsize, 10), ##周期性记录训练速度和精度。每50个batch一次。
                begin_epoch         = config.epochload,##开始的循环次数3
                )
