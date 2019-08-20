# -*- coding: utf-8 -*-
import sys
#sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx
import time
import cv2
import random

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None  # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class TextIter(mx.io.DataIter):
    def __init__(self, data_path, data_root, batch_size, num_label, 
                 init_states = None, config = None, data_name='data', label_name='label', bucket_key = 5):
        super(TextIter, self).__init__()
        self.data_path = data_path
        self.data_root = data_root
        self.batch_size = batch_size
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states] ##每个元素是 nd.zeros(（batch_size, 256）)
        
        self.num_label = num_label ##
        self.config = config
        self.data_name = data_name
        self.label_name = label_name
        self.bucket_key = bucket_key * 4
        
        self.data = self.read_content(self.data_path)##[[index, label, path],]
        self.data_length = len(self.data)
        self.batch_idx = []
        
        self.num_filter = 3 ##彩色图是3
        self.base_hight = 32##最大宽高比为10，这里设定的高度为32
        self.max_ratio = 5
        self.min_ratio = 3
        
        if 'lstm' in self.config.configname or 'sru' in self.config.configname:
            self.provide_data = [('data', (self.batch_size, self.num_filter, self.base_hight, self.max_ratio * self.base_hight))] + init_states
        else:
            self.provide_data = [('data', (self.batch_size, self.num_filter,self.base_hight, self.max_ratio * self.base_hight))] ##关键参数，决定输入的形状
            
        self.provide_label = [('label', (self.batch_size, self.num_label))]
               
        self.count = 0 ##batch计数用的
        self.bucket_list = []
        self.default_bucket_key = 40
        self.data_root = data_root
        
        self.data_plan,self.data_labels=self.get_batch_plan()
        
    def iter_next(self): ##必须函数
        return self.count < self.data_length / self.batch_size
    
    def next(self): ##必须next函数。next_batch都不行。这是mx.io.DataIter的标准输入框架
        timetotalCost=0   
        if  self.iter_next():
            starttime=time.time()
#            current_batch_idxes = self.get_batch_plan() ##内存打乱的index
            current_batch_idxes = self.data_plan[self.count*self.batch_size:self.count*self.batch_size+self.batch_size] ##内存打乱的index
            data_labels = self.data_labels[self.count*self.batch_size:self.count*self.batch_size+self.batch_size]
            self.count += 1
            
            if 'lstm' in self.config.configname or 'sru' in self.config.configname:
                init_state_names = [x[0] for x in self.init_states]
            
            
            if self.num_filter == 1:
                data_images,data_labels = self.get_batch_gray(current_batch_idxes)
            else:
#                data_images,data_labels = self.get_batch_color(current_batch_idxes)
                data_images = self.get_batch_color(current_batch_idxes)
            
            endtime=time.time()
           
            if 'lstm' in self.config.configname or 'sru' in self.config.configname:
                data_images = [mx.nd.array(data_images)] + self.init_state_arrays
            else:
                data_images = [mx.nd.array(data_images)]
            
            data_labels = [mx.nd.array(data_labels)] ##记得加[]，不然维度不对
           
            
            if 'lstm' in self.config.configname or 'sru' in self.config.configname:
                data_names = ['data'] + init_state_names ##['data', ('l0_init_c', (20, 256))]
            else:
                data_names = ['data']
            
            timetotalCost += (endtime-starttime)
            # print timetotalCost
            data_batch = SimpleBatch(data_names, data_images, ['label'], data_labels, 20) ## 不要写  data_names = ['data']，否则会出错 
            return data_batch
        else:
            raise StopIteration
            
    def reset(self): ##必须函数
        self.count = 0
        self.data_plan,self.data_labels=self.get_batch_plan()
        
    def get_batch_plan(self):
        tmp_idx = [i for i in range(self.data_length)]
        random.shuffle(tmp_idx)
        
        data_labels=[]
        for idx in tmp_idx:
            label_data = self.data[idx][1:-1]
            for ii in range(len(label_data)):
                label_data[ii] = int(label_data[ii])
            data_labels.append(label_data)
        
#        num_batch=self.data_length//self.batch_size
#        
#        batch_idx=[]
#        for i in range(num_batch):
#            start=i*self.batch_size
#            end=(i+1)*self.batch_size
#            batch_idx.append(tmp_idx[start:end])

#        count = self.batch_size
#        tmp_idx = [i for i in range(self.data_length)]
#        random.shuffle(tmp_idx)
#        
#        batch_idx = []
#        for i in range(count):
#            batch_idx.append(tmp_idx[i])
            
#        batch_idx= random.sample(range(0,self.data_length),self.batch_size)
#        print batch_idx[0:5]
        return tmp_idx, data_labels
    
    def read_content(self, path):
        with open(path) as ins:
            content = ins.read()
            content = content.split('\n')
            content.pop()
        data = []
        for line in content:
            line = line.split('\t')
            data.append(line)
        return data ##[[index, label, path],]
    
    def get_batch_gray(self, current_batch_idxes): ##一个batch中img路径集合
        data_images = []
        data_labels = []
        
        base_hight = self.base_hight
        max_ratio = self.max_ratio
        
        for idx in current_batch_idxes:
            label_data = self.data[idx][1:-1]
            for ii in range(len(label_data)):
                label_data[ii] = int(label_data[ii])
            data_labels.append(label_data)
            
            img_path = self.data[idx][-1]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            hight,width = img.shape
        
            ratio = (1.0 * width / hight)            
            if ratio > max_ratio:
                ratio = max_ratio
            if ratio < 1:
                ratio = 1
                
            hight = base_hight ##等比缩放图片
            width = int(base_hight*ratio)
        
            img = cv2.resize(img, (int(width), hight))
            img = np.array(img, dtype=np.float16)
            
#            img  = np.array([img])
            img = img/255
            if width % hight != 0: ##不整除不灵
                padding_ratio = (min(int(ratio + 1), max_ratio))
                new_img = np.zeros((1, base_hight, base_hight * padding_ratio))
                for i in range(1): ###?
                    padding_value = int(np.mean(img[i][:][-1]))
                    z = np.ones((base_hight, base_hight *
                                 padding_ratio - width)) * padding_value
                    new_img[i] = np.hstack((img[i], z))
                data_images.append(new_img)
            else:
                data_images.append(img)
                
        return data_images, data_labels
    
    def get_batch_color(self, current_batch_idxes): ##一个batch中img路径集合
        data_images = []
        data_labels = []
        
        base_hight = self.base_hight
        max_ratio = self.max_ratio
        min_ratio = self.min_ratio        
        
     
        for idx in current_batch_idxes:            
            img_path = self.data[idx][-1]
            img = cv2.imread(img_path) ##彩色图像
            hight,width,_ = img.shape
        
        
            #ratio = (1.0 * width / hight)            
            #if ratio > max_ratio:
            #    ratio = max_ratio
            #if ratio < min_ratio:
            #    ratio = min_ratio
            #fix ratio by wzm
            ratio = 4    
            hight = base_hight ##等比缩放图片
            width = int(base_hight*ratio)
            
            img = cv2.resize(img, (width, hight), interpolation=cv2.INTER_AREA)
            img = np.array(img, dtype=np.float32)
            
            img = np.transpose(img, (2, 0, 1)) ##通道转换
#            img  = np.array([img])
            img = img / 255

            if (ratio != max_ratio) or width % hight != 0: ##不整除不灵
                padding_ratio = max_ratio
                new_img = np.zeros((self.num_filter, base_hight, base_hight * padding_ratio))
                for i in range(self.num_filter): ##
                    padding_value = int(np.mean(img[i][:][-1]))
                    z = np.zeros((base_hight, base_hight * padding_ratio - width)) * padding_value
                    new_img[i] = np.hstack((img[i], z)) ##第i个通道
                data_images.append(new_img)
            else:
                data_images.append(img)        
            

#        print np.array(data_images).shape

        return data_images            

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


if __name__ == '__main__':
    num_label = 10
    data_train = TextIter('testData.lst', '/home/zechenhu/data/Plate/data_all', 30, num_label)
    batch = data_train.next_batch()
    print batch.provide_data
    print batch.data, batch.label
