import numpy as np
import mxnet as mx
import time
import cv2
import random

class configure(object):
    def __init__(self, f='train', name='Jugg_net'):
        self.continuetrain = True
        self.lr_scheduler = mx.lr_scheduler.FactorScheduler(step=450,factor=0.99)
        self.epochload = 21
        self.trainpath = ""
        self.testpath = ""
        self.function = f
        self.configname = name
        self.lr = 0.005
        self.clip_gradient = 0.1

        if name == 'no_slice_convnet':
            self.lr = 0.005
            self.clip_gradient = 0.01
        self.optimizer = 'sgd'
        self.num_epoch = 500
        self.cnn_size = 1
        self.path = 'Trainset.lst'
        self.path_test = 'Testset.lst'
        if self.function == 'test':
            self.batchsize = 400
            self.contexts=[mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
            if name == 'no_slice_convnet':
                self.batchsize=600
                self.contexts = [mx.gpu(0), mx.gpu(1), mx.gpu(2)]
            self.path = 'nosep_Testset.lst'
        if self.function == 'train':
            self.batchsize = 400
            self.contexts = [mx.gpu(0), mx.gpu(1), mx.gpu(2),mx.gpu(3)]
            if name == 'no_slice_convnet':
                self.batchsize=600
                self.contexts = [mx.gpu(0), mx.gpu(1), mx.gpu(2)]
        if self.function == 'timetest':
            self.batchsize = 1
            self.contexts = [mx.gpu(0)]
        if 'lstm' in self.configname:
            self.num_hidden = 256
            self.num_lstm_layer = 2

def default_read_content(path):
    with open(path) as ins:
        content = ins.read()
        content = content.split('\n')
        content.pop()
        return content


def get_path_from_content(content, num_label):
    paths = []
    for line in content:
        path = line.split('\t')[-1]
        paths.append(path)
    return paths


def get_label_from_content(content, num_label):
    labels = []
    for line in content:
        label = line.split('\t')[1:-1]
        labels.append(label)
    #print (labels)
    return labels


def get_image_batch(paths, data_root):

    data = []
    base_hight = 32
    max_ratio = 10
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #img = cv2.imread(path)

        shape = img.shape
        hight = shape[0]
        width = shape[1]
        ratio = (1.0 * width / hight)
        if ratio > max_ratio:
            ratio = max_ratio
        if ratio < 1:
            ratio = 1
        img = cv2.resize(img, (int(32 * ratio), 32))
        img = np.array(img, dtype=np.float16)
        img = img / 255
        hight = 32
        width = int(32 * ratio)
        assert hight == base_hight

        #img = np.transpose(img, (2, 0, 1))
        img  = np.array([img])
        if width % hight != 0:

            padding_ratio = (min(int(ratio + 1), max_ratio))
            new_img = np.zeros((1, base_hight, base_hight * padding_ratio))
            for i in range(1):
                padding_value = int(np.mean(img[i][:][-1]))
                z = np.ones((base_hight, base_hight *
                             padding_ratio - width)) * padding_value
                new_img[i] = np.hstack((img[i], z))
            data.append(new_img)
        else:
            data.append(img)
    return np.array(data)

def get_image_batch_colour(paths, data_root):

    data = []
    base_hight = 32
    max_ratio = 10
    for path in paths:
        img = cv2.imread(path)
        shape = img.shape
        hight = shape[0]
        width = shape[1]
        ratio = (1.0 * width / hight)
        if ratio > max_ratio:
            ratio = max_ratio
        if ratio < 1:
            ratio = 1
        img = cv2.resize(img, (int(32 * ratio), 32))
        img = np.array(img, dtype=np.float16)
        img = img / 255
        hight = 32
        width = int(32 * ratio)
        assert hight == base_hight

        img = np.transpose(img, (2, 0, 1))
        if width % hight != 0:

            padding_ratio = (min(int(ratio + 1), max_ratio))
            new_img = np.zeros((3, base_hight, base_hight * padding_ratio))
            for i in range(3):
                padding_value = int(np.mean(img[i][:][-1]))
                z = np.ones((base_hight, base_hight *
                             padding_ratio - width)) * padding_value
                new_img[i] = np.hstack((img[i], z))
            data.append(new_img)
        else:
            data.append(img)
    return np.array(data)

def get_image_batch_color_aff(paths):

    data = []
    base_hight = 32
    max_ratio = 5
    for path in paths:
        img = cv2.imread(path)
        shape = img.shape
        hight = shape[0]
        width = shape[1]
        ang = (np.random.random() * 20 - 10)/180*np.pi
        bias = float(np.tan(ang)) * width/2
        pst1 = np.float32([(width / 4, hight / 4), (width / 4 * 3, hight / 4), (width / 4 * 3, hight / 4 * 3)])
        pst2 = np.float32([(width / 4, hight / 4), (width / 4 * 3, hight / 4), (width / 4 * 3 + bias, hight / 4 * 3)])
        dstsize = (width+bias*2, hight)

        affm = cv2.getAffineTransform(pst1, pst2)
        img = cv2.warpAffine(img, affm, dstsize)
        shape = img.shape
        hight = shape[0]
        width = shape[1]
        ratio = (1.0 * width / hight)
        if ratio > max_ratio:
            ratio = max_ratio
        if ratio < 3:
            ratio = 3
        img = cv2.resize(img, (int(32 * ratio), 32))
        img = np.array(img, dtype=np.float16)
        img = img / 255
        hight = 32
        width = int(32 * ratio)
        assert hight == base_hight

        img = np.transpose(img, (2, 0, 1))
        if width % hight != 0:

            padding_ratio = (min(int(ratio + 1), max_ratio))
            new_img = np.zeros((3, base_hight, base_hight * padding_ratio))
            for i in range(3):
                padding_value = int(np.mean(img[i][:][-1]))
                z = np.ones((base_hight, base_hight *
                             padding_ratio - width)) * padding_value
                new_img[i] = np.hstack((img[i], z))
            data.append(new_img)
        else:
            data.append(img)
    return np.array(data)

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


class newTextIter(mx.io.DataIter):
    def __init__(self, path, data_root, batch_size,
                 init_states, num_label, config, data_name='data', label_name='label',
                 get_image_function=None, read_content=None, data_shape=[32, 320], buckets=[2, 6]):
        super(newTextIter, self).__init__()
        if get_image_function == None:
            if 'gray' in config.configname:
                self.get_image_function = get_image_batch
            else:
                self.get_image_function = get_image_batch_color_aff
        if read_content == None:
            self.read_content = default_read_content
            self.config = config
        #self.Sample_equilibrium = {'newenergy_train': 1, 'NMLP_train': 0.5, 'yellowplate_train': 2,
        #                           'deepv_train+brushplate_train': 3,
        #                           '13province_train+kakou_train+xunzhimei_train+qujing_train+shoufeizhan_train': 3.5}
        self.Sample_equilibrium = {'newenergy_train': 1, 'yellowplate_train': 1,
                                'deepv_train': 2, 'doubrows_train': 1,
                                '13province_train+kakou_train+xunzhimei_train+qujing_train+shoufeizhan_train': 3}
        self.num = 8.0
        self.data_root = data_root
        self.char_content = {}
        self.string_content = {}
        self.char_imagepaths = {}
        self.string_imagepaths = {}
        self.char_labels = {}
        self.string_labels ={}
        self.child_batchsize = {}
        self.current_id = {}
        for key in self.Sample_equilibrium.keys():
            keys = key.split('+')
            content = []
            for i in keys:
                content = content + self.read_content(path + '/' + i + '_string.lst')
            self.string_content[key] = content
        '''
        for key in self.Sample_equilibrium.keys():
            keys = key.split('+')
            content = []
            for i in keys:
                content = content + self.read_content(path + '/' + i + '_char.txt')
            self.char_content[key] = content
        '''
        #self.content = self.read_content(path)
        #print (path + 'records number : ', len(self.content))
        self.num_label = num_label
        for key in self.string_content.keys():
            self.string_imagepaths[key] = np.array(get_path_from_content(self.string_content[key], num_label))
        # for key in self.char_content.keys():
        #     self.char_imagepaths[key] = np.array(get_path_from_content(self.char_content[key], num_label))
        for key in self.string_content.keys():
            self.string_labels[key] = np.array(get_label_from_content(self.string_content[key], num_label))
        # for key in self.char_content.keys():
        #     self.char_labels[key] = np.array(get_label_from_content(self.char_content[key], num_label))
        #self.imagepaths = get_path_from_content(self.content, num_label)
        #self.imagelabels = get_label_from_content(self.content, num_label)
        self.ids = {}
        self.default_bucket_key = max(buckets)
        self.factor = 4
        #self.imagepaths = np.array(self.imagepaths)
        #self.imagelabels = np.array(self.imagelabels)
        self.bucket_list = []
        self.batch_size = batch_size
        self.make_data_plan()

        #self.bucket_images, self.bucket_labels, self.data_plan = self.make_buckets(
         #   buckets, data_root, batch_size)
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        if 'gray' in self.config.configname:
            num_filter = 1
        else:
            num_filter = 3
        if 'lstm' in self.config.configname or 'sru' in self.config.configname:
            self.provide_data = [('data', (batch_size, num_filter, data_shape[0], data_shape[1]))] + init_states
        else:
            self.provide_data = [('data', (batch_size, num_filter, data_shape[0], data_shape[1]))]
        self.provide_label = [('label', (batch_size, self.num_label))]

        #self.all_idx = range(len(self.content))
        self.current = 0

        self.size = len(self.data_plan)


    def make_data_plan(self):
        max_batch_num = 0
        for key in self.string_imagepaths.keys():
            self.ids[key] = []
            for i in range(len(self.string_imagepaths[key])):
                self.ids[key].append([key, i])
            self.child_batchsize[key] = int(self.Sample_equilibrium[key]*(self.batch_size/self.num))
            self.current_id[key] = 0
            length = len(self.ids[key])
            batch_num = int(length/(self.batch_size*self.Sample_equilibrium[key]/self.num))
            if batch_num > max_batch_num:
                max_batch_num = batch_num
            if max_batch_num*self.batch_size > 200000:
                max_batch_num = 200000/self.batch_size
        self.data_plan = []
        for idx in range(max_batch_num):
            for key in self.string_imagepaths.keys():
                if self.current_id[key] == 0:
                    random.shuffle(self.ids[key])
                self.data_plan = self.data_plan + self.ids[key][self.current_id[key]:(self.current_id[key]+self.child_batchsize[key])]
                if self.current_id[key]+self.child_batchsize[key] > len(self.ids[key]):
                    self.current_id[key] = 0
                else:
                    self.current_id[key] = self.current_id[key]+self.child_batchsize[key]

        self.current = 0

    def make_buckets(self, buckets, data_root, batch_size):
        print ("making buckets")
        buckets_len = len(buckets)
        bucket_images = []
        bucket_labels = []
        for i in range(buckets_len):
            bucket_images.append([])
            bucket_labels.append([])
        data_plan = []
        max_ratio = 10
        for label_idx, img in enumerate(self.imagepaths):
            if 'gray' in self.config.configname:
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                num_filter = 1
            else:
                image = cv2.imread(img)
                num_filter = 3
            #image = np.array(image,dtype=np.float16)
            image = image/255
            shape = image.shape
            hight = shape[0]
            width = shape[1]
            ratio = (1.0 * width / hight)
            if ratio > max_ratio:
                ratio = max_ratio
            if ratio < 1:
                ratio = 1
            hight = 32
            width = int(32 * ratio)
            if width % hight != 0:
                ratio = min(int(ratio + 1), max_ratio)
            else:
                ratio = int(ratio)
            bucket_images[ratio - 1].append(img)
            bucket_labels[ratio - 1].append(self.imagelabels[label_idx])
        #print (len(bucket_images))
        for bucket_idx, i in enumerate(bucket_images):
            length_bucket = len(i)
            print ("bucket " + " length :", length_bucket)
            for idx in range(length_bucket / batch_size):
                data_plan.append([bucket_idx, idx])
            if length_bucket != 0:
                self.bucket_list.append(bucket_idx)
        return bucket_images, bucket_labels, data_plan

    def iter_next(self):
        return self.current < self.size

    def next(self):
        if self.iter_next():
            i =self.current
            init_state_names = [x[0] for x in self.init_states]
            bucket = 4
            data = self.get_image_batch_color_affine(self.data_plan[self.current:(self.current+self.batch_size)])
            label = self.get_label(self.data_plan[self.current:(self.current+self.batch_size)])
            if 'lstm' in self.config.configname or 'sru' in self.config.configname:
                data_all = [mx.nd.array(data)] + self.init_state_arrays
            else:
                data_all = [mx.nd.array(data)]
            label_names = ['label']
            label_all = [mx.nd.array(label)]
            if 'lstm' in self.config.configname or 'sru' in self.config.configname:
                data_names = ['data'] + init_state_names
            else:
                data_names = ['data']
            data_batch = SimpleBatch(
                data_names, data_all, label_names, label_all, bucket*self.factor)
            self.current = self.current+self.batch_size
            return data_batch
        else:
            raise StopIteration
    def reset(self):
        self.make_data_plan()


    def get_label(self,data_plan):
        labels = []
        for key_id in data_plan:
            key = key_id[0]
            id = key_id[1]
            labels.append(self.string_labels[key][id])
        return np.array(labels)

    def get_image_batch_colour(self, data_plan):

        data = []
        base_hight = 32
        max_ratio = 6
        for key_id in data_plan:
            key = key_id[0]
            id = key_id[1]
            path = self.string_imagepaths[key][id]
            img = cv2.imread(path)

            shape = img.shape
            hight = shape[0]
            width = shape[1]
            ratio = (1.0  * width / hight)
            if ratio > max_ratio:
                ratio = max_ratio
            if ratio < 1:
                ratio = 1
            img = cv2.resize(img, (int(32 * ratio), 32), interpolation=cv2.INTER_AREA)
            img = np.array(img, dtype=np.float16)
            img = img / 255
            hight = 32
            width = int(32 * ratio)
            assert hight == base_hight

            img = np.transpose(img, (2, 0, 1))
            if (ratio != max_ratio) or (width % hight != 0):
                padding_ratio = max_ratio
                new_img = np.zeros((3, base_hight, base_hight * padding_ratio))
                for i in range(3):
                    padding_value = int(np.mean(img[i][:][-1]))
                    z = np.zeros((base_hight, base_hight *
                                 padding_ratio - width)) * padding_value
                    new_img[i] = np.hstack((img[i], z))
                data.append(new_img)
            else:
                data.append(img)
        return np.array(data)

    def get_image_batch_color_affine(self, data_plan):

        data = []
        base_hight = 32
        max_ratio = 4
        min_ratio = 3
        for key_id in data_plan:
            key = key_id[0]
            id = key_id[1]
            path = self.string_imagepaths[key][id]
            img = cv2.imread(path)
            shape = img.shape
            hight = shape[0]
            width = shape[1]
            ang = 0
            bias = float(np.tan(ang)) * width / 2
            pst1 = np.float32([(width / 4, hight / 4), (width / 4 * 3, hight / 4), (width / 4 * 3, hight / 4 * 3)])
            pst2 = np.float32(
                [(width / 4, hight / 4), (width / 4 * 3, hight / 4), (width / 4 * 3 + bias, hight / 4 * 3)])
            bias = int(bias)
            dstsize = (int(width + bias * 2), int(hight))

            affm = cv2.getAffineTransform(pst1, pst2)
            img = cv2.warpAffine(img, affm, dstsize)
            ratio = 3.3
            if ratio > max_ratio:
                ratio = max_ratio
            if ratio < min_ratio:
                ratio = min_ratio
            img = cv2.resize(img, (int(32 * ratio), 32), interpolation=cv2.INTER_AREA)
            img = np.array(img, dtype=np.float32)
            # img = img / 255
            hight = 32
            width = int(32 * ratio)
            assert hight == base_hight

            img = np.transpose(img, (2, 0, 1))
            if (ratio != max_ratio) or (width % hight != 0):
                padding_ratio = max_ratio
                new_img = np.zeros((3, base_hight, base_hight * padding_ratio))
                for i in range(3):
                    padding_value = int(np.mean(img[i][:][-1]))
                    z = np.zeros((base_hight, base_hight *
                                 padding_ratio - width)) * padding_value
                    new_img[i] = np.hstack((img[i], z))
                data.append(new_img)
            else:
                data.append(img)
        return np.array(data)

if __name__ == '__main__':
    config =configure()
    buckets = [4 * i for i in range(1, 10 + 1)]
    init_c = [('l%d_init_c' % l, (config.batchsize, 256)) for l in range(256)]
    init_h = [('l%d_init_h' % l, (config.batchsize, 256)) for l in range(256)]
    if config.configname.split('+')[-1] == 'lstm':
        init_states = init_c + init_h
    else:
        init_states = init_c
    dataiter = newTextIter('/home/zechenhu/work/Project', '', config.batchsize, init_states, 10, config, buckets=buckets)
    for databatch in dataiter:
        pass