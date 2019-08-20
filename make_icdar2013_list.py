#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import cv2
def make_list(out_fn, in_fn):
    xue_list = []
    train_list = open(in_fn, 'r')
    # print train_list.readlines()
    train_lines = train_list.readlines()
    print 'total-lines : ', len(train_lines)
    dot_count = 0
    blank_count = 0
    counter = 0
    max_len_fixed = 10
    crop_cocotext_train = open(out_fn, 'w')
    char2idx = {}
    char2idx[' '] = 0
    idx2char = {}
    idx2char[0] = ' '
    vocab = []
    vocab.append(' ')
    idx = 1
    max_len = 0
    sig = 0
    char2idx = json.load(open('newchar2idx.json', 'r'))
    char2idx = json.JSONDecoder().decode(char2idx)

    #huzechen 2018.1.16
    train_len = 0

    for line in train_lines[train_len:]:
        #line = line[:-2]
        print(line)
        img_id = line.split(",")[0]
        spline = line.split(",")[1]
        text = line.split(",")[1][2:-2]

        if "?" in text:
            print("?")
            continue

        if '学' in text:
            xue_list.append(line)
        # print img_id, text

        max_len = max(max_len, len(text))
        label_string = ""
        text = text.decode('utf-8')
        if len(text) > max_len_fixed:
            print('bug')
        for char in text:
            if char == '警':
                print('233')
            if char2idx.has_key(char):
                label_string += '\t' + str(char2idx[char])
            else:
                sig = 1
        if sig == 1 :
            sig = 0
            continue
        if len(text) < max_len_fixed:
            for i in range(max_len_fixed - len(text)):
                label_string += '\t' + str(0)
        line_to_lst = str(counter) + label_string + '\t' + img_id + '\n'
        path = line_to_lst.split('\t')[-1].replace('\n', '')
        # line_to_lst = line_to_lst.replace('/home/huzechen/work/')
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('error')
        else:
            counter += 1
            crop_cocotext_train.write(line_to_lst)
    print counter


def get_char(file_name, train_path, char):
    xue_list = []
    train_list = open(train_path, 'r')
    # print train_list.readlines()
    train_lines = train_list.readlines()
    print 'total-lines : ', len(train_lines)
    char2idx = {}
    char2idx[' '] = 0
    idx2char = {}
    idx2char[0] = ' '
    vocab = []
    vocab.append(' ')
    idx = 1
    max_len = 0
    sig = 0
    char2idx = json.load(open('newchar2idx.json', 'r'))
    char2idx = json.JSONDecoder().decode(char2idx)

    # huzechen 2018.1.16
    train_len = 0

    for line in train_lines[train_len:]:
        # line = line[:-2]
        img_id = line.split(",")[0]
        spline = line.split(",")[1]
        length = len(spline)
        text = line.split(",")[1][2:-2]
        line = line.replace(',', '')
        if "?" in text:
            print("?")
            continue
        if text == '�':
            print("?")
            continue
        if char in text:
            xue_list.append(line)
    return xue_list

train_keys = ['13province_train', 'kakou_train', 'hongkong_train',
              'xunzhimei_train',  'newenergy_train',
              'brushplate_train', 'qujing_train',
              'deepv_train', 'NMLP_train',
              'doubrows_train', 'shoufeizhan_train', 'yellowplate_train']

test_keys = ['xunzhimei_test', 'newenergy_test', 'hongkong_test',
             'doubrows_test', 'deepv_test', 'kakou_test', 'yellowplate_test',
             'shoufeizhan_test', 'NMLP_test', '13province_test']

current_keys = ['newenergy_train']

path = '/home/huzechen/data/Recog_data/data_list/'

def make_trainset():
    trainset = []
    for key in train_keys:
        make_list(path + key + '_gt.lst', path + key + '_string.txt')
        f = open(path + key+'_gt.lst', 'r')
        trainset += f.readlines()
        f.close()
    print 'trainset_len:', len(trainset)
    f = open('Trainset.lst', 'w')
    f.writelines(trainset)
    f.close()


def make_testset():
    testset = []
    for key in test_keys:
        make_list(path + key + '_clean_gt.lst', path + key + '_clean_string.txt')
        f = open(path + key + '_clean_gt.lst', 'r')
        testset += f.readlines()
        f.close()
    print 'testset_len:', len(testset)
    f = open('Testset.lst', 'w')
    f.writelines(testset)
    f.close()

if __name__ == '__main__':
    path = '/data/huzechen/rectify2.0.0.1_recog_data/data_list/'
    # make_list('Testset.lst', '/home/huzechen/work/py_dglp/val.lst')
    for key in train_keys:
    # key = 'newenergy_train'
        make_list(path + key+'_string.lst', path + key+'_string.txt')
    # make_testset()
    # make_trainset()