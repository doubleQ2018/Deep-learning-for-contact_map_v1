#!/usr/bin/env python

import cPickle as pickle
import numpy as np
from collections import defaultdict
from collections import OrderedDict
from config import *

def read_pkl(name):
    with open(name) as fin:
        return pickle.load(fin)


class feature(object):
    def __init__(self, train, valid, test):
        self.train_info = read_pkl(train)
        self.valid_info = read_pkl(valid)
        self.test_info = read_pkl(test)

        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = len(self.train_info)

    def __extract_single__(self, info):
        seq = info['sequence']
        seqLen = len(seq)
        acc = info['ACC']
        ss3 = info['SS3']
        pssm = info['PSSM']
        sequence_profile = np.concatenate((pssm, ss3, acc), axis = 1)
        #shape = (L, 26)
        ccmpred = info['ccmpredZ']
        psicov = info['psicovZ']
        other = info['OtherPairs']
        pairwise_profile = np.dstack((ccmpred, psicov))
        pairwise_profile = np.concatenate((pairwise_profile, other), axis = 2)
        #shape = (L, L, 5)
        true_contact = info['contactMatrix']
        true_contact[true_contact < 0] = 0 # transfer -1 to 0
        #shape = (L, L)
        ####### change the shape to (L, L, 2) 
        tmp = np.where(true_contact>0, 0, 1)
        true_contact = np.stack((tmp, true_contact), axis=-1)
        #######

        return sequence_profile, pairwise_profile, true_contact

    def __process_feature__(self, infos):
        f1 = []
        f2 = []
        fl = []
        for info in infos:
            x1, x2, y = self.__extract_single__(info)
            f1.append(x1)
            f2.append(x2)
            fl.append(y)
            #data['features'].append([x1, x2])
            #data['labels'].append(y)
        #return [np.concatenate(f1, axis=0), np.concatenate(f2, axis=0), np.concatenate(fl, axis=0)]
        return [np.array(f1), np.array(f2), np.array(fl)]

    def get_feature(self):
        # training data
        self.train_data = self.__process_feature__(self.train_info)
        # validation data
        self.valid_data = self.__process_feature__(self.valid_info)
        # testing data
        self.test_data = self.__process_feature__(self.test_info)
        return self.train_data, self.valid_data, self.test_data
    
    def next_batch(self, batch_size, shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._f1 = self.train_data[0][idx]
            self._f2 = self.train_data[1][idx]
            self._fl = self.train_data[2][idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part1 = self._f1[start:self._num_examples]
            data_rest_part2 = self._f2[start:self._num_examples]
            data_rest_part3 = self._fl[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._f1 = self.train_data[0][idx0]
            self._f2 = self.train_data[1][idx0]
            self._fl = self.train_data[2][idx0]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part1 = self._f1[start:end]  
            data_new_part2 = self._f2[start:end]  
            data_new_part3 = self._fl[start:end]  
            return np.concatenate((data_rest_part1, data_new_part1), axis=0), np.concatenate((data_rest_part2, data_new_part2), axis=0), np.concatenate((data_rest_part3, data_new_part3), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._f1[start:end], self._f2[start:end], self._fl[start:end]

'''
    def lensBin(self):
        dic = {}
        count = 0
        for info in self.train_info:
            count += 1
            l = len(info['sequence'])
            if dic.has_key(l):
                dic[l] += 1
            else:
                dic[l] = 1
        order_dict = OrderedDict(dic)
        for k, v in order_dict.items():
            print "%d %d" %(k, v)
        print 'count = %d' %count


F = feature(train_file, valid_file, test_file)
F.lensBin()
train_data, valid_data, test_data = F.get_feature()
for i in xrange(1):
    batch = F.next_batch(1)
'''
