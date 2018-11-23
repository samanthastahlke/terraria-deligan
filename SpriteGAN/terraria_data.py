import pickle
import os
import numpy as np

fine_label_map = {
    10 : 0,
    11 : 1,
    12 : 2,
    13 : 3,
    14 : 4,
    20 : 5,
    21 : 6,
    22 : 7,
    23 : 8,
    30 : 9,
    31 : 10,
    32 : 11,
    33 : 12,
    34 : 13,
    35 : 14,
    40 : 15,
    41 : 16,
    42 : 17,
    43 : 18,
    44 : 19
}

def unpickle(file):
    fo = open(file, 'rb')
    d = pickle.load(fo, encoding='latin1')
    fo.close()
    num_samples = len(d['labels'])
    print('Loaded ', num_samples, ' samples from ', file)
    return {'x': np.cast[np.float32]((-127.5 + d['data'].reshape((num_samples,3,32,32)))/128.),
            'y': np.array(d['labels']).astype(np.uint8)}

def load(data_dir, subset='train', fine_labels=False):
    if subset == 'train':
        train_data = unpickle(os.path.join(data_dir,'32Uncut\\terraria-batch-train'))
        trainx = train_data['x']
        if fine_labels:
            trainy = np.array([fine_label_map[lab] for lab in train_data['y']])
        else:
            trainy = np.array([((lab // 10) - 1) for lab in train_data['y']])
        return trainx, trainy
    elif subset == 'test':
        test_data = unpickle(os.path.join(data_dir,'32Uncut\\terraria-batch-test'))
        testx = test_data['x']
        if fine_labels:
            testy = np.array([fine_label_map[lab] for lab in test_data['y']])
        else:
            testy = np.array([((lab // 10) - 1) for lab in test_data['y']])
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')

def cifar_unpickle(file):
    fo = open(file, 'rb')
    d = pickle.load(fo, encoding='latin1')
    fo.close()
    return {'x': np.cast[np.float32]((-127.5 + d['data'].reshape((10000,3,32,32)))/128.), 'y': np.array(d['labels']).astype(np.uint8)}

def cifar_load(data_dir, subset='train'):
    if subset=='train':
        train_data = [cifar_unpickle(os.path.join(data_dir,'cifar-10-batches-py\\data_batch_' + str(i))) for i in range(1,6)]
        trainx = np.concatenate([d['x'] for d in train_data],axis=0)
        trainy = np.concatenate([d['y'] for d in train_data],axis=0)
        return trainx, trainy
    elif subset=='test':
        test_data = cifar_unpickle(os.path.join(data_dir,'cifar-10-batches-py\\test_batch'))
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')
