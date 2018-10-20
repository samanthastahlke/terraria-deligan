import pickle
import os
import numpy as np

def unpickle(file):
    fo = open(file, 'rb')
    d = pickle.load(fo, encoding='latin1')
    fo.close()
    num_samples = len(d['labels'])
    print('Loaded ', num_samples, ' samples from ', file)
    return {'x': np.cast[np.float32]((-127.5 + d['data'].reshape((num_samples,3,32,32)))/128.),
            'y': np.array(d['labels']).astype(np.uint8)}

def load(data_dir, subset='train'):
    if subset=='train':
        train_data = unpickle(os.path.join(data_dir,'32Uncut\\terraria-batch-train'))
        trainx = train_data['x']
        trainy = np.array([((lab // 10) - 1) for lab in train_data['y']])
        return trainx, trainy
    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'32Uncut\\terraria-batch-test'))
        testx= test_data['x']
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
