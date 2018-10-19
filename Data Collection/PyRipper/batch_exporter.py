import pickle
from PIL import Image
import os
import pandas
import math
import numpy as np
import random
from sklearn.utils import shuffle

img_size = 32
test_subsample = 50

spritepath = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Sprites\\32Uncut\\'))
itemtable = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Tables\\terraria_items_classes.csv'))
classtable = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Tables\\terraria_classes_ref.csv'))
outpath = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Dataset\\32Uncut\\'))
testspritepath = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Sprites\\PicklingTest\\'))

print('Batching and pickling sprites from ', spritepath, ', saving to ', outpath)

itemdata = pandas.read_csv(itemtable)
classdata = pandas.read_csv(classtable)

v_flippable = {}

masterdata = {}
bcid_list = []
bcid_test_sample_set = {}
bcid_sample_indices = {}
data_length = {}
item_totals = {}
bcid_indices = {}
bcid_process_totals = {}

data_length['test'] = 0
bcid_list.append('test')

for i in range(0, len(classdata) - 1):

    row = classdata.iloc[i]

    if not math.isnan(row['BCID']):
        bcid = int(row['BCID'])
        bcid_list.append(bcid)
        data_length[bcid] = 0
        item_totals[bcid] = 0
        bcid_test_sample_set[bcid] = []
        item_count = sum(itemdata['BC'] == bcid)
        #print('Count found for BCID ', bcid, ': ', item_count)
        masterdata[bcid] = {}

    if not math.isnan(row['NCID']):
        ncid = int(row['NCID'])
        v_flippable[ncid] = True if (row['NCFlippable'] == 1) else False
        item_count = sum(itemdata['NC'] == ncid)
        item_count *= 2
        if v_flippable[ncid]:
            item_count *= 2
        item_totals[ncid // 10] += item_count
        data_length[ncid // 10] += item_count - test_subsample
        data_length['test'] += test_subsample
        #print('Count found for NCID ', ncid, ': ', item_count)

print('Item counts by BCID: ', item_totals, ', Data lengths by BCID: ', data_length)

for bcid in bcid_test_sample_set:
    num_indices = item_totals[bcid] - data_length[bcid]
    bcid_test_sample_set[bcid] = (np.random.permutation(item_totals[bcid]))[:num_indices]
    bcid_test_sample_set[bcid] = np.sort(bcid_test_sample_set[bcid])
    print('Indices sampled for BCID ', bcid, ': ', len(bcid_test_sample_set[bcid]))
    #print(bcid_test_sample_set[bcid])

for bcid in data_length:
    bcid_indices[bcid] = 0
    bcid_sample_indices[bcid] = 0
    bcid_process_totals[bcid] = 0
    masterdata[bcid] = {}
    masterdata[bcid]['data'] = np.zeros((data_length[bcid], img_size * img_size * 3), dtype=np.uint8)
    masterdata[bcid]['labels'] = np.zeros(data_length[bcid], dtype=np.uint8)
    print('BCID ', bcid, ': data array shape ', np.shape(masterdata[bcid]['data']),
          ', label array shape ', np.shape(masterdata[bcid]['labels']))

print('Loading and processing sprites...')

process_count = 0

if not os.path.exists(outpath):
    os.makedirs(outpath)

if not os.path.exists(testspritepath):
    os.makedirs(testspritepath)

def add_image(img, bcid, ncid):

    index_label = bcid

    if(bcid_sample_indices[bcid] < len(bcid_test_sample_set[bcid])
            and bcid_process_totals[bcid] == bcid_test_sample_set[bcid][bcid_sample_indices[bcid]]):
        index_label =  'test'
        bcid_sample_indices[bcid] += 1

    index = bcid_indices[index_label]

    if(index >= data_length[index_label]):
        print("ERROR, exceeded array limit at ID " + str(process_count) + ' with index label ' + index_label)
        return

    #Insert label.
    masterdata[index_label]['labels'][index] = ncid

    #Insert data.
    arr = np.array(img)
    len_channel = img_size * img_size

    #Red channel.
    masterdata[index_label]['data'][index][0:len_channel] = (arr[:,:,0]).flatten()
    #Green channel.
    masterdata[index_label]['data'][index][len_channel:(2*len_channel)] = (arr[:,:,1]).flatten()
    #Blue channel.
    masterdata[index_label]['data'][index][(2*len_channel):(3*len_channel)] = (arr[:,:,2]).flatten()

    bcid_indices[index_label] += 1
    bcid_process_totals[bcid] += 1

def test_reconstitute(bcid, index, filename):

    arr = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    len_channel = img_size * img_size

    arr[:,:,0] = np.reshape(masterdata[bcid]['data'][index][0:len_channel], (img_size, img_size))
    arr[:,:,1] = np.reshape(masterdata[bcid]['data'][index][len_channel:(2*len_channel)], (img_size, img_size))
    arr[:,:,2] = np.reshape(masterdata[bcid]['data'][index][(2*len_channel):(3*len_channel)], (img_size, img_size))

    test_img = Image.fromarray(arr, mode="RGB")
    test_img.save(os.path.join(testspritepath, filename))
    test_img.close()


for i in range(0, len(itemdata)):

    if i % 500 == 0:
        print('Processing ', i, ' of ', len(itemdata), ' sprites...')

    row = itemdata.iloc[i]

    filename = str(row['ID']) + '.png'
    filepath = os.path.join(spritepath, filename)

    bcid = int(row['BC'])
    ncid = int(row['NC'])

    img_base = Image.open(filepath)
    img_flip = img_base.transpose(Image.FLIP_LEFT_RIGHT)

    add_image(img_base, bcid, ncid)
    add_image(img_flip, bcid, ncid)

    if v_flippable[ncid]:

        img_vflip = img_base.transpose(Image.FLIP_TOP_BOTTOM)
        img_flip_vflip = img_flip.transpose(Image.FLIP_TOP_BOTTOM)

        add_image(img_vflip, bcid, ncid)
        add_image(img_flip_vflip, bcid, ncid)

        img_vflip.close()
        img_flip_vflip.close()

    img_base.close()
    img_flip.close()

    process_count += 1

#Try to reconstitute some images for testing.
'''
for bcid in bcid_list:

    if bcid_indices[bcid] == 0:
        continue

    #Exhaustive search.
    #for in in range(0, bcid_indices[bcid]):
    for i in range(0, 8):
        index = random.randint(0, bcid_indices[bcid] - 1)
        label = masterdata[bcid]['labels'][index]
        sample_filename = 'PrePickle-' + str(bcid) + '-' + str(index) + '-' + str(label) + '.png'
        test_reconstitute(bcid, index, sample_filename)
'''
print('Processed ', process_count, ' sprites.')
print('Index counts: ', bcid_indices, ', Sample index counts: ', bcid_sample_indices)

print('Pickling...')

batch_list = ['train', 'test']
train_batches = [1, 2, 3, 4]
exportdata = {}

exportdata['test'] = {}
exportdata['test']['data'] = np.copy(masterdata['test']['data'])
exportdata['test']['labels'] = np.copy(masterdata['test']['labels'])

exportdata['train'] = {}
exportdata['train']['data'] = np.zeros((0, img_size * img_size * 3), dtype=np.uint8)
exportdata['train']['labels'] = np.zeros(0, dtype=np.uint8)

train_index = 0

for bcid in bcid_list:

    if bcid == 'test':
        continue

    exportdata['train']['data'] = np.concatenate((exportdata['train']['data'], masterdata[bcid]['data']))
    exportdata['train']['labels'] = np.concatenate((exportdata['train']['labels'], masterdata[bcid]['labels']))

for batch in batch_list:

    shuffle(exportdata[batch]['data'], exportdata[batch]['labels'], random_state=0)

    print('Batch ', batch, ' data array shape ', np.shape(exportdata[batch]['data']),
          ', label array shape ', np.shape(exportdata[batch]['labels']))

    batch_filename = os.path.join(outpath, 'terraria-batch-' + batch)

    with open(batch_filename, 'wb+') as batch_file:
        pickle.dump(exportdata[batch], batch_file)

    print('Saved ', len(exportdata[batch]['labels']), ' samples in ', batch_filename)

#Unpickle and test data integrity.
print('Unpickling and testing...')

unpickledata = {}

def test_unpickle_img(batch, index, filename):

    arr = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    len_channel = img_size * img_size

    arr[:,:,0] = np.reshape(unpickledata[batch]['data'][index][0:len_channel], (img_size, img_size))
    arr[:,:,1] = np.reshape(unpickledata[batch]['data'][index][len_channel:(2*len_channel)], (img_size, img_size))
    arr[:,:,2] = np.reshape(unpickledata[batch]['data'][index][(2*len_channel):(3*len_channel)], (img_size, img_size))

    test_img = Image.fromarray(arr, mode="RGB")
    test_img.save(os.path.join(testspritepath, filename))
    test_img.close()

for batch in batch_list:

    batch_filename = os.path.join(outpath, 'terraria-batch-' + batch)

    with open(batch_filename, 'rb') as batch_file:
        unpickledata[batch] = pickle.load(batch_file)
        print(np.shape(unpickledata[batch]['data']), np.shape(unpickledata[batch]['labels']))

    print('Recovered ', len(unpickledata[batch]['labels']), ' items in ', batch_filename)

    for i in range(0, 16):
        index = random.randint(0, len(unpickledata[batch]['labels']) - 1)
        label = unpickledata[batch]['labels'][index]
        sample_filename = 'Unpickle-' + batch + '-' + str(index) + '-' + str(label) + '.png'
        test_unpickle_img(batch, index, sample_filename)

print('Unpickled and reconstituted sample sprites.')