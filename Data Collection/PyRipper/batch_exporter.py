import pickle
from PIL import Image
import os
import pandas
import math
import numpy as np
import random

img_size = 32

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
data_length = {}
bcid_indices = {}

for i in range(0, len(classdata) - 1):

    row = classdata.iloc[i]

    if not math.isnan(row['BCID']):
        bcid = int(row['BCID'])
        bcid_list.append(bcid)
        data_length[bcid] = 0
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
        data_length[ncid // 10] += item_count
        #print('Count found for NCID ', ncid, ': ', item_count)

#print('Data lengths by BCID: ', data_length)

for bcid in data_length:
    bcid_indices[bcid] = 0
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

    index = bcid_indices[bcid]

    if(index >= data_length[bcid]):
        print("ERROR, exceeded array limit at ID " + str(process_count))
        return

    #Insert label.
    masterdata[bcid]['labels'][index] = ncid

    #Insert data.
    arr = np.array(img)
    len_channel = img_size * img_size

    #Red channel.
    masterdata[bcid]['data'][index][0:len_channel] = (arr[:,:,0]).flatten()
    #Green channel.
    masterdata[bcid]['data'][index][len_channel:(2*len_channel)] = (arr[:,:,1]).flatten()
    #Blue channel.
    masterdata[bcid]['data'][index][(2*len_channel):(3*len_channel)] = (arr[:,:,2]).flatten()

    bcid_indices[bcid] += 1

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

print('Pickling...')

for bcid in bcid_list:

    batch_filename = os.path.join(outpath, 'terraria-batch-' + str(bcid))

    with open(batch_filename, 'wb+') as batch_file:
        pickle.dump(masterdata[bcid], batch_file)

    print('Saved ', bcid_indices[bcid], ' samples in ', batch_filename)

print('Pickled and saved.')

#Unpickle and test data integrity.
print('Unpickling and testing...')

unpickledata = {}

def test_unpickle_img(bcid, index, filename):

    arr = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    len_channel = img_size * img_size

    arr[:,:,0] = np.reshape(unpickledata[bcid]['data'][index][0:len_channel], (img_size, img_size))
    arr[:,:,1] = np.reshape(unpickledata[bcid]['data'][index][len_channel:(2*len_channel)], (img_size, img_size))
    arr[:,:,2] = np.reshape(unpickledata[bcid]['data'][index][(2*len_channel):(3*len_channel)], (img_size, img_size))

    test_img = Image.fromarray(arr, mode="RGB")
    test_img.save(os.path.join(testspritepath, filename))
    test_img.close()

for bcid in bcid_list:

    batch_filename = os.path.join(outpath, 'terraria-batch-' + str(bcid))

    with open(batch_filename, 'rb') as batch_file:
        unpickledata[bcid] = pickle.load(batch_file)

    print('Recovered ', len(unpickledata[bcid]['labels']), ' items in ', batch_filename)

    for i in range(0, 8):
        index = random.randint(0, len(unpickledata[bcid]['labels']) - 1)
        label = unpickledata[bcid]['labels'][index]
        sample_filename = 'Unpickle-' + str(bcid) + '-' + str(index) + '-' + str(label) + '.png'
        test_unpickle_img(bcid, index, sample_filename)

print('Unpickled and reconstituted sample sprites.')