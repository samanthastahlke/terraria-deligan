import os
import pandas
import time
import urllib.request
from PIL import Image


datapath = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'terraria_items.csv'))
spritepath = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Sprites\\Base\\'))

if not os.path.exists(spritepath):
    os.makedirs(spritepath)

print('Extracting sprites from ' + datapath + '...')
data = pandas.read_csv(datapath, dtype=str)

for i in range(0, len(data)):

    if i % 100 == 0:
        print('Downloaded ' + str(i) + ' images...')

    row = data.iloc[i]

    with open(os.path.join(spritepath, row['ID'] + '.png'), 'wb+') as img_file:

        resp_ok = False
        err_count = 0

        while not resp_ok:

            try:
                url_data = urllib.request.urlopen(row['Image'])
            except urllib.request.HTTPError as err:
                print("Encountered HTTP error, waiting...")
                err_count += 1

                if err_count > 4:
                    print("Image download failed, ID = " + row['ID'])
                    break

                time.sleep(1)
                continue

            resp_ok = True
            img = Image.open(url_data)
            img.save(img_file)

    time.sleep(0.1)
