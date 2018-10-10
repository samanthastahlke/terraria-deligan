import os
import time
from PIL import Image

outfolder = '64Uncut\\'

excludeLarger = False
outsize = 64

spritehome = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Sprites\\'))
datapath = os.path.abspath(os.path.join(spritehome, 'Base\\'))
outpath = os.path.abspath(os.path.join(spritehome, outfolder))

def process_image(imgpath):

    with Image.open(imgpath).convert("RGBA") as img:

        w, h = img.size

        #Exclude if the image is too big.

        if (w > outsize or h > outsize):

            if excludeLarger:
                #print("Excluded image ", imgpath)
                return False

            new_w = w
            new_h = h

            if w > h:
                new_w = outsize
                new_h = int((h / w) * outsize)
            else:
                new_h = outsize
                new_w = int((w / h) * outsize)

            w = new_w
            h = new_h

            img.thumbnail((new_w, new_h))

        #Required for the mask during paste process.
        img.load()

        imgoutpath = os.path.join(outpath, filename)

        with open(imgoutpath, 'wb+') as imgout:

            newimg = Image.new("RGB", (outsize, outsize), (255, 255, 255))

            #Paste the sprite into the centre of the new image.
            newimg.paste(img, mask=img.split()[3],
                         box=(outsize // 2 - w // 2, outsize // 2 - h // 2))

            newimg.save(imgout)
            newimg.close()
            return True

print('Converting sprites from ', spritehome, ' at ', outsize, 'px, saving to ', outpath)

images = sorted([filename for filename in os.listdir(datapath)],
                key=lambda f: int(f.rsplit(os.path.extsep, 1)[0]))

imgcount = 0
savecount = 0

if not os.path.exists(outpath):
    os.makedirs(outpath)

for filename in images:

    imgfile = os.path.join(datapath, filename)

    if process_image(imgfile):
        savecount += 1

    imgcount += 1

    if imgcount % 100 == 0:
        print("Processed ", imgcount, " images...")

print("Processed ", imgcount, " images in total.")
print("Saved ", savecount, " images.")


