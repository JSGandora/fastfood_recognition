'''
The following code takes preprocesses the image files for training the SVM
'''

from PIL import Image
import glob


# This function prepocesses a single file
def extract_data(filename, item, length):
    num_showed = 0
    data = open(filename, 'w')
    for file in glob.glob(str(item)+"/*.jpg"):
        img = Image.open(file)

        # crop the image into a length/2 x length/2 pixel file
        width = img.size[0]
        height = img.size[1]
        left = width/2 - length/2
        right = width/2 + length/2
        top = height/2 - length/2
        bottom = height/2 + length/2
        box = (left, top, right, bottom)
        img = img.crop(box)
        img.thumbnail((64, 64))

        # show image for debugging
        if 10 > num_showed:
            img.show()
            num_showed += 1

        # write the pixel data into the text file
        pixels = list(img.getdata())
        for pixel in pixels:
            data.write(str(pixel[0])+' '+str(pixel[1])+' '+str(pixel[2])+' ')
        data.write('\n')

    data.close()

extract_data('pizza.txt', 1, 1600)
extract_data('chicken.txt', 2, 1600)
extract_data('burger.txt', 3, 1600)
extract_data('burrito.txt', 4, 1600)