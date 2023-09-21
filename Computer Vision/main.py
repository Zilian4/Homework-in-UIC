# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from HW2 import *

if __name__ == '__main__':
    img_path = r'.\images\boat.jpg'
    # decimation(img_path)
    # interpolate_3(r'.\images\shirt.jpg')
    img = cv.imread(img_path,cv.IMREAD_GRAYSCALE).astype(dtype=float)
    img_dct = cv.dct(img)
    img = cv.idct(img_dct).astype('uint8')
    cv_show('sds',img)
