import cv2 as cv
import numpy as np


# show the image
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def MF(img_path, kernel_size):
    img = cv.imread(img_path)
    if img is None:
        print('There is no image!')

    # show original image
    cv_show('original_img', img)

    # median filtering
    img_performed = cv.medianBlur(img, kernel_size)
    cv.imwrite(r'.\images\pikachu_median filtering.jpg', img_performed)
    cv_show("pikachu_without noise(kernel_size={0})".format(kernel_size), img_performed)


def Conv_h(img_path):
    img = cv.imread(img_path)
    if img is None:
        print('There is no image!')
    #     convolution with “h” in Q3
    filter_kernel = np.array([[1 / 4, 1 / 4],
                              [1 / 4, 1 / 4]])
    img_convolved = cv.filter2D(img, ddepth=-1, kernel=filter_kernel)

    cv.imwrite(r'.\images\pikachu_with convolved.jpg', img_convolved)
    cv_show("pikachu_convolved", img_convolved)


'''The image output results show that the median filter of 3x3 has a good clear effect on the noise,
 and the grayscale of the noise point is usually far from the grayscale of the neighboring pixel points,
 there will be a small probability that the median value will be replaced by the noise point'''
