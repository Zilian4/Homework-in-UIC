import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi

def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def decimation(img_path):
    img = cv.imread(img_path)
    if img is None:
        raise Exception("There is no img")
    img = np.array(img)
    cv_show('shirt', img)
    kernel = np.array([[1 / 16, 1 / 8, 1 / 16],
                       [1 / 8, 1 / 4, 1 / 8],
                       [1 / 16, 1 / 8, 1 / 16]])

    img_convolved = cv.filter2D(img, ddepth=-1, kernel=kernel)
    print(img_convolved.shape)
    # cv_show('img_convolved', img_convolved)

    img_decimated = []
    print(len(img_convolved[0]), len(img_convolved))
    for r in range(len(img_convolved)):
        for c in range(len(img_convolved[0])):
            if r % 2 == 0 and c % 2 == 0:
                img_decimated.append(img_convolved[r][c])
    img_decimated = np.reshape(img_decimated,( 1200, 1140, 3))
    cv_show('shirt_decimated', img_decimated)
    # cv.imwrite(r'.\images\shirt_with_1.5factor.jpg', img_decimated)


def interpolate_2(img_path):
    img = cv.imread(img_path)
    img_horizontal_inter = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            img_horizontal_inter.append(img[i][j])
            if j < 379:
                greyscale = img[i][j] / 2 + img[i][j + 1] / 2
                img_horizontal_inter.append(greyscale)
            else:
                img_horizontal_inter.append(img[i][j])

    img_vertical_inter = np.reshape(img_horizontal_inter, (400, 760, 3)).astype(dtype="uint8")

    img_interpolated = []
    for j in range(len(img_vertical_inter[0])):
        for i in range(len(img_vertical_inter)):
            img_interpolated.append(img_vertical_inter[i][j])
            if i < 399:
                greyscale = img_vertical_inter[i][j] / 2 + img_vertical_inter[i + 1][j] / 2
                img_interpolated.append(greyscale)
            else:
                img_interpolated.append(img_vertical_inter[i][j])

    img_interpolated = np.reshape(img_interpolated, (760, 800, 3)).astype(dtype="uint8")
    img_interpolated = img_interpolated.transpose([1, 0, 2])
    print(np.shape(img_interpolated))
    cv_show('img_interpolated', img_interpolated)
    # cv.imwrite(r'.\images\shirt_interpolated.jpg', img_interpolated)


def interpolate_3(img_path):
    img = cv.imread(img_path)
    img_horizontal_inter = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            img_horizontal_inter.append(img[i][j])
            # print(j)
            if j < len(img[0]) - 1:
                greyscale_1 = img[i][j] * 2 / 3 + img[i][j + 1] * 1 / 3
                greyscale_2 = img[i][j] * 1 / 3 + img[i][j + 1] * 2 / 3
                img_horizontal_inter.append(greyscale_1)
                img_horizontal_inter.append(greyscale_2)
            else:
                img_horizontal_inter.append(img[i][j])
                img_horizontal_inter.append(img[i][j])

    img_vertical_inter = np.reshape(img_horizontal_inter, (800, 2280, 3)).astype(dtype="uint8")
    print(np.shape(img_vertical_inter))
    img_interpolated = []
    for j in range(len(img_vertical_inter[0])):
        for i in range(len(img_vertical_inter)):
            img_interpolated.append(img_vertical_inter[i][j])
            if i < len(img) - 1:
                greyscale_1 = img_vertical_inter[i][j] * 2 / 3 + img_vertical_inter[i + 1][j] * 1 / 3
                greyscale_2 = img_vertical_inter[i][j] * 1 / 3 + img_vertical_inter[i + 1][j] * 2 / 3
                img_interpolated.append(greyscale_1)
                img_interpolated.append(greyscale_2)
            else:
                img_interpolated.append(img_vertical_inter[i][j])
                img_interpolated.append(img_vertical_inter[i][j])

    img_interpolated = np.reshape(img_interpolated, (2280, 2400, 3)).astype(dtype="uint8")
    img_interpolated = img_interpolated.transpose([1, 0, 2])
    print(np.shape(img_interpolated))
    cv_show('img_interpolated', img_interpolated)
    # cv.imwrite('./images/shirt_1.5.jpg', img_interpolated)


def plot_2_D_frequency_response():
    fig = plt.figure(num=1)
    ax = Axes3D(fig)
    w1 = np.arange(- pi, pi, 0.25)
    w2 = np.arange(- pi, pi, 0.25)
    w1, w2 = np.meshgrid(w1, w2)
    intensity_1 = (1 + np.cos(w1) + np.cos(w2) + np.cos(w2) * np.cos(w1)) / 4
    intensity_2 = (1+2*(np.cos(w1+w2)+np.cos(w1)+np.cos(w2)+np.cos(w1-w2)))
    intensity_3 = (0.1*np.cos(2*w1)+0.5*np.cos(w1)+0.4)*(0.1*np.cos(2*w2)+0.5*np.cos(w2)+0.4)
    intensity = intensity_3
    ax.plot_surface(w1, w2, intensity, rstride=1, cstride=1, edgecolor='black', cmap=plt.get_cmap('rainbow'))
    ax.contourf(w1, w2, intensity, zdir='z', offset=-2, cmap='rainbow')
    plt.show()
