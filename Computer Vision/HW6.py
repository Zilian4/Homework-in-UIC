import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./images/chess.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Prewitt
def Prewitt(img_gray):
    # kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    # kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    kernelx = np.array([[1, 0, -1]], dtype=int)
    kernely = kernelx.T
    x = cv2.filter2D(img_gray, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img_gray, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    return Prewitt


# Sobel
def Sobel(img_gray):
    x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Sobel


def Highpass(img_gray):
    kernely = np.array([[-0.1717, 0.5, -0.1717]])
    kernelx = kernely.T
    x = cv2.filter2D(img_gray, -1, kernel=kernelx)
    y = cv2.filter2D(img_gray, -1, kernel=kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    highpass = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return highpass


img_Prewitt = Prewitt(img_gray)
img_Sobel = Sobel(img_gray)
img_highpass = Highpass(img_gray)
cv2.imwrite('./images/chess_Prewitt.jpg',img_Prewitt)
cv2.imwrite('./images/chess_Sobel.jpg',img_Sobel)
cv2.imwrite('./images/chess_highpass.jpg',img_highpass)

plt.imshow(img_highpass, 'gray')
plt.title('HP')
plt.show()
plt.imshow(img_Sobel, 'gray')
plt.title('Sobel')
plt.show()
plt.imshow(img_Prewitt, 'gray')
plt.title('Prewitt')
plt.show()
