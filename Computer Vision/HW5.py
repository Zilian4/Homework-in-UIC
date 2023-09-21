import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./images/boat.jpg")
img = cv2.resize(img, (320, 208))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
Coefficients = pywt.dwt2(img, 'haar')

# cA:Low Frequency
# cH:Horizontal High Frequency
# cV:Vertical High Frequency
# cD:Diagonal High Frequency
cA, (cH, cV, cD) = Coefficients

AH = np.concatenate([cA, cH+255], axis=1)
VD = np.concatenate([cV+255, cD+255], axis=1)
subband_images = np.concatenate([AH, VD], axis=0)

re_img = pywt.idwt2(Coefficients, 'haar')

plt.imshow(subband_images, 'gray')
plt.title('subband_images')
plt.show()




