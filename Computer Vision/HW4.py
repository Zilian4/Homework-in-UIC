import cv2 as cv
import numpy as np


def MSE(img, img_reconstructed):
    mse = np.mean((img - img_reconstructed) ** 2)
    return mse


def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


path = './images/boat.jpg'
boat = cv.imread(path, cv.IMREAD_GRAYSCALE)
size = 16
top = 10

boat_array = np.array(boat).astype(dtype='float')  # (208, 320)
blocks = []  # 1040

# divide image into 1040 blocks
for i in range(0, len(boat_array), size):
    for j in range(0, len(boat_array[0]), size):
        block = boat_array[i:i + size, j:j + size]
        block = np.array(block)
        blocks.append(block)

# DCT
for i in range(len(blocks)):
    block = blocks[i]
    block = cv.dct(block)  # DCT
    # block = np.fft.fft2(block)  # DFT

    block = block.reshape(size ** 2)
    block_sort = np.sort(block)

    # pick top 10 numbers
    for j in range(len(block)):
        if block[j] < block_sort[-top:][0]:
            block[j] = 0

    block = block.reshape(size, size)
    block = cv.idct(block)  # IDCT
    # block = np.fft.ifft2(block)  # IDFT
    blocks[i] = block


# reconstruct
k = 0
boat_reconstructed = np.ones((208, 320))
for i in range(0, len(boat_reconstructed), size):
    for j in range(0, len(boat_reconstructed[0]), size):
        boat_reconstructed[i:i + size, j:j + size] = blocks[k]
        k += 1
boat_reconstructed = boat_reconstructed.astype("int")
cv.imwrite('./images/boat_reconstructed.jpg', boat_reconstructed)
print("MSE of the reconstructed image:", MSE(boat_array, boat_reconstructed))
