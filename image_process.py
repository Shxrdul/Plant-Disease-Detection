import numpy as np
import cv2
import matplotlib.pyplot as plot


def brightness_distortion(I, mu, sigma):
    return np.sum(I * mu / sigma ** 2, axis=-1) / np.sum((mu / sigma) ** 2, axis=-1)


def chromacity_distortion(I, mu, sigma):
    alpha = brightness_distortion(I, mu, sigma)[..., None]
    return np.sqrt(np.sum(((I - alpha * mu) / sigma) ** 2, axis=-1))


def mask(img):
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    val = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    sat = cv2.medianBlur(sat, 11)
    val = cv2.medianBlur(val, 11)
    thresh_S = cv2.adaptiveThreshold(sat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 401, 10);

    mean_S, stdev_S = cv2.meanStdDev(img, mask=255 - thresh_S)
    mean_S = mean_S.ravel().flatten()
    stdev_S = stdev_S.ravel()
    chrom_S = chromacity_distortion(img, mean_S, stdev_S)
    chrom255_S = cv2.normalize(chrom_S, chrom_S, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)[:, :,
                 None]
    thresh2_S = cv2.adaptiveThreshold(chrom255_S, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 401, 10);
    img2 = cv2.bitwise_and(img, img, mask=thresh2_S)
    return img2


def imgprocess():
    img1 = cv2.imread('classify/test3.jpg')
    img2 = mask(img1)
    img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img4 = cv2.equalizeHist(img3)
    # return img4

    images = [img1, img2, img3, img4]
    titles = ['Original Image', 'Masked', 'Grayscale', 'Histo Equilize']
    for i in range(4):
        plot.subplot(2, 2, i + 1),
        if i == 0:
            plot.imshow(images[i])
        else:
            plot.imshow(images[i], cmap='gray')
        plot.title(titles[i])
        plot.xticks([]), plot.yticks([])
    plot.show()


imgprocess()
