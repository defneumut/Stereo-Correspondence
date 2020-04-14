"""
Final Project: Stereo Correspondence
"""
import cv2
import numpy as np
import os
import datetime


def disparity(img1, img2, direction, window_size, pxls):

    w, h = img1.shape
    ssd = np.zeros((w, h, pxls))
    disparity = np.zeros((w, h))

    for i in range(pxls):
        ssd[:, :, i] = (img1 - np.roll(img2, i)) ** 2

    if direction == 0:
        for y in range(0, h - window_size):
            for x in range(0, w - window_size):
                ssd_k = ssd[x:(x+window_size), y:(y+window_size), :]
                ssd_k_sum = np.sum(np.sum(ssd_k, axis=0), axis=0)
                d = ssd_k_sum.argmin()
                disparity[x, y] = d
                output = disparity[:, 0:disparity.shape[1]*9/10]
    else:
        for y in range(0, h - window_size):
            for x in reversed(range( window_size, w)):
                ssd_k = ssd[(x - window_size + 1):(x + 1), y:(y + window_size), :]
                ssd_k_sum = np.sum(np.sum(ssd_k, axis=0), axis=0)
                d = ssd_k_sum.argmin()
                disparity[x, y] = d
                output = disparity[:, disparity.shape[1]/10:]

    return cv2.normalize(output, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)


def segmentation(image):

    image_new = np.float32(image.reshape((image.shape[0]*image.shape[1],1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    labels = cv2.kmeans(image_new, 5, criteria, 10, cv2.KMEANS_USE_INITIAL_LABELS)[1]
    output = labels.reshape((image.shape[0], image.shape[1]))
    output = cv2.normalize(output, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return output


if __name__ == '__main__':

    begin = datetime.datetime.now()

    left_img = cv2.imread('im2.png', 0) / 255.
    right_img = cv2.imread('im6.png', 0) / 255.

    # shrink the images for fast running
    # l_img = cv2.resize(left_img, (left_img.shape[1]/2, left_img.shape[0]/2), interpolation=cv2.INTER_CUBIC)
    # r_img = cv2.resize(right_img, (right_img.shape[1]/2, right_img.shape[0]/2), interpolation=cv2.INTER_CUBIC)

    # enlarge the images for better result
    l_img = cv2.resize(left_img, (left_img.shape[1]*2, left_img.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    r_img = cv2.resize(right_img, (right_img.shape[1]*2, right_img.shape[0]*2), interpolation=cv2.INTER_CUBIC)

    # l_img = cv2.resize(left_img, (900, 750), interpolation=cv2.INTER_CUBIC)
    # r_img = cv2.resize(right_img, (900, 750), interpolation=cv2.INTER_CUBIC)

    window_size = 10
    pxls = 100

    l_img_blur = cv2.GaussianBlur(l_img, (15,15), 1)
    r_img_blur = cv2.GaussianBlur(r_img, (15,15), 1)
    sharp_k = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    l_img_shape = cv2.filter2D(l_img_blur, -1, sharp_k)
    r_img_shape = cv2.filter2D(r_img_blur, -1, sharp_k)

    disparity_l = disparity(l_img_shape, r_img_shape, 0, window_size, pxls)
    disparity_r = disparity(l_img_shape, r_img_shape, 1, window_size, pxls)

    cv2.imwrite('output1.png', disparity_l)
    cv2.imwrite('output2.png', disparity_r)

    img1_seg = segmentation(disparity_l)
    img2_seg = segmentation(disparity_r)
    cv2.imwrite('img1_seg1.png', img1_seg)
    cv2.imwrite('img1_seg2.png', img2_seg)

    end = datetime.datetime.now()
    print "Running time:"
    print end - begin
