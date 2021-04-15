import numpy
from skimage import io
import tensorflow as tf
import cv2

#def log10(x):
    #numerator = tf.log(x)
    #denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    #return numerator / denominator
#def psnr(im1, im2):
"""
img_arr1 = numpy.array(im1).astype('float32')
    img_arr2 = numpy.array(im2).astype('float32')
    mse = tf.reduce_mean(tf.squared_difference(img_arr1, img_arr2))
    psnr = tf.constant(255**2, dtype=tf.float32)/mse
    result = tf.constant(10, dtype=tf.float32)*log10(psnr)
    with tf.Session():
        result = result.eval()
    return result
psnr(img1,img2)"""
import cv2
import os
from skimage.measure import compare_ssim

img1=cv2.imread('mixed pics/covid19_01.png')
img2=cv2.imread('mixed pics/painting_covid19_01.png')
#sr_dir = os.listdir('mixed pics/covid19_01.png')
#hr_dir = os.listdir('mixed pics/painting_covid19_01.png')

psnr = 0.0
ssim = 0.0
n = 0

def to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

psnr = cv2.PSNR(cv2.imread('mixed pics/covid19_01.png' ), cv2.imread('mixed pics/painting_covid19_01.png'))
ssim = compare_ssim(to_grey(cv2.imread('mixed pics/covid19_01.png' ),to_grey(cv2.imread('mixed pics/painting_covid19_01.png' )))


print(psnr)
print(ssim)


