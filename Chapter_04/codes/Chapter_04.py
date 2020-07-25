# Chapter 4: Image Enhancement and Face Recognition

# Author: Sandipan Dey

# Image Enhancement Filters with PIL for noise removal and smoothing

# BLUR Filter to remove Salt & Pepper Noise

import numpy as np
import matplotlib.pylab as plt
from PIL import Image, ImageFilter
from copy import deepcopy

def plot_image(image, title=None, sz=20):
    plt.imshow(image)
    plt.title(title, size=sz)
    plt.axis('off')

def add_noise(im, prop_noise, salt=True, pepper=True):
    im = deepcopy(im)
    n = int(im.width * im.height * prop_noise)
    x, y = np.random.randint(0, im.width, n), np.random.randint(0, im.height, n)
    for (x,y) in zip(x,y):
        im.putpixel((x, y),         # geenrate salt-and-pepper noise
        ((0,0,0) if np.random.rand() < 0.5 else (255,255,255)) if salt and pepper \
        else (255,255,255) if salt \
        else (0, 0, 0)) # if pepper
    return im

orig = Image.open('images/Img_04_01.jpg')
i = 1
plt.figure(figsize=(12,35))
for prop_noise in np.linspace(0.05,0.3,6):
 # choose random locations inside image
 im = add_noise(orig, prop_noise)
 plt.subplot(6,2,i), plot_image(im, 'Original Image with ' + str(int(100*prop_noise)) + '% added noise')
 im1 = im.filter(ImageFilter.BLUR)
 plt.subplot(6,2,i+1), plot_image(im1, 'Blurred Image')
 i += 2

plt.show()

# Gaussian BLUR Filter to remove Salt & Pepper Noise

im = Image.open('images/Img_04_01.jpg')
im = add_noise(im, prop_noise = 0.2)

plt.figure(figsize=(20,15))
i = 1
for radius in np.linspace(1, 3, 12): 
    im1 = im.filter(ImageFilter.GaussianBlur(radius))
    plt.subplot(3,4,i)
    plot_image(im1, 'radius = ' + str(round(radius,2)))
    i += 1
plt.suptitle('PIL Gaussian Blur with different Radius', size=30)
plt.show()

#  Median Filter to remove Salt & Pepper Noise

im = Image.open('images/Img_04_02.jpg')
im = add_noise(im, prop_noise = 0.1)

plt.figure(figsize=(20,10))
plt.subplot(1,4,1)
plot_image(im, 'Input noisy image')
i = 2
for sz in [3,7,11]:
 im1 = im.filter(ImageFilter.MedianFilter(size=sz)) 
 plt.subplot(1,4,i), plot_image(im1, 'Output (Filter size=' + str(sz) + ')', 20)
 i += 1
plt.tight_layout()
plt.show()

# Max, Min and Mode filters to remove outliers from image

orig = Image.open('images/Img_04_11.jpg')
im = add_noise(orig, prop_noise = 0.2, pepper=False)

plt.figure(figsize=(20,10))
plt.subplot(1,4,1)
plot_image(im, 'Input noisy image')
i = 2
for sz in [3,7,11]:
 im1 = im.filter(ImageFilter.MinFilter(size=sz)) 
 plt.subplot(1,4,i), plot_image(im1, 'Output (Filter size=' + str(sz) + ')')
 i += 1
plt.tight_layout()
plt.show()

im = add_noise(orig, prop_noise = 0.3, salt=False)

plt.figure(figsize=(20,10))
plt.subplot(1,4,1)
plot_image(im, 'Input noisy image')
i = 2
for sz in [3,7,11]:
 im1 = im.filter(ImageFilter.MaxFilter(size=sz)) 
 plt.subplot(1,4,i), plot_image(im1, 'Output (Filter size=' + str(sz) + ')')
 i += 1
plt.show()

orig = Image.open('images/Img_04_20.jpg')
im = add_noise(orig, prop_noise = 0.1)

plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plot_image(im, 'Input noisy image', 25)
i = 2
for sz in [3,5]:
 im1 = im.filter(ImageFilter.ModeFilter(size=sz)) 
 plt.subplot(1,3,i), plot_image(im1, 'Output (Filter size=' + str(sz) + ')', 25)
 i += 1
plt.tight_layout()
plt.show()

# Progressive Application of Gaussian Blur, Median, Mode and Max Filters on an image

im = Image.open('images/Img_04_02.jpg')
plt.figure(figsize=(10,15))
plt.subplots_adjust(0,0,1,0.95,0.05,0.05)
im1 = im.copy()
sz = 5
for i in range(8):
 im1 = im1.filter(ImageFilter.GaussianBlur(radius=sz)) 
 if i % 2 == 0:
    plt.subplot(4,4,4*i//2+1), plot_image(im1, 'Gaussian Blur' if i == 0 else None, 25)
im1 = im.copy()
for i in range(8):
 im1 = im1.filter(ImageFilter.MedianFilter(size=sz)) 
 if i % 2 == 0:
    plt.subplot(4,4,4*i//2+2), plot_image(im1, 'Median' if i == 0 else None, 25)
im1 = im.copy()
for i in range(8):
 im1 = im1.filter(ImageFilter.ModeFilter(size=sz)) 
 if i % 2 == 0:
    plt.subplot(4,4,4*i//2+3), plot_image(im1, 'Mode' if i == 0 else None, 25)
im1 = im.copy()
for i in range(8):
 im1 = im1.filter(ImageFilter.MaxFilter(size=sz)) 
 if i % 2 == 0:
    plt.subplot(4,4,4*i//2+4), plot_image(im1, 'Max' if i == 0 else None, 25)
plt.show()

# Unsharp masking to Sharpen an Image

# With scikit-image filters module

import numpy as np
import matplotlib.pylab as plt
from skimage.io import imread
from skimage.filters import unsharp_mask

im = imread('images/Img_04_04.jpg')
im1 = unsharp_mask(im, radius=1, amount=1)
im2 = unsharp_mask(im, radius=5, amount=2)
im3 = unsharp_mask(im, radius=20, amount=3)

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(20, 12))
axes = axes.ravel()
axes[0].set_title('Original image', size=20), axes[0].imshow(im)
axes[1].set_title('Enhanced image, radius=1, amount=1.0', size=20), axes[1].imshow(im1)
axes[2].set_title('Enhanced image, radius=5, amount=2.0', size=20), axes[2].imshow(im2)
axes[3].set_title('Enhanced image, radius=20, amount=3.0', size=20), axes[3].imshow(im3)
for ax in axes:
 ax.axis('off')
fig.tight_layout()
plt.show()

# With PIL ImageFilter module

from PIL import Image, ImageFilter
im = Image.open('images/Img_04_05.jpg')

plt.figure(figsize=(15,16))
plt.subplot(221), plot_image(im, 'original')
im1 = im.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
plt.subplot(222), plot_image(im1, 'unsharp masking, raidus=2, percent=150')
im1 = im.filter(ImageFilter.UnsharpMask(radius=5, percent=200))
plt.subplot(223), plot_image(im1, 'unsharp masking, raidus=5, percent=200')
im1 = im.filter(ImageFilter.UnsharpMask(radius=10, percent=250))
plt.subplot(224), plot_image(im1, 'unsharp masking, raidus=10, percent=250')
plt.tight_layout()
plt.show()

# Laplacian Sharpening with SimpleITK

import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt

image = sitk.ReadImage('images/Img_04_20.jpg', sitk.sitkFloat32)

filt = sitk.UnsharpMaskImageFilter() 
filt.SetAmount(1.5) # typically set between 1 and 2
filt.SetSigmas(0.15)
sharpened = filt.Execute(image)

np_image = sitk.GetArrayFromImage(image)
np_image = np_image / np_image.max()
np_sharpened = sitk.GetArrayFromImage(sharpened)
np_sharpened = np_sharpened / np_sharpened.max()

plt.figure(figsize=(20,10))
plt.gray()
plt.subplots_adjust(0,0,1,1,0.05,0.05)
plt.subplot(121), plot_image(np_image, 'Original Image')
plt.subplot(122), plot_image(np_sharpened, 'Sharpened Image (with UnsharpMask)')
plt.show()

# Implementing Unsharp Mask with opencv-python

import cv2

im = cv2.imread("images/Img_04_13.png")
im_smoothed = cv2.GaussianBlur(im, (11,11), 10, 10)

im1 = cv2.addWeighted(im, 1.0 + 3.0, im_smoothed, -3.0, 0) # im1 = im + 3.0*(im - im_smoothed)

plt.figure(figsize=(20,25))
plt.subplots_adjust(0,0,1,0.95,0.05,0.05)
plt.subplot(211), plot_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 'Original Image')
plt.subplot(212), plot_image(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB), 'Sharpened Image')
plt.show()

# Averaging of Images to remove Random Noise

from skimage import img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread
import matplotlib.pylab as plt
import numpy as np

im = img_as_float(imread('images/Img_04_06.jpg')) # original image
n = 100
images = np.zeros((n, im.shape[0], im.shape[1], im.shape[2]))
sigma = 0.2
for i in range(n):
    images[i,...] = random_noise(im, var=sigma**2)

im_mean = images.mean(axis=0)
im_median = np.median(images, axis=0)

plt.figure(figsize=(10,10))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
plt.subplot(221), plot_image(im, 'Original image')
plt.subplot(222), plot_image(images[0], 'Noisy PSNR: ' + str(round(peak_signal_noise_ratio(im, images[0]),3)))
plt.subplot(223), plot_image(im_mean, 'Mean PSNR: ' + str(round(peak_signal_noise_ratio(im, im_mean),3)))
plt.subplot(224), plot_image(im_median, 'Median PSNR: ' + str(round(peak_signal_noise_ratio(im, im_median),3)))
plt.show()

plt.figure(figsize=(10,5))
plt.hist(images[:,100,100,0], color='red', alpha=0.2, label='red')
plt.hist(images[:,100,100,1], color='green', alpha=0.2, label='green')
plt.hist(images[:,100,100,2], color='blue', alpha=0.2, label='blue')
plt.vlines(im[100,100,0], 0, 20, color='red', label='original')
plt.vlines(im[100,100,1], 0, 20, color='green', label='original')
plt.vlines(im[100,100,2], 0, 20, color='blue', label='original')
plt.vlines(im_mean[100,100,0], 0, 20, color='red', linestyles='dashed', label='estimated')
plt.vlines(im_mean[100,100,1], 0, 20, color='green', linestyles='dashed', label='estimated')
plt.vlines(im_mean[100,100,2], 0, 20, color='blue', linestyles='dashed', label='estimated')
plt.legend()
plt.grid()
plt.show()

# Image Denoising with Curvature-Driven Algorithms

import SimpleITK as sitk
import matplotlib.pylab as plt

img = sitk.ReadImage('images/Img_04_11.png', sitk.sitkFloat64)

normfilter = sitk.NormalizeImageFilter()
caster = sitk.CastImageFilter()
caster.SetOutputPixelType(sitk.sitkFloat64)
        
tkfilter = sitk.ShotNoiseImageFilter()
tkfilter.SetScale(0.2)
img_noisy = tkfilter.Execute (img)
img_noisy = sitk.RescaleIntensity(img_noisy)

tkfilter = sitk.CurvatureFlowImageFilter() 
tkfilter.SetNumberOfIterations(50)
tkfilter.SetTimeStep(0.1)
img_res_TK = tkfilter.Execute(img_noisy)

tkfilter = sitk.MinMaxCurvatureFlowImageFilter() 
tkfilter.SetNumberOfIterations(50)
tkfilter.SetTimeStep(0.1)
tkfilter.SetStencilRadius(4)
img_res_TK1 = tkfilter.Execute(img_noisy)
img_res_TK1 = sitk.RescaleIntensity(img_res_TK1)

tkfilter = sitk.CurvatureAnisotropicDiffusionImageFilter()
tkfilter.SetNumberOfIterations(100);
tkfilter.SetTimeStep(0.05);
tkfilter.SetConductanceParameter(3);
img_res_TK2 = tkfilter.Execute(img_noisy)

tkfilter = sitk.GradientAnisotropicDiffusionImageFilter()
tkfilter.SetNumberOfIterations(100);
tkfilter.SetTimeStep(0.05);
tkfilter.SetConductanceParameter(3);
img_res_TK3 = tkfilter.Execute(img_noisy)

plt.figure(figsize=(16,20))
plt.gray()
plt.subplots_adjust(0,0,1,1,0.01,0.05)
plt.subplot(321), plt.imshow(sitk.GetArrayFromImage(img)), plt.axis('off'), plt.title('Original', size=20)
plt.subplot(322), plt.imshow(sitk.GetArrayFromImage(img_noisy)), plt.axis('off'), plt.title('Noisy (with added Shot Noise)', size=20)
plt.subplot(323), plt.imshow(sitk.GetArrayFromImage(img_res_TK)), plt.axis('off'), plt.title('Denoised (with CurvatureFlowImageFilter)', size=20)
plt.subplot(324), plt.imshow(sitk.GetArrayFromImage(img_res_TK1)), plt.axis('off'), plt.title('Denoised (with MinMaxCurvatureFlowImageFilter)', size=20)
plt.subplot(325), plt.imshow(sitk.GetArrayFromImage(img_res_TK2)), plt.axis('off'), plt.title('Denoised (with CurvatureAnisotropicDiffusionImageFilter)', size=20)
plt.subplot(326), plt.imshow(sitk.GetArrayFromImage(img_res_TK3)), plt.axis('off'), plt.title('Denoised (with GradientAnisotropicDiffusionImageFilter)', size=20)
plt.show()

# Contrast Strectching / Histogram Equalization with opencv-python

import numpy as np
import matplotlib.pylab as plt
import cv2

def plot_hist(img, col='r'):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = col)
    plt.hist(img.flatten(),256,[0,256], color = col, alpha = 0.1)
    plt.xlim([0,256])
    plt.title('CDF and histogram of the color channels', size=20)
    #plt.legend(('cdf','histogram'), loc = 'upper left')
    return bins, cdf

def plot_img_hist(img, title):
    plt.figure(figsize=(20,10))
    plt.subplot(121), plot_image(img, title)
    plt.subplot(122), plot_hist(img[...,0], 'r'), plot_hist(img[...,1], 'g'), plot_hist(img[...,2], 'b')
    plt.show()

img = cv2.imread('images/Img_04_07.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img2 = img.copy()
for i in range(3):
    hist,bins = np.histogram(img[...,i].flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    #cdf_m = 255 * cdf / cdf[-1] # normalize
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2[...,i] = cdf[img[...,i]]
    # use linear interpolation of cdf to find new pixel values
    #img2[...,i] = np.reshape(np.interp(img[...,i].flatten(),bins[:-1],cdf), img[...,i].shape)

img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
equ = img_lab.copy()
equ[...,0] = cv2.equalizeHist(equ[...,0])
equ = np.clip(cv2.cvtColor(equ, cv2.COLOR_LAB2RGB), 0, 255)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = img_lab.copy()
cl[...,0] = clahe.apply(cl[...,0])
cl = np.clip(cv2.cvtColor(cl, cv2.COLOR_LAB2RGB), 0, 255)

plot_img_hist(img, 'Original Image')

plot_img_hist(img2, 'Hist. Equalized')

plot_img_hist(equ, 'Hist. Equalized (LAB space)')

plot_img_hist(cl, 'Adaptive Hist. Equalized (LAB space)')

# Fingerprint Cleaning and Minutiaes extraction

# Fingerprint Cleaning with Morphological operations

from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pylab as plt
from skimage.morphology import binary_opening, binary_closing, skeletonize, square
from scipy.ndimage import morphological_gradient
from skimage.filters import threshold_otsu

im = rgb2gray(imread('images/Img_04_09.jpg'))
im[im <= 0.5] = 0 # binarize
im[im > 0.5] = 1
im_o = binary_opening(im, square(2))
im_c = binary_closing(im, square(2))
im_oc = binary_closing(binary_opening(im, square(2)), square(3))
im_s = skeletonize(im_oc)
im_g = morphological_gradient(im_oc.astype(np.uint8), size=(2,2))

plt.figure(figsize=(20,12))
plt.gray()
plt.subplot(231), plot_image(im, 'original')
plt.subplot(232), plot_image(im_o, 'opening')
plt.subplot(233), plot_image(im_c, 'closing')
plt.subplot(234), plot_image(im_oc, 'opening + closing')
plt.subplot(235), plot_image(im_s, 'skeletonizing')
plt.subplot(236), plot_image(im_g, 'morphological gradient')
plt.show()

# Feature (Minutiaes) extraction from an enhanced fingerprint

from PIL import Image, ImageDraw

cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

def minutiae_at(pixels, i, j):
    
    values = [pixels[i + k][j + l] for k, l in cells]
    crossings = 0
    for k in range(0, 8):
        crossings += abs(values[k] - values[k + 1])
    crossings /= 2
    if pixels[i][j] == 1:
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"
    return "none"

def calculate_minutiaes(im):
    
    pixels = 255 - np.array(im).T
    pixels = 1.0*(pixels > 10)   
    (x, y) = im.size
    result = im.convert("RGB")
    draw = ImageDraw.Draw(result)
    colors = {"ending" : (150, 0, 0), "bifurcation" : (0, 150, 0)}
    ellipse_size = 2
    for i in range(1, x - 1):
        for j in range(1, y - 1):
            minutiae = minutiae_at(pixels, i, j)
            if minutiae != "none":
                draw.ellipse([(i - ellipse_size, j - ellipse_size), (i + ellipse_size, j + ellipse_size)], outline = colors[minutiae])
    del draw
    return result

im = Image.open('images/Img_04_10.jpg').convert("L") # covert to grayscale
out = calculate_minutiaes(im)
plt.figure(figsize=(15,12))
plt.gray()
plt.subplot(121), plot_image(im, 'input thinned')
plt.subplot(122), plot_image(out, 'with minutiaes extracted')
plt.show()

# Edge Detection with LOG / Zero-Crossing, Canny vs. Holistically-Nested

# Computing the Image Derivatives

from scipy.signal import convolve
from skimage.io import imread
from skimage.color import rgb2gray

img = rgb2gray(imread('images/Img_04_38.png'))
h, w = img.shape

kd1 = [[1, -1]]
kd2 = [[1, -2, 1]]
imgd1 = convolve(img, kd1, mode='same')
imgd2 = convolve(img, kd2, mode='same')

plt.figure(figsize=(20,10))
plt.gray()
plt.subplot(231), plt.imshow(img), plt.title('image', size=15)
plt.subplot(232), plt.imshow(imgd1), plt.title('1st derivative', size=15)
plt.subplot(233), plt.imshow(imgd2), plt.title('2nd derivative', size=15)
plt.subplot(234), plt.plot(range(w), img[0,:]), plt.title('image function', size=15)
plt.subplot(235), plt.plot(range(w), imgd1[0,:]), plt.title('1st derivative function', size=15)
plt.subplot(236), plt.plot(range(w), imgd2[0,:]), plt.title('2nd derivative function', size=15)
plt.show()

: With LoG / Zero-Crossing

import numpy as np
from scipy import ndimage
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

def any_neighbor_neg(img, i, j):
    for k in range(-1,2):
      for l in range(-1,2):
         if img[i+k, j+k] < 0:
            return True, img[i, j] - img[i+k, j+k]
    return False, None

def zero_crossing(img, th):
  out_img = np.zeros(img.shape)
  for i in range(1,img.shape[0]-1):
    for j in range(1,img.shape[1]-1):
      found, slope = any_neighbor_neg(img, i, j)
      if img[i,j] > 0 and found and slope > th:
        out_img[i,j] = 255
  return out_img

img = rgb2gray(imread('images/Img_04_18.jpg'))

print(np.max(img))
fig = plt.figure(figsize=(10,16))
plt.subplots_adjust(0,0,1,0.95,0.05,0.05)
plt.gray() # show the filtered result in grayscale
for sigma, thres in zip(range(3,10,2), [1e-3, 1e-4, 1e-5, 1e-6]):
    plt.subplot(3,2,sigma//2)
    result = ndimage.gaussian_laplace(img, sigma=sigma)
    result = zero_crossing(result, thres)
    plt.imshow(result)
    plt.axis('off')
    plt.title('LoG with zero-crossing, sigma=' + str(sigma), size=20)

plt.tight_layout()
plt.show()

# With Canny and Holistically-nested (deep learning model based)

import cv2
import numpy as np
import matplotlib.pylab as plt

image = cv2.imread('images/Img_04_18.jpg')
(h, w) = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blurred, 80, 150)

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0
        
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]
        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width
        return [[batchSize, numChannels, height, width]]
    
    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

prototxt_path = "models/deploy.prototxt"
model_path = "models/hed_pretrained_bsds.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cv2.dnn_registerLayer('Crop', CropLayer)

blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(w, h),                              mean=(104.00698793, 116.66876762, 122.67891434),                              swapRB=False, crop=False)

net.setInput(blob)
hed = net.forward()
hed = cv2.resize(outs[i][0][0,:,:], (w, h))
hed = (255 * hed).astype("uint8")

plt.figure(figsize=(20, 8))
plt.gray()
plt.subplots_adjust(0,0,1,0.975,0.05,0.05)
plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('input', size=20)
plt.subplot(132), plt.imshow(canny), plt.axis('off'), plt.title('canny', size=20)
plt.subplot(133), plt.imshow(hed), plt.axis('off'), plt.title('holistically-nested', size=20)
plt.show()

# Creating different Hatched Contour Patterns for different levels

from skimage.io import imread
from skimage.color import rgb2gray

img = rgb2gray(imread('images/Img_04_02.jpg'))

y = np.arange(img.shape[0]) 
x = np.arange(img.shape[1]) 
x, y = np.meshgrid(x, y)
z = img

plt.figure(figsize=(10,10))
cs = plt.contourf(x, img.shape[0]-y, z, hatches=['-', '/', '\\', '//', '//\\', '//\\\\'], cmap='gray', extend='both', levels=6, alpha=0.5)
cs.cmap.set_over('red')
cs.cmap.set_under('blue')
cs.changed()
plt.colorbar()
plt.axis('off')
plt.show()

8: Object detection with Hough Transform and Colors

: Counting cirular objects in an image with Circle Hough Transform

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from sklearn.neighbors import KDTree

orig = imread('pathology_hemolyticanemia.jpg')
h, w = orig.shape[:2]
image = rgb2gray(orig)
edges = canny(image, sigma=1, low_threshold=0.15, high_threshold=0.45)

hough_radii = np.arange(10, 20, 1)
hough_res = hough_circle(edges, hough_radii)

accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           min_xdistance = 10,
                                           min_ydistance = 10,
                                           #num_peaks = 5,
                                           total_num_peaks=400)

cells = []
image = orig.copy()
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
    if len(cells) > 1:
        tree = KDTree(np.array(cells), leaf_size=2) 
        count = tree.query_radius(np.array([[center_y, center_x]]), r=10, count_only=True)
        if count[0] > 0: continue
    cells.append([center_y, center_x])
    for j in range(-3,4):
        image[np.minimum(circy+j,h-1), np.minimum(circx+j,w-1)] = (255, 0, 0)

print(len(cx))

plt.figure(figsize=(20, 8))
plt.gray()
plt.subplots_adjust(0,0,1,0.975,0.05,0.05)
plt.subplot(131), plt.imshow(orig), plt.axis('off'), plt.title('original', size=20)
plt.subplot(132), plt.imshow(edges), plt.axis('off'), plt.title('edges with canny', size=20)
plt.subplot(133), plt.imshow(image), plt.axis('off'), plt.title('cells detected', size=20)
plt.suptitle('Counting blood-cells with Circle Hough transform, number of cells={}'.format(len(cells)), size=30)
plt.show()

# Detecting lines with Progressive Probabilistic Hough Transform

from skimage.color import rgb2gray
from skimage.transform import probabilistic_hough_line

image = rgb2gray(imread('images/Img_04_20.jpg')) # the image have pixel values in the range [0,1]
edges = canny(image, 2, 30/255, 80/255)
lines = probabilistic_hough_line(edges, threshold=20, line_length=20, line_gap=5)

fig, axes = plt.subplots(1, 3, figsize=(30, 20), sharex=True, sharey=True)
ax = axes.ravel()
plt.gray()
ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Input image', size=25)
ax[1].imshow(edges, cmap=plt.cm.gray)
ax[1].set_title('Canny edges', size=25)
ax[2].imshow(edges * 0)
for line in lines:
 p0, p1 = line
 ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]), linewidth=5)
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough', size=25)
for a in ax:
 a.set_axis_off()
plt.axis('off')
plt.tight_layout()
plt.show()

# Detecting Objects of arbitrary shapes using Generalized Hough Transform

from matplotlib.pylab import imshow, title, show
from skimage.filters import threshold_otsu
import cv2
print(cv2.__version__)

import numpy as np
import matplotlib,pylab as plt

orig = cv2.imread('images/Img_04_25.png')
img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
templ = cv2.imread('images/Img_04_26.png', 0)
edges = cv2.Canny(img, 130,150)

alg = cv2.createGeneralizedHoughBallard()
alg.setTemplate(templ)
[positions,votes] = alg.detect(edges)

clone = orig.copy() #np.dstack([edges, edges, edges])
for i in range(len(positions[0])):
    pos, scale, angle = positions[0][i][:2], positions[0][i][2], positions[0][i][3]
    print(pos, scale, angle)
    # need to write code here to rotate the bounding rect if angle is not zero and scale is not 1
    cv2.rectangle(clone, (int(pos[0]) - templ.shape[1]//2, int(pos[1]) - templ.shape[0]//2), 
                         (int(pos[0] + templ.shape[1]//2), int(pos[1] + templ.shape[0]//2)), 
                         (0,0,255), 2)

plt.figure(figsize=(20, 8))
plt.gray()
plt.subplots_adjust(0,0,1,0.975,0.05,0.05)
plt.subplot(131), plt.imshow(img), plt.axis('off'), plt.title('input', size=20)
plt.subplot(132), plt.imshow(templ), plt.axis('off'), plt.title('template', size=20)
plt.subplot(133), plt.imshow(clone), plt.axis('off'), plt.title('object detection with generalized Hough', size=20)
plt.show()

: Detecting Objects with Colors in HSV colorspace

img = cv2.cvtColor(cv2.imread('covid_19_blood.jpg'), cv2.COLOR_BGR2RGB)
img_hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

low_green = np.array([30, 25, 10])
high_green = np.array([80, 255, 255])
green_mask = cv2.inRange(img_hsv, low_green, high_green)
green = cv2.bitwise_and(img, img, mask=green_mask)

output_img = img.copy()
output_img[np.where(green_mask==0)] = (0,0,0)

plt.figure(figsize=(20, 8))
plt.gray()
plt.subplots_adjust(0,0,1,0.975,0.05,0.05)
plt.subplot(131), plt.imshow(img), plt.axis('off'), plt.title('original', size=20)
plt.subplot(132), plt.imshow(green_mask), plt.axis('off'), plt.title('mask', size=20)
plt.subplot(133), plt.imshow(output_img), plt.axis('off'), plt.title('covi-19 virus cells', size=20)
plt.suptitle('Filtering out the covid-19 virus cells', size=30)
plt.show()

# Object Saliency Map, Depth Map and Tone Map (HDR) with opencv-python

# Creating Object Saliency Map 

import cv2
import numpy as np
from matplotlib import pylab as plt

image = cv2.imread('images/Img_04_24.jpg')
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliency_map) = saliency.computeSaliency(image)

thresh_map = cv2.threshold(saliency_map.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

plt.figure(figsize=(20,20))
plt.gray()
plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('input image', size=20)
plt.subplot(132), plt.imshow(saliency_map), plt.axis('off'), plt.title('saliancy', size=20)
plt.subplot(133), plt.imshow(thresh_map), plt.axis('off'), plt.title('threshold', size=20)
plt.tight_layout()
plt.show()

# Creating Depth-Map from Stereo images

import cv2
import matplotlib.pylab as plt

img_left = cv2.imread('images/stereo/im0.ppm', 0)
img_right = cv2.imread('images/stereo/im1.ppm', 0)

matcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = matcher.compute(img_left, img_right)

plt.figure(figsize=(20,20))
plt.gray()
plt.subplot(131), plt.imshow(img_left), plt.axis('off'), plt.title('left unput image', size=20)
plt.subplot(132), plt.imshow(img_right), plt.axis('off'), plt.title('right input image', size=20)
plt.subplot(133), plt.imshow(disparity), plt.axis('off'), plt.title('disparity map', size=20)
plt.tight_layout()
plt.show()

# Tone mapping and High Dynamic Range (HDR) Imaging

import cv2
print(cv2.__version__)

import numpy as np

import matplotlib.pylab as plt

hdr_image = cv2.imread("images/hdr/GCanyon_C_YumaPoint_3k.hdr", -1)

tonemap_drago = cv2.createTonemapDrago(1.0, 0.7)
ldr_drago = tonemap_drago.process(hdr_image)
ldr_drago = 3 * ldr_drago
ldr_drago = cv2.cvtColor(ldr_drago, cv2.COLOR_BGR2RGB)

tonemap_durand = cv2.createTonemapDurand(1.5,4,1.0,1,1)
ldr_durand = tonemap_durand.process(hdr_image)
ldr_durand = 3 * ldr_durand
ldr_durand = cv2.cvtColor(ldr_durand, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))
plt.subplot(211), plt.imshow(ldr_drago), plt.axis('off'), plt.title('Tone mapping with Drago\'s method', size=20)
plt.subplot(212), plt.imshow(ldr_durand), plt.axis('off'), plt.title('Tone mapping with Durand\'s method', size=20)
plt.tight_layout()
plt.show()

10: Pyramid Blending

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import pyramid_reduce, pyramid_laplacian, pyramid_expand, resize

def get_gaussian_pyramid(image):
    rows, cols, dim = image.shape
    gaussian_pyramid = [image]
    while rows > 1 and cols > 1:
        #print(rows, cols)
        image = pyramid_reduce(image, downscale=2)
        gaussian_pyramid.append(image)
        #print(image.shape)
        rows //= 2
        cols //= 2
    return gaussian_pyramid

def get_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = [gaussian_pyramid[len(gaussian_pyramid)-1]]
    for i in range(len(gaussian_pyramid)-2, -1, -1):
        image = gaussian_pyramid[i] - resize(pyramid_expand(gaussian_pyramid[i+1]), gaussian_pyramid[i].shape)
        #print(i, image.shape)
        laplacian_pyramid.append(np.copy(image))
    laplacian_pyramid = laplacian_pyramid[::-1]
    return laplacian_pyramid

def reconstruct_image_from_laplacian_pyramid(pyramid):
    i = len(pyramid) - 2
    prev = pyramid[i+1]
    plt.figure(figsize=(20,18))
    j = 1
    while i >= 0:
        prev = resize(pyramid_expand(prev, upscale=2), pyramid[i].shape)
        im = np.clip(pyramid[i] + prev,0,1)
        plt.subplot(3,3,j)
        plt.imshow(im)
        plt.title('Level=' + str(j) + ', ' + str(im.shape[0]) + 'x' + str(im.shape[1]), size=20)
        prev = im
        i -= 1
        j += 1
    plt.suptitle('Image constructed from the Laplacian Pyramid', size=30)
    plt.show()
    return im

A = imread('images/Img_04_10.png')[...,:3] / 255
B = imread('images/Img_04_11.png')[...,:3] / 255
M = imread('images/Img_04_12.png')[...,:3] / 255

rows, cols, dim = A.shape
pyramidA = get_laplacian_pyramid(get_gaussian_pyramid(A))
pyramidB = get_laplacian_pyramid(get_gaussian_pyramid(B))
pyramidM = get_gaussian_pyramid(M)

pyramidC = []
for i in range(len(pyramidM)):
    im = pyramidM[i]*pyramidA[i] + (1-pyramidM[i])*pyramidB[i]
    #print(np.max(im), np.min(im), np.mean(im))
    pyramidC.append(im)

im = reconstruct_image_from_laplacian_pyramid(pyramidC)

plt.figure(figsize=(20,10))
plt.imshow(im), plt.axis('off'), plt.title('Blended output image with Pyramid', size=20)
plt.show()

# Face Recognition with FisherFaces

import cv2
import time
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

cascade_path = "models/haarcascade_frontalface_default.xml"

def get_images_and_labels(path):
    faceCascade = cv2.CascadeClassifier(cascade_path)
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('subject') and not f.endswith('glasses.gif')]
    # images will contains face images
    images = np.zeros((15*11, 160*160))
    # labels will contains the label that is assigned to the image
    labels = []
    i = 0
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        (x, y, w, h) = faces[0]
        images[i,:] = np.ravel(cv2.resize(image[y: y + h, x: x + w], (160,160)))
        labels.append(nbr)
        i += 1 
    # return the images list and labels list
    return images, labels

images, labels = get_images_and_labels('images/yalefaces')

indices = np.random.choice(165, 64)
plt.figure(figsize=(20,20))
plt.gray()
plt.subplots_adjust(0,0,1,0.925,0.05,0.15)
for i in range(len(indices)):
    plt.subplot(8,8,i+1), plt.imshow(np.reshape(images[indices[i],:], (160,160))), plt.axis('off')
    plt.title(labels[indices[i]], size=20)
plt.suptitle('Faces from Yale Face database for 15 different subjects', size=25)
plt.show()

def fisherfaces(X_train, X_test, y_train, y_test):

    stamp = time.time()
    
    print('Fitting data into LDA..')
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    print(lda.explained_variance_ratio_)
    
    meanfaces, classes = lda.means_, lda.classes_
    plt.figure(figsize=(20,12))
    for i in range(15):
        plt.subplot(3,5,i+1), plt.imshow(np.reshape(meanfaces[i], (160,160))), plt.axis('off')
        plt.title(classes[i], size=20)
    plt.suptitle('FisherFaces for the subjects', size=25)
    plt.show()
    
    X_train = lda.transform(X_train)
    X_test = lda.transform(X_test)

    print('Data fitting into LDA finished.')
    print('Training process finished in ' + str(round((time.time() - stamp), 5))+' seconds.')
    
    cls = KNeighborsClassifier(n_neighbors=5)
    cls.fit(X_train, y_train)
    prediction = cls.predict(X_test)
    print('Data prediction finished.')
    print('Classification report\n'+classification_report(
        y_test, prediction
    ))
    print(accuracy_score(y_test, prediction)) 

scaler = StandardScaler()
images = scaler.fit_transform(images)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, stratify=labels, random_state=42)

fisherfaces(X_train, X_test, y_train, y_test)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train([np.reshape(im, (160,160)) for im in X_train.tolist()], np.array(y_train))
correct = 0
for i in range(len(y_test)):
    nbr_predicted, conf = recognizer.predict(np.reshape(X_test[i,:], (160,160)))
    nbr_actual = y_test[i]
    if nbr_actual == nbr_predicted:
        print("{} is Correctly Recognized with confidence {}".format(nbr_actual, conf))
        correct += 1
    else:
        print("{} is Incorrectly Recognized as {}".format(nbr_actual, nbr_predicted))
print('test accuracy = {}'.format(correct/len(y_test)))

# Face Detection and Recognition with Microsoft Cognitive Vision APIs

import os
import sys
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from skimage.color import gray2rgb

subscription_key ='<your key here>' # must use your key here, otherwise you will not be able to use the service
endpoint = 'https://xxxxxxxxxxxxx.api.cognitive.microsoft.com' # replace this by your endpoint

analyze_url = endpoint + "/vision/v3.0/analyze"

def detect_face_age_geneder(img):
    
    # Read the image into a byte array
    image_data = open(img, "rb").read()
    
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
    params = {'visualFeatures': 'Faces', "details": "", "language": "en"}
    response = requests.post(analyze_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    out = response.json()
    print(out)
    
    image = Image.open(BytesIO(image_data))
    image = gray2rgb(np.array(image))
    for face in out['faces']:
        age = face["age"]
        gender = face["gender"]
        box = face["faceRectangle"]
        x1, y1, x2, y2 = int(box['left']), int(box['top']), int(box['left'] + box['height']), int(box['top'] + box['width'])
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2) 
        image = cv2.putText(image, 'age={}'.format(age), (x1, y1-5),                                 cv2.FONT_HERSHEY_SIMPLEX ,  1, (255,0,0), 2, cv2.LINE_AA) 
        image = cv2.putText(image, gender, (x1, y2+25),                                 cv2.FONT_HERSHEY_SIMPLEX ,  1, (255,0,0), 2, cv2.LINE_AA) 

    # Display the image and overlay it with the caption.
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis("off")
    plt.title('Face Recognition with Microsoft Cognitive Vision API', size=20)
    plt.show()

detect_face_age_geneder('images/Img_04_02.jpg')

analyze_url = endpoint + "/vision/v3.0/models/celebrities/analyze"

def recognize_celeb_face(img):
    
    # Read the image into a byte array
    image_data = open(img, "rb").read()
    image = Image.open(BytesIO(image_data))
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
    params = {'visualFeatures': 'Categories,Description,Color', "details": "", "language": "en"}
    response = requests.post(analyze_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    out = response.json()
    print(out)
    
    image = gray2rgb(np.array(image))
    for celeb in out['result']["celebrities"]:
        name = celeb["name"].capitalize()
        confidence = celeb["confidence"]
        box = celeb["faceRectangle"]
        x1, y1, x2, y2 = int(box['left']), int(box['top']), int(box['left'] + box['height']), int(box['top'] + box['width'])
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2) 
        image = cv2.putText(image, name, (x1-5, y1-5),                                 cv2.FONT_HERSHEY_SIMPLEX ,  1, (255,0,0), 2, cv2.LINE_AA) 
        image = cv2.putText(image, 'conf={:.03f}'.format(confidence), (x1, y2+25),                                 cv2.FONT_HERSHEY_SIMPLEX ,  1, (255,0,0), 2, cv2.LINE_AA) 

    # Display the image and overlay it with the caption.
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis("off")
    plt.title('Face Recognition with Microsoft Cognitive Vision API', size=20)
    plt.show()

recognize_celeb_face("images/Img_04_21.jpg")

# Image Super Resolution with deep learning model (SRGAN)

import tensorflow as tf
tf.compat.v1.enable_eager_execution() 

from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg19 import VGG19
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0

def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1

def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)

def res_block(x_in, num_filters, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x

def sr_resnet(num_filters=64, num_res_blocks=16):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize_01)(x_in)

    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample(x, num_filters * 4)
    x = upsample(x, num_filters * 4)

    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    x = Lambda(denormalize_m11)(x)

    return Model(x_in, x)

generator = sr_resnet

LR_SIZE = 24
HR_SIZE = 96

def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    return LeakyReLU(alpha=0.2)(x)

def discriminator(num_filters=64):
    x_in = Input(shape=(HR_SIZE, HR_SIZE, 3))
    x = Lambda(normalize_m11)(x_in)

    x = discriminator_block(x, num_filters, batchnorm=False)
    x = discriminator_block(x, num_filters, strides=2)

    x = discriminator_block(x, num_filters * 2)
    x = discriminator_block(x, num_filters * 2, strides=2)

    x = discriminator_block(x, num_filters * 4)
    x = discriminator_block(x, num_filters * 4, strides=2)

    x = discriminator_block(x, num_filters * 8)
    x = discriminator_block(x, num_filters * 8, strides=2)

    x = Flatten()(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)

def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]

def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

weights_dir = 'models/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)

gan_generator = generator()
gan_generator.load_weights(weights_file('pre_generator.h5'))

lr = Image.open('images/Img_04_19.jpg')
sr = lr.resize((lr.width*4, lr.height*4), Image.BICUBIC)
lr, sr = np.array(lr), np.array(sr)

gan_sr = resolve_single(gan_generator, lr)
gan_sr = gan_sr.numpy()
gan_sr = gan_sr / gan_sr.max()

plt.figure(figsize=(2.5, 1.5))
plt.imshow(lr), plt.title('original\n(low resolution)', size=15)
plt.show()
plt.figure(figsize=(15, 9))
plt.subplots_adjust(0,0,1,1,0.05,0.05)
images = [sr, gan_sr]
titles = ['4x high resolution\n(with bicubic interpolation)', ' X4 super resolution\n(with SRGAN)']
positions = [1, 2]
for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
    plt.subplot(1, 2, pos)
    plt.imshow(img)
    plt.title(title, size=20)
plt.show()

# Low-light Image Enhancement Using CNNs

import tensorflow as tf
print(tf.__version__)

import numpy as np
from skimage.io import imread
import matplotlib.pylab as plt
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model

def build_mbllen(input_shape):

    def EM(input, kernal_size, channel):
        conv_1 = Conv2D(channel, (3, 3), activation='relu', padding='same', data_format='channels_last')(input)
        conv_2 = Conv2D(channel, (kernal_size, kernal_size), activation='relu', padding='valid', data_format='channels_last')(conv_1)
        conv_3 = Conv2D(channel*2, (kernal_size, kernal_size), activation='relu', padding='valid', data_format='channels_last')(conv_2)
        conv_4 = Conv2D(channel*4, (kernal_size, kernal_size), activation='relu', padding='valid', data_format='channels_last')(conv_3)
        conv_5 = Conv2DTranspose(channel*2, (kernal_size, kernal_size), activation='relu', padding='valid', data_format='channels_last')(conv_4)
        conv_6 = Conv2DTranspose(channel, (kernal_size, kernal_size), activation='relu', padding='valid', data_format='channels_last')(conv_5)
        res = Conv2DTranspose(3, (kernal_size, kernal_size), activation='relu', padding='valid', data_format='channels_last')(conv_6)
        return res

    inputs = Input(shape=input_shape)
    FEM = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(inputs)
    EM_com = EM(FEM, 5, 8)

    for j in range(3):
        for i in range(0, 3):
            FEM = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(FEM)
            EM1 = EM(FEM, 5, 8)
            EM_com = Concatenate(axis=3)([EM_com, EM1])

    outputs = Conv2D(3, (1, 1), activation='relu', padding='same', data_format='channels_last')(EM_com)
    return Model(inputs, outputs)

mbllen = build_mbllen((None, None, 3))
mbllen.load_weights('models/LOL_img_lowlight.h5')

img = imread('images/Img_04_27.jpg')
print(img.max())
out_pred = mbllen.predict(img[np.newaxis, :] / 255)
out = out_pred[0, :, :, :3]

plt.figure(figsize=(20,10))
plt.subplot(121), plot_image(img, 'low-light input')
plt.subplot(122), plot_image(np.clip(out, 0, 1), 'enhanced output')
plt.tight_layout()
plt.show()

# Realistic Image Dehazing using deep neural net

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Resize

import matplotlib.pylab as plt

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      reflection_padding = kernel_size // 2
      self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) * rgb_range

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

class Net(nn.Module):
    def __init__(self, res_blocks=18):
        super(Net, self).__init__()

        rgb_mean = (0.5204, 0.5167, 0.5129)
        self.sub_mean = MeanShift(1., rgb_mean, -1)
        self.add_mean = MeanShift(1., rgb_mean, 1)

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)

        self.dehaze = nn.Sequential()
        for i in range(1, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))

        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.relu(self.conv_input(x))
        res2x = self.relu(self.conv2x(x))
        res4x = self.relu(self.conv4x(res2x))

        res8x = self.relu(self.conv8x(res4x))
        res16x = self.relu(self.conv16x(res8x))

        res_dehaze = res16x
        res16x = self.dehaze(res16x)
        res16x = torch.add(res_dehaze, res16x)

        res16x = self.relu(self.convd16x(res16x))
        res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)

        res8x = self.relu(self.convd8x(res8x))
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)

        res4x = self.relu(self.convd4x(res4x))
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)

        res2x = self.relu(self.convd2x(res2x))
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)

        x = self.conv_output(x)

        return x

rb = 13
checkpoint = "models/I-HAZE_O-HAZE.pth"

net = Net(rb)
net.load_state_dict(torch.load(checkpoint)['state_dict'])
net.eval()

im_path = "images/Img_04_28.jpg"
im = Image.open(im_path)
h, w = im.size
print(h, w)

imt = ToTensor()(im)
imt = Variable(imt).view(1, -1, w, h)

with torch.no_grad():
    imt = net(imt)
out = torch.clamp(imt, 0., 1.)
out = out.cpu()
out = out.data[0]
out = ToPILImage()(out)

plt.figure(figsize=(20,10))
plt.subplot(121), plot_image(im, 'hazed input')
plt.subplot(122), plot_image(out, 'de-hazed output')
plt.tight_layout()
plt.show()

# Distributed Image Processing with Dask

from skimage.io import imread, imsave
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pylab as plt
import dask_image.imread
import dask.array
import dask_image.ndfilters
import os, glob
from IPython.display import display

def partion_image(imgfile, n_h=2, n_w=2, plot=True):
    im = imread(imgfile)
    h, w, _ = im.shape
    h_s, w_s = h // n_h, w // n_w
    k = 0
    for i in range(n_h):
        for j in range(n_w):
            imsave(imgfile[:-4] + '_part_{:03d}'.format(k) + imgfile[-4:], im[i*h_s:(i+1)*h_s, j*w_s:(j+1)*w_s, :])          
            k += 1
    if plot:
        k = 0
        plt.figure(figsize=(20,16))
        plt.subplots_adjust(0,0,1,1,0.05,0.05)
        for i in range(n_h):
            for j in range(n_w):
                im = plt.imread(imgfile[:-4] + '_part_{:03d}'.format(k) + imgfile[-4:])
                plt.subplot(n_h, n_w, k+1), plt.imshow(im), plt.title('image part-{}'.format(k+1), size=20)
                k += 1
        plt.show()

def plot_image(image):
    plt.figure(figsize=(20,20))
    plt.imshow(image, cmap='gray')
    plt.show()

imgfile = 'images/Img_04_22.png'
partion_image(imgfile)

filename_pattern = os.path.join('./', imgfile[:-4] + '_part_*' + imgfile[-4:])
partitioned_images = dask_image.imread.imread(filename_pattern)
print(partitioned_images)

result  = (rgb2gray(partitioned_images))
print(result.shape)
plot_image(result[0])

data = [result[i, ...] for i in range(result.shape[0])]
data = [data[i:i+2] for i in range(0, len(data), 2)]
combined_image = dask.array.block(data)
print(combined_image.shape)    
plot_image(combined_image)

edges = dask_image.ndfilters.sobel(combined_image)
print(edges)
display(edges.visualize())

edges = np.clip(edges, 0, 1)
plot_image(edges)

