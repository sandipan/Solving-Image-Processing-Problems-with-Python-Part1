# Chapter 3: Sampling, Convolution, Discrete Fourier, Cosine and Wavelet Transform

# Author: Sandipan Dey

# Fourier Transform Basics

import numpy as np
import numpy.fft as fp
from scipy import signal
import scipy.fftpack
from skimage.io import imread
from skimage.color import rgb2gray 
from skimage.metrics import peak_signal_noise_ratio
from scipy.ndimage import convolve
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot_image(im, title):
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title(title, size=20)

def plot_freq_spectrum(F, title, cmap=plt.cm.gray, show_axis=True, colorbar=False):
    plt.imshow((20*np.log10(0.1 + fp.fftshift(F))).real.astype(int), cmap=cmap)
    if not show_axis:
        plt.axis('off')
    if colorbar:
        plt.colorbar()
    plt.title(title, size=20)

h, w = 100, 100
images = list()
im = np.zeros((h,w))
for x in range(h):
    im[x,:] = np.sin(x)
images.append(im)
im = np.zeros((h,w))
for y in range(w):
    im[:,y] = np.sin(y)
images.append(im)

im = np.zeros((h,w))
for x in range(h):
    for y in range(w):
        im[x,y] = np.sin(x + y) 
images.append(im)
im = np.zeros((h,w))
for x in range(h):
    for y in range(w):
        if (x-h/2)**2 + (y-w/2)**2 < 100:
            im[x,y] = np.sin(x + y) 
images.append(im)

im = np.zeros((h,w))
for x in range(h):
    for y in range(w):
        if (x-h/2)**2 + (y-w/2)**2 < 25:
            im[x,y] = 1 
images.append(im)
im = np.zeros((h,w))
im[h//2 -5:h//2 + 5, w//2 -5:w//2 + 5] = 1 
images.append(im)

plt.figure(figsize=(25,10))
i = 1
for im in images:
    plt.subplot(2,6,i), plot_image(im, 'image {}'.format(i))
    plt.subplot(2,6,i+6), plot_freq_spectrum(fp.fft2(im), 'DFT {}'.format(i), show_axis=False)
    i += 1
plt.tight_layout()
plt.show() 

im = rgb2gray(imread("images/Img_03_01.jpg"))
h, w = im.shape
F = fp.fft2(im)
F_shifted = fp.fftshift(F)

xs = list(map(int, np.linspace(1, h//5, 10)))
ys = list(map(int, np.linspace(1, w//5, 10)))
plt.figure(figsize=(20,8))
plt.gray()
for i in range(10):
    F_mask = np.zeros((h, w))
    F_mask[h//2-xs[i]:h//2+xs[i]+1, w//2-ys[i]:w//2+ys[i]+1] = 1 
    F1 = F_shifted*F_mask
    im_out =  fp.ifft2(fp.ifftshift(F1)).real #np.abs()
    plt.subplot(2,5,i+1), plt.imshow(im_out), plt.axis('off')
    plt.title('{}x{},PSNR={}'.format(2*xs[i]+1, 2*ys[i]+1, round(peak_signal_noise_ratio(im, im_out),2)), size=15)
plt.suptitle('Fourier reconstruction by keeping first few frequency basis vectors', size=25)
plt.show()

# Sampling to increase/decrease the resolution of an image

# Up-sampling an image by using DFT and a Low-pass-filter (LPF)

im = 255*rgb2gray(imread('images/Img_03_01.jpg'))
im1 = np.zeros((2*im.shape[0], 2*im.shape[1]))
print(im.shape, im1.shape)
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        im1[2*i,2*j] = im[i,j]

kernel = [[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]

def pad_with_zeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

kernel = np.pad(kernel, (((im1.shape[0]-3)//2,(im1.shape[0]-3)//2+1), ((im1.shape[1]-3)//2,(im1.shape[1]-3)//2+1)), pad_with_zeros) 

freq = fp.fft2(im1)
freq_kernel = fp.fft2(fp.ifftshift(kernel))

freq_LPF = freq*freq_kernel # by the Convolution theorem

im2 = fp.ifft2(freq_LPF).real

plt.figure(figsize=(15,10))
plt.gray() # show the filtered result in grayscale
cmap = 'nipy_spectral' #'viridis'
plt.subplot(231), plot_image(im, 'Original Input Image')
plt.subplot(232), plot_image(im1, 'Padded Input Image')
plt.subplot(233), plot_freq_spectrum(freq, 'Original Image Spectrum', cmap=cmap)
plt.subplot(234), plot_freq_spectrum(freq_kernel, 'Image Spectrum of the LPF', cmap=cmap)
plt.subplot(235), plot_freq_spectrum(fp.fft2(im2), 'Image Spectrum after LPF', cmap=cmap)
plt.subplot(236), plot_image(im2.astype(np.uint8), 'Output Image')
plt.show()

from skimage.filters import gaussian
from skimage import img_as_float

im = img_as_float(imread('images/Img_03_08.jpg'))
print(im.shape)

im_blurred = gaussian(im, sigma=1.25, multichannel=True) 

n = 4 # create and image 16 times smaller in size
h, w = im.shape[0] // n, im.shape[1] // n
im_small = np.zeros((h, w, 3))
for i in range(h):
   for j in range(w):
      im_small[i,j] = im[n*i, n*j]
im_small_aa = np.zeros((h, w, 3))
for i in range(h):
   for j in range(w):
      im_small_aa[i,j] = im_blurred[n*i, n*j]

plt.figure(figsize=(15,15))
plt.imshow(im), plt.title('Original Image', size=15)
plt.show()

plt.figure(figsize=(15,15))
plt.imshow(im_small), plt.title('Resized Image (without Anti-aliasing)', size=15)
plt.show()

plt.figure(figsize=(15,15))
plt.imshow(im_small_aa), plt.title('Resized Image (with Anti-aliasing)', size=15)
plt.show()

# Denoising an Image with LPF/Notch filter in the Frequency domain

# Removing Periodic Noise with Notch Filter

im_noisy = rgb2gray(imread("images/Img_03_23.jpg"))
F_noisy = fp.fft2((im_noisy))
print(F_noisy.shape)

def plot_freq_spectrum(F, title, cmap=plt.cm.gray):
    plt.imshow((20*np.log10(0.1 + fp.fftshift(F))).real.astype(int), cmap=cmap)
    plt.xticks(np.arange(0, im.shape[1], 25))
    plt.yticks(np.arange(0, im.shape[0], 25))
    plt.title(title, size=20)

plt.figure(figsize=(20,10))

plt.subplot(121), plot_image(im_noisy, 'Noisy Input Image')
plt.subplot(122), plot_freq_spectrum(F_noisy, 'Noisy Image Spectrum') 

plt.tight_layout()
plt.show()

F_noisy_shifted = fp.fftshift(F_noisy)
F_noisy_shifted[180,210] = F_noisy_shifted[200,190] = 0
im_out =  fp.ifft2(fp.ifftshift(F_noisy_shifted)).real #np.abs()

plt.figure(figsize=(10,8))
plot_image(im_out, 'Output Image')
plt.show()

# Removing salt-and-pepper noise using Gaussian LPF with scipy fftpack

from scipy import ndimage
from scipy import fftpack
from skimage.util import random_noise

im = rgb2gray(imread('images/Img_03_02.jpg'))
noisy = random_noise(im, mode='s&p')

im_freq = fftpack.fft2(im)
noisy_freq = fftpack.fft2(noisy)
sigma = 1 #0.1
noisy_smoothed_freq = ndimage.fourier_gaussian(noisy_freq, sigma=sigma)
noisy_smoothed = fftpack.ifft2(noisy_smoothed_freq)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20,20))
plt.gray()  # show the filtered result in grayscale
ax1.imshow(im), ax1.axis('off'), ax1.set_title('Original Image', size=20)
ax2.imshow((20*np.log10(0.1 + fftpack.fftshift(im_freq))).real.astype(int))
ax2.set_title('Original Image (Freq Spec)', size=20)
ax3.imshow(noisy), ax3.axis('off'), ax3.set_title('Noisy Image', size=20)
ax4.imshow((20*np.log10( 0.1 + fftpack.fftshift(noisy_freq))).real.astype(int))
ax4.set_title('Noisy Image (Freq Spec)', size=20)

ax5.imshow(noisy_smoothed.real), ax5.axis('off'), ax5.set_title('Output Image (with LPF)', size=20)
ax6.imshow( (20*np.log10( 0.1 + fftpack.fftshift(noisy_smoothed_freq))).real.astype(int))
ax6.set_title('Output Image (Freq Spec)', size=20)
plt.tight_layout()
plt.show()

# Blurring an Image with an LPF in the Frequency domain

# Different Blur Kernels and Convolution in the Frequency domain 

def get_gaussian_edge_blur_kernel(sigma, sz=15):
    # First create a 1-D Gaussian kernel
    x = np.linspace(-10, 10, sz)
    kernel_1d = np.exp(-x**2/sigma**2)
    kernel_1d /= np.trapz(kernel_1d) # normalize the sum to 1.0
    # create a 2-D Gaussian kernel from the 1-D kernel
    kernel_2d = kernel_1d[:, np.newaxis] * kernel_1d[np.newaxis, :]
    return kernel_2d

def get_motion_blur_kernel(ln, angle, sz=15):
    kern = np.ones((1, ln), np.float32)
    angle = -np.pi*angle/180
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((ln-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

def get_out_of_focus_kernel(r, sz=15):
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), r, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255
    return kern

def dft_convolve(im, kernel):
    F_im = fp.fft2(im)
    #F_kernel = fp.fft2(kernel, s=im.shape)
    F_kernel = fp.fft2(fp.ifftshift(kernel), s=im.shape)
    F_filtered = F_im * F_kernel
    im_filtered = fp.ifft2(F_filtered)
    cmap = 'RdBu'
    plt.figure(figsize=(20,10))
    plt.gray()
    plt.subplot(131), plt.imshow(im), plt.axis('off'), plt.title('input image', size=20)
    plt.subplot(132), plt.imshow(kernel, cmap=cmap), plt.title('kernel', size=20)
    plt.subplot(133), plt.imshow(im_filtered.real), plt.axis('off'), plt.title('output image', size=20)
    plt.tight_layout()
    plt.show()

im = rgb2gray(imread('images/Img_03_03.jpg'))

kernel = get_gaussian_edge_blur_kernel(25, 25)
dft_convolve(im, kernel)

kernel = get_motion_blur_kernel(30, 60, 25)
dft_convolve(im, kernel)

kernel = get_out_of_focus_kernel(15, 20)
dft_convolve(im, kernel)

# Blurring with scipy.ndimage frequency-domain filters

im = imread('images/Img_03_31.png')
freq = fp.fft2(im)

fig, axes = plt.subplots(2, 3, figsize=(20,15))
plt.subplots_adjust(0,0,1,0.95,0.05,0.05)
plt.gray() # show the filtered result in grayscale
axes[0, 0].imshow(im), axes[0, 0].set_title('Original Image', size=20)
axes[1, 0].imshow((20*np.log10( 0.1 + fp.fftshift(freq))).real.astype(int)), axes[1, 0].set_title('Original Image Spectrum', size=20)
i = 1
for sigma in [3,5]:
    convolved_freq = ndimage.fourier_gaussian(freq, sigma=sigma)
    convolved = fp.ifft2(convolved_freq).real # the imaginary part is an artifact
    axes[0, i].imshow(convolved) 
    axes[0, i].set_title(r'Output with FFT Gaussian Blur, $\sigma$={}'.format(sigma), size=20)
    axes[1, i].imshow((20*np.log10( 0.1 + fp.fftshift(convolved_freq))).real.astype(int))
    axes[1, i].set_title(r'Spectrum with FFT Gaussian Blur, $\sigma$={}'.format(sigma), size=20)
    i += 1
for a in axes.ravel():
    a.axis('off')    
plt.show()

im = imread('images/Img_03_31.png')
freq = fp.fft2(im)
freq_uniform = ndimage.fourier_uniform(freq, size=10)

fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(20,10))
plt.gray() # show the result in grayscale
im1 = fp.ifft2(freq_uniform)
axes1.imshow(im), axes1.axis('off')
axes1.set_title('Original Image', size=20)
axes2.imshow(im1.real) # the imaginary part is an artifact
axes2.axis('off')
axes2.set_title('Blurred Image with Fourier Uniform', size=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,10))
plt.imshow( (20*np.log10( 0.1 + fp.fftshift(freq_uniform))).real.astype(int))
plt.title('Frequency Spectrum with fourier uniform', size=20)
plt.show()

freq_ellipsoid = ndimage.fourier_ellipsoid(freq, size=10)
im1 = fp.ifft2(freq_ellipsoid)

fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(20,10))
axes1.imshow(im), axes1.axis('off')
axes1.set_title('Original Image', size=20)
axes2.imshow(im1.real) # the imaginary part is an artifact
axes2.axis('off')
axes2.set_title('Blurred Image with Fourier Ellipsoid', size=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,10))
plt.imshow( (20*np.log10( 0.1 + fp.fftshift(freq_ellipsoid))).real.astype(int))
plt.title('Frequency Spectrum with Fourier ellipsoid', size=20)
plt.show()

# Gaussian Blur LowPass Filter with scipy.fftpack

im = rgb2gray(imread('images/Img_03_11.jpg'))
freq = fp.fft2(im)

kernel = np.outer(signal.gaussian(im.shape[0], 1), signal.gaussian(im.shape[1], 1))

freq_kernel = fp.fft2(fp.ifftshift(kernel))

convolved = freq*freq_kernel # by the Convolution theorem

im_blur = fp.ifft2(convolved).real
im_blur = 255 * im_blur / np.max(im_blur)

plt.figure(figsize=(20,20))
plt.subplot(221), plt.imshow(kernel, cmap='coolwarm'), plt.colorbar()
plt.title('Gaussian Blur Kernel', size=20)

plt.subplot(222)
plt.imshow( (20*np.log10( 0.01 + fp.fftshift(freq_kernel))).real.astype(int), cmap='inferno')
plt.colorbar()
plt.title('Gaussian Blur Kernel (Freq. Spec.)', size=20)
plt.subplot(223), plt.imshow(im, cmap='gray'), plt.axis('off'), plt.title('Input Image', size=20)
plt.subplot(224), plt.imshow(im_blur, cmap='gray'), plt.axis('off'), plt.title('Output Blurred Image', size=20)
plt.tight_layout()
plt.show()

def plot_3d(X, Y, Z, cmap=plt.cm.seismic):
    fig = plt.figure(figsize=(20,20))
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=5, antialiased=False)
    #ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    #ax.set_zscale("log", nonposx='clip')
    #ax.zaxis.set_scale('log')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('F1', size=30)
    ax.set_ylabel('F2', size=30)
    ax.set_zlabel('Freq Response', size=30)
    #ax.set_zlim((-40,10))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf) #, shrink=0.15, aspect=10)
    #plt.title('Frequency Response of the Gaussian Kernel')
    plt.show()    

Y = np.arange(freq.shape[0]) #-freq.shape[0]//2,freq.shape[0]-freq.shape[0]//2)
X = np.arange(freq.shape[1]) #-freq.shape[1]//2,freq.shape[1]-freq.shape[1]//2)
X, Y = np.meshgrid(X, Y)
Z = (20*np.log10( 0.01 + fp.fftshift(freq_kernel))).real
plot_3d(X,Y,Z)

Z = (20*np.log10( 0.01 + fp.fftshift(freq))).real
plot_3d(X,Y,Z)

Z = (20*np.log10( 0.01 + fp.fftshift(convolved))).real
plot_3d(X,Y,Z)

# Convolution in frequency domain with a colored image using fftconvolve from scipy signal

from skimage import img_as_float
from scipy import signal
im = img_as_float(plt.imread('images/Img_03_07.jpg'))

kernel = get_gaussian_edge_blur_kernel(sigma=10, sz=15)
im1 = signal.fftconvolve(im, kernel[:, :, np.newaxis], mode='same')
im1 = im1 / np.max(im1)

kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
im2 = signal.fftconvolve(im, kernel[:, :, np.newaxis], mode='same')
im2 = im2 / np.max(im2)
im2 = np.clip(im2, 0, 1)

plt.figure(figsize=(20,10))
plt.subplot(131), plt.imshow(im), plt.axis('off'), plt.title('original image', size=20)
plt.subplot(132), plt.imshow(im1), plt.axis('off'), plt.title('output with Gaussian LPF', size=20)
plt.subplot(133), plt.imshow(im2), plt.axis('off'), plt.title('output with Laplacian HPF', size=20)
plt.tight_layout()
plt.show()

# Edge Detection with High-Pass Filters (HPF) in the Frequency domain 

def dft2(im):
    
    freq = cv2.dft(np.float32(im), flags = cv2.DFT_COMPLEX_OUTPUT)
    freq_shift = np.fft.fftshift(freq)
    mag, phase = freq_shift[:,:,0], freq_shift[:,:,1]

    return mag + 1j*phase

def idft2(freq):
    
    real, imag = freq.real, freq.imag
    back = cv2.merge([real, imag]) 
    back_ishift = np.fft.ifftshift(back)
    im = cv2.idft(back_ishift, flags=cv2.DFT_SCALE)
    im = cv2.magnitude(im[:,:,0], im[:,:,1])

    return im

def ideal(sz, D0):
    h, w = sz
    u, v = np.meshgrid(range(-w//2,w//2), range(-h//2,h//2)) #, sparse=True)
    return np.sqrt(u**2 + v**2) > D0
    
def gaussian(sz, D0):
    h, w = sz
    u, v = np.meshgrid(range(-w//2,w//2), range(-h//2,h//2)) #, sparse=True)
    return 1-np.exp(-(u**2 + v**2)/(2*D0**2)) 

def butterworth(sz, D0, n=1):
    h, w = sz
    u, v = np.meshgrid(range(-w//2,w//2), range(-h//2,h//2)) #, sparse=True)
    return 1 / (1 + (D0/(0.01+np.sqrt(u**2 + v**2)))**(2*n))

def plot_HPF(im, f, D0s):
    freq = dft2(im) 
    fig = plt.figure(figsize=(20,20))
    plt.subplots_adjust(0,0,1,0.95,0.05,0.05)
    i = 1
    for D0 in D0s:
        freq_kernel = f(im.shape, D0) 
        convolved = freq*freq_kernel # by the Convolution theorem
        im_convolved = idft2(convolved).real 
        im_convolved = (255 * im_convolved / np.max(im_convolved)).astype(np.uint8)
        plt.subplot(2,2,i)
        last_axes = plt.gca()
        img = plt.imshow((20*np.log10(0.01 + freq_kernel)).astype(int), cmap='coolwarm')
        divider = make_axes_locatable(img.axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax)
        plt.sca(last_axes), plt.title('{} HPF Kernel (freq)'.format(f.__name__), size=20)
        plt.subplot(2,2,i+2), plt.imshow(im_convolved), plt.axis('off')
        plt.title(r'output with {} HPF ($D_0$={})'.format(f.__name__, D0), size=20)
        i += 1
    plt.show()

def plot_HPF_3d(im, f, D0s):
    freq = dft2(im) 
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(0,0,1,0.95,0.05,0.05)
    i = 1
    for D0 in D0s:
        freq_kernel = f(im.shape, D0) 
        convolved = freq*freq_kernel # by the Convolution theorem
        Y = np.arange(freq_kernel.shape[0]) 
        X = np.arange(freq_kernel.shape[1]) 
        X, Y = np.meshgrid(X, Y)
        Z = (20*np.log10( 0.01 + convolved)).real
        ax = fig.add_subplot(1, 2, i, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10)), ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_xlabel('F1', size=30), ax.set_ylabel('F2', size=30)
        plt.title(r'output with {} HPF (freq)'.format(f.__name__, D0), size=20)
        fig.colorbar(surf, shrink=0.5, aspect=10)
        i += 1
    plt.show()

def plot_filter_3d(sz, f, D0s, cmap=plt.cm.coolwarm):
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(0,0,1,0.95,0.05,0.05)
    i = 1
    for D0 in D0s:
        freq_kernel = f(sz, D0) 
        Y = np.arange(freq_kernel.shape[0])
        X = np.arange(freq_kernel.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = (20*np.log10( 0.01 + freq_kernel)).real
        ax = fig.add_subplot(1, 3, i, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_xlabel('F1', size=30)
        ax.set_ylabel('F2', size=30)
        ax.set_title('{} HPF Kernel (freq)'.format(f.__name__), size=20)
        fig.colorbar(surf, shrink=0.5, aspect=10)
        i += 1

im = plt.imread('images/Img_03_12.png')
im = rgb2gray(im)
plt.figure(figsize=(7,12))
plt.imshow(im), plt.axis('off'), plt.title('original image')
plt.show()

D0 = [10, 30]

plot_HPF(im, ideal, D0)
plot_HPF_3d(im, ideal, D0)
plot_filter_3d(im.shape, ideal, D0)
plot_HPF(im, gaussian, D0)
plot_HPF_3d(im, gaussian, D0)
plot_filter_3d(im.shape, gaussian, D0)
plot_HPF(im, butterworth, D0)
plot_HPF_3d(im, butterworth, D0)
plot_filter_3d(im.shape, butterworth, D0)

# Homomorphic Filters

from skimage.filters import sobel, threshold_otsu

def homomorphic_filter(im, D0, g_l=0, g_h=1, n=1):
    im_log = np.log(im.astype(np.float)+1)
    im_fft = dft2(im_log)
    H = (g_h - g_l) * butterworth(im.shape, D0, n) + g_l
    #H = np.fft.ifftshift(H)
    im_fft_filt = H*im_fft
    #im_fft_filt = np.fft.ifftshift(im_fft_filt)
    im_filt = idft2(im_fft_filt)
    im = np.exp(im_filt.real)-1
    im = np.uint8(255*im/im.max())
    return im

image = rgb2gray(imread('images/Img_03_13.jpg'))
image_filtered = homomorphic_filter(image, D0=30, n=2, g_l=0.3, g_h=1)

image_edges = sobel(image)
image_edges = image_edges <= threshold_otsu(image_edges)

image_filtered_edges = sobel(image_filtered)
image_filtered_edges = image_filtered_edges <= threshold_otsu(image_filtered_edges)

plt.figure(figsize=(21,17))
plt.gray()
plt.subplots_adjust(0,0,1,0.95,0.01,0.05)
plt.subplot(221), plt.imshow(image), plt.axis('off'), plt.title('original image', size=20)
plt.subplot(222), plt.imshow(image_filtered), plt.axis('off'), plt.title('filtered image', size=20)
plt.subplot(223), plt.imshow(image_edges), plt.axis('off'), plt.title('original image edges', size=20)
plt.subplot(224), plt.imshow(image_filtered_edges), plt.axis('off'), plt.title('filtered image edges', size=20)
plt.show()

# Template matching with Phase-Correlation in Frequency Domain

import scipy.fftpack as fp
from skimage.color import rgb2gray, gray2rgb
from skimage.draw import rectangle_perimeter

im = 255*rgb2gray(imread('images/Img_03_04.jpg'))
im_tm = 255*rgb2gray(imread('images/Img_03_22.png'))

F = fp.fftn(im)
F_tm = fp.fftn(im_tm, shape=im.shape)

F_cc = F * np.conj(F_tm)
c = (fp.ifftn(F_cc/np.abs(F_cc))).real
i, j = np.unravel_index(c.argmax(), c.shape)
print(i, j)

im2 = (gray2rgb(im)).astype(np.uint8)
rr, cc = rectangle_perimeter((i,j), end=(i + im_tm.shape[0], j + im_tm.shape[1]), shape=im.shape)
for x in range(-2,2):
    for y in range(-2,2):
        im2[rr + x, cc + y] = (255,0,0)

plt.figure(figsize=(2,3))
plt.gray()
plt.imshow(im_tm), plt.title('template', size=20), plt.axis('off')
plt.show()
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12,7))
ax[0].imshow(im), ax[0].set_title('target', size=20)
ax[1].imshow(im2), ax[1].set_title('matched template', size=20)
for a in ax.ravel():
    a.set_axis_off()
plt.tight_layout()
plt.show()
Y = np.arange(F_cc.shape[0])
X = np.arange(F_cc.shape[1])
X, Y = np.meshgrid(X, Y)
Z = c
plot_3d(X,Y,Z, cmap='YlOrRd') #PiYG

7: Image Compression with Discrete Cosine Transform (DCT)

from scipy.fftpack import dct, idct

def dct2(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(a):
    return idct(idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')    

im = rgb2gray(imread('images/Img_03_01.jpg')) 
imF = dct2(im)
im1 = idct2(imF)

print(np.allclose(im, im1))

plt.figure(figsize=(10,5))
plt.gray()
plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('original image', size=15)
plt.subplot(122), plt.imshow(im1), plt.axis('off'), plt.title('reconstructed image (DCT+IDCT)', size=15)
plt.tight_layout()
plt.show()

im = rgb2gray(imread('images/Img_03_27.png'))

dct_coeffs = np.zeros(im.shape)

for i in range(0, im.shape[0], 8):
    for j in range(0, im.shape[1], 8):
        dct_coeffs[i:(i+8),j:(j+8)] = dct2(im[i:(i+8),j:(j+8)])

index = 112
plt.figure(figsize=(10,6))
plt.gray()
plt.subplot(121), plt.imshow(im[index:index+8,index:index+8]), plt.title( "An 8x8 Image block", size=15)
plt.subplot(122), plt.imshow(dct_coeffs[index:index+8,index:index+8], vmax= np.max(dct_coeffs)*0.01, vmin = 0, extent=[0, np.pi, np.pi, 0])
plt.title("An 8x8 DCT block", size=15)
plt.show()

thresh = 0.03
dct_thresh = dct_coeffs * (abs(dct_coeffs) > (thresh*np.max(dct_coeffs)))
percent_nonzeros = np.sum( dct_thresh != 0.0 ) / (im.shape[0]*im.shape[1])
print ("Keeping only {}% of the DCT coefficients".format(percent_nonzeros*100.0))

plt.figure(figsize=(12,7))
plt.gray()
plt.subplot(121), plt.imshow(dct_coeffs,cmap='gray',vmax = np.max(dct_coeffs)*0.01,vmin = 0), plt.axis('off')
plt.title("8x8 DCTs of the image", size=15)
plt.subplot(122), plt.imshow(dct_thresh, vmax = np.max(dct_coeffs)*0.01, vmin = 0), plt.axis('off')
plt.title("Thresholded 8x8 DCTs of the image", size=15)
plt.tight_layout()
plt.show()

im_out = np.zeros(im.shape)
for i in range(0, im.shape[0], 8):
    for j in range(0, im.shape[1], 8):
        im_out[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )

plt.figure(figsize=(15,7))
plt.gray()
plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('original image', size=20)
plt.subplot(122), plt.imshow(im_out), plt.axis('off'), plt.title('DCT compressed image', size=20)
plt.tight_layout()
plt.show()

# Image Denoising with Discrete Cosine Transform (DCT)

from skimage import img_as_float
from skimage.restoration import estimate_sigma

im = img_as_float(imread('images/Img_03_07.jpg'))
sigma = 0.25
noisy = im + sigma * np.random.standard_normal(im.shape)
noisy = np.clip(noisy, 0, 1)

sigma_est = np.mean(estimate_sigma(noisy, multichannel=True))
print("estimated noise standard deviation = {}".format(sigma_est))

out = noisy.copy()
cv2.xphoto.dctDenoising(noisy, out, sigma_est)
out = np.clip(out, 0, 1)

plt.figure(figsize=(20,10))
plt.subplot(131), plt.imshow(im), plt.axis('off'), plt.title('original', size=20)
plt.subplot(132), plt.imshow(noisy), plt.axis('off'), plt.title('noisy', size=20)
plt.subplot(133), plt.imshow(out), plt.axis('off'), plt.title('denoised (DCT)', size=20)
plt.tight_layout()
plt.show()

# Deconvolution for Image Deblurring

import SimpleITK as sitk
from skimage import restoration
from skimage.metrics import peak_signal_noise_ratio

. Blur Detection

from scipy.signal import convolve2d

def convolve(im, kernel):
    im1 = convolve2d(im, kernel, mode='same')
    return im1 / np.max(im1)

def check_if_blurry(image, threshold):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    var = cv2.Laplacian(image, cv2.CV_64F).var()
    return 'Var Laplacian = {}\n{}'.format(round(var, 6), 'Blurry' if var < threshold else 'Not Blurry')

def plot_blurry(im, title):
    plt.imshow(im), plt.axis('off'), plt.title(title, size=20)

threshold = 0.01
imlist = []

im = rgb2gray(imread('images/Img_03_07.jpg'))
imlist.append((im, 'original image\n' + check_if_blurry(im, threshold)))

kernel = get_gaussian_edge_blur_kernel(3)
im1 = convolve(im, kernel)
imlist.append((im1, '(edge) blurred image\n' + check_if_blurry(im1, threshold)))

kernel = get_motion_blur_kernel(9, 45)
im1 = convolve(im, kernel)
imlist.append((im1, '(motion) blurred image\n' + check_if_blurry(im1, threshold)))

kernel = get_out_of_focus_kernel(7)
im1 = convolve(im, kernel)
imlist.append((im1, '(out-of-focus) blurred image\n' + check_if_blurry(im1, threshold)))

plt.figure(figsize=(20,7))
plt.gray()
for i in range(len(imlist)):
    im, title = imlist[i]
    plt.subplot(1,4,i+1), plot_blurry(im, title)
plt.tight_layout()
plt.show()

# Non-blind Deblurring with SimpleITK deconvolution filters

import SimpleITK as sitk
im = rgb2gray(imread('images/img_03_35.png'))
psf = get_out_of_focus_kernel(7, 9).astype(np.float) 
im_blur = signal.convolve2d(im, psf, 'same')
im_blur = im_blur / np.max(im_blur)

tkfilter = sitk.InverseDeconvolutionImageFilter()
tkfilter.SetNormalize(True)
im_res_IN = sitk.GetArrayFromImage(tkfilter.Execute (sitk.GetImageFromArray(im_blur), sitk.GetImageFromArray(psf)))

tkfilter = sitk.WienerDeconvolutionImageFilter()
tkfilter.SetNoiseVariance(0)
tkfilter.SetNormalize(True)
im_res_WN = sitk.GetArrayFromImage(tkfilter.Execute (sitk.GetImageFromArray(im_blur), sitk.GetImageFromArray(psf)))

tkfilter = sitk.TikhonovDeconvolutionImageFilter() 
tkfilter.SetRegularizationConstant(0.008) #0.06)
tkfilter.SetNormalize(True)
im_res_TK = sitk.GetArrayFromImage(tkfilter.Execute (sitk.GetImageFromArray(im_blur), sitk.GetImageFromArray(psf)))

tkfilter = sitk.RichardsonLucyDeconvolutionImageFilter() 
tkfilter.SetNumberOfIterations(100)
tkfilter.SetNormalize(True)
im_res_RL = sitk.GetArrayFromImage(tkfilter.Execute (sitk.GetImageFromArray(im_blur), sitk.GetImageFromArray(psf)))

plt.figure(figsize=(20, 60))
plt.subplots_adjust(0,0,1,1,0.07,0.07)
plt.gray()

plt.subplot(611), plt.imshow(im), plt.axis('off'), plt.title('Original Image', size=20)
plt.subplot(612), plt.imshow(im_blur), plt.axis('off'), plt.title('Blurred (out-of-focus) Image, PSNR={:.3f}'.format(peak_signal_noise_ratio(im, im_blur)), size=20)
plt.subplot(613), plt.imshow(im_res_IN, vmin=im_blur.min(), vmax=im_blur.max()), plt.axis('off') 
plt.title('Deconvolution using SimpleITK (Inverse Deconv.), PSNR={:.3f}'.format(peak_signal_noise_ratio(im, im_res_IN)), size=20)
plt.subplot(614), plt.imshow(im_res_WN, vmin=im_blur.min(), vmax=im_blur.max()), plt.axis('off') 
plt.title('Deconvolution using SimpleITK (Wiener Deconv.), PSNR={:.3f}'.format(peak_signal_noise_ratio(im, im_res_WN)), size=20)
plt.subplot(615), plt.imshow(im_res_RL, vmin=im_blur.min(), vmax=im_blur.max()), plt.axis('off')
plt.title('Deconvolution using SimpleITK (Richardson-Lucy), PSNR={:.3f}'.format(peak_signal_noise_ratio(im, im_res_RL)), size=20)
plt.subplot(616), plt.imshow(im_res_TK, vmin=im_blur.min(), vmax=im_blur.max()), plt.axis('off')
plt.title('Deconvolution using SimpleITK (Tikhonov Deconv.), PSNR={:.3f}'.format(peak_signal_noise_ratio(im, im_res_TK)), size=20)

plt.show()

# Non-blind Deblurring with scikit-image restoration module functions

from skimage import restoration

im_res_RL = restoration.richardson_lucy(im_blur, psf, iterations=20)

plt.figure(figsize=(20, 15))
plt.subplots_adjust(0,0,1,1,0.07,0.07)
plt.gray()
plt.imshow(im_res_RL, vmin=im_blur.min(), vmax=im_blur.max()), plt.axis('off')
plt.title('Deconvolution using skimage (Richardson-Lucy), PSNR={:.3f}'.format(peak_signal_noise_ratio(im, im_res_RL)), size=20)
plt.show()

10: Image Denoising with Wavelets

from numpy import *
import scipy
import pywt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
from skimage.filters import threshold_otsu
from skimage import img_as_float

. Wavelet Basics 

x = pywt.data.ascent().astype(np.float32)
shape = x.shape

plt.rcParams.update({'font.size': 20})

max_lev = 3       # how many levels of decomposition to draw
label_levels = 3  # how many levels to explicitly label on the plots

fig, axes = plt.subplots(4, 2, figsize=[15, 35])
plt.subplots_adjust(0, 0, 1, 0.95, 0.05, 0.05)
for level in range(0, max_lev + 1):
    if level == 0:
        # show the original image before decomposition
        axes[0, 0].set_axis_off()
        axes[0, 1].imshow(x, cmap=plt.cm.gray)
        axes[0, 1].set_title('Image')
        axes[0, 1].set_axis_off()
        continue

    # plot subband boundaries of a standard DWT basis
    draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[level, 0],
                     label_levels=label_levels)
    axes[level, 0].set_title('{} level\ndecomposition'.format(level))

    # compute the 2D DWT
    c = pywt.wavedec2(x, 'db2', mode='periodization', level=level)
    # normalize each coefficient array independently for better visibility
    c[0] /= np.abs(c[0]).max()
    for detail_level in range(level):
        c[detail_level + 1] = [d/np.abs(d).max() > threshold_otsu(d/np.abs(d).max()) for d in c[detail_level + 1]]
    # show the normalized coefficients
    arr, slices = pywt.coeffs_to_array(c)
    axes[level, 1].imshow(arr, cmap=plt.cm.gray)
    axes[level, 1].set_title('Coefficients\n({} level)'.format(level))
    axes[level, 1].set_axis_off()

plt.tight_layout()
plt.show()

#  Image Denoising using Wavelets with pywt

image = img_as_float(imread('images/Img_03_01.jpg')) 
noise_sigma = 0.25 #16.0
image += random.normal(0, noise_sigma, size=image.shape)

wavelet = pywt.Wavelet('haar')
levels  = int(floor(log2(image.shape[0])))
print(levels)
wavelet_coeffs = pywt.wavedec2(image, wavelet, level=levels)

def denoise(image, wavelet, noise_sigma):
    levels = int(floor(log2(image.shape[0])))
    wc = pywt.wavedec2(image, wavelet, level=levels)
    nwc = list(map(lambda x: pywt.threshold(x, noise_sigma, mode='soft'), wc))
    return pywt.waverec2(nwc, wavelet)

print(pywt.wavelist(kind='discrete'))
wlts = ['bior1.5', 'coif5', 'db6', 'dmey', 'haar', 'rbio2.8', 'sym15'] # pywt.wavelist(kind='discrete')
Denoised={}
for wlt in wlts:
    out = image.copy()
    for i in range(3):
        out[...,i] = denoise(data=image[...,i], wavelet=wlt, noise_sigma=3/2*noise_sigma)
    Denoised[wlt] = np.clip(out, 0, 1)
print(len(Denoised))

plt.figure(figsize=(15,8))
plt.subplots_adjust(0,0,1,0.9,0.05,0.07)
plt.subplot(241), plt.imshow(np.clip(image,0,1)), plt.axis('off'), plt.title('original image', size=15)
i = 2
for wlt in Denoised:
    plt.subplot(2,4,i), plt.imshow(Denoised[wlt]), plt.axis('off'), plt.title(wlt, size=15)
    i += 1
plt.suptitle('Image Denoising with Wavelets', size=20)
plt.show()

# Image Denoising with Wavelets using scikit-image restoration

from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio

original = img_as_float(imread('images/Img_03_08.png'))[...,:3]
sigma = 0.12
noisy = random_noise(original, var=sigma**2)

sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)
print(f"Estimated Gaussian noise standard deviation = {sigma_est}")

im_bayes = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)
im_visushrink = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                method='VisuShrink', mode='soft',
                                sigma=sigma_est, rescale_sigma=True)

im_visushrink2 = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                 method='VisuShrink', mode='soft',
                                 sigma=sigma_est/2, rescale_sigma=True)
im_visushrink4 = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                 method='VisuShrink', mode='soft',
                                 sigma=sigma_est/4, rescale_sigma=True)

psnr_noisy = peak_signal_noise_ratio(original, noisy)
psnr_bayes = peak_signal_noise_ratio(original, im_bayes)
psnr_visushrink = peak_signal_noise_ratio(original, im_visushrink)
psnr_visushrink2 = peak_signal_noise_ratio(original, im_visushrink2)
psnr_visushrink4 = peak_signal_noise_ratio(original, im_visushrink4)

plt.figure(figsize=(20,20))
plt.subplots_adjust(0,0,1,1,0.05,0.05)
plt.subplot(231), plt.imshow(original), plt.axis('off'), plt.title('Original', size=20)
plt.subplot(232), plt.imshow(noisy), plt.axis('off'), plt.title('Noisy\nPSNR={:0.4g}'.format(psnr_noisy), size=20)
plt.subplot(233), plt.imshow(im_bayes/im_bayes.max()), plt.axis('off'), plt.title('Wavelet denoising\n(BayesShrink)\nPSNR={:0.4f}'.format(psnr_bayes), size=20)
plt.subplot(234), plt.imshow(im_visushrink/im_visushrink.max()), plt.axis('off')
plt.title('Wavelet denoising\n' + r'(VisuShrink, $\sigma=\sigma_{est}$)' + '\nPSNR={:0.4g}'.format(psnr_visushrink), size=20)
plt.subplot(235), plt.imshow(im_visushrink2/im_visushrink2.max()), plt.axis('off')
plt.title('Wavelet denoising\n' + r'(VisuShrink, $\sigma=\sigma_{est}/2$)' + '\nPSNR={:0.4g}'.format(psnr_visushrink2), size=20)
plt.subplot(236), plt.imshow(im_visushrink4/im_visushrink4.max()), plt.axis('off')
plt.title('Wavelet denoising\n' + r'(VisuShrink, $\sigma=\sigma_{est}/4$)' + '\nPSNR={:0.4g}'.format(psnr_visushrink4), size=20)
plt.show()

# Image Fusion with Wavelets

import pywt
import cv2
import numpy as np

def fuseCoeff(cooef1, cooef2, method):

    if (method == 'mean'):
        cooef = (cooef1 + cooef2) / 2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    else:
        cooef = []

    return cooef

fusion_method = 'mean' 

im1 = cv2.imread('images/Img_03_05.jpg',0)
im2 = cv2.imread('images/Img_03_06.jpg',0)

im2 = cv2.resize(im2,(im1.shape[1], im1.shape[0])) # I do this just because i used two random images

wavelet = 'sym2' #'bior1.1' #'haar' #'db1'
cooef1 = pywt.wavedec2(im1[:,:], wavelet)
cooef2 = pywt.wavedec2(im2[:,:], wavelet)

fused_cooef = []
for i in range(len(cooef1)):
    # The first values in each decomposition is the apprximation values of the top level
    if(i == 0):
        fused_cooef.append(fuseCoeff(cooef1[0], cooef2[0], fusion_method))
    else:
        # For the rest of the levels we have tupels with 3 coeeficents
        c1 = fuseCoeff(cooef1[i][0], cooef2[i][0],fusion_method)
        c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], fusion_method)
        c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], fusion_method)
        fused_cooef.append((c1,c2,c3))

fused_image = pywt.waverec2(fused_cooef, wavelet)

fused_image = 255*fused_image / np.max(fused_image)
fused_image = fused_image.astype(np.uint8)

plt.figure(figsize=(20,20))
plt.subplot(221), plt.imshow(im1), plt.axis('off'), plt.title('Image1', size=20) #cv2.cvtColor(fused_image,cv2.COLOR_BGR2RGB))
plt.subplot(222), plt.imshow(im2), plt.axis('off'), plt.title('Image2', size=20) #cv2.cvtColor(fused_image,cv2.COLOR_BGR2RGB))

plt.subplot(223), plt.imshow(im1//2 + im2// 2), plt.axis('off'), plt.title('Average Image', size=20) #cv2.cvtColor(fused_image,cv2.COLOR_BGR2RGB))

plt.subplot(224), plt.imshow(fused_image), plt.axis('off'), plt.title('Fused Image with Wavelets', size=20) #cv2.cvtColor(fused_image,cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()

# Secure Spread Spectrum Digital Watermarking with DCT

from scipy.fftpack import dct, idct
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    

def embed(im, k, alpha):
    m, n = im.shape
    d = dct2(im)
    indices = np.dstack(np.unravel_index(np.argsort(d.ravel()), (m, n)))[0]
    indices = indices[-(k+1):-1]
    cw = d[indices[:,0], indices[:,1]]
    w = np.random.randn(k)
    #w = w / np.linalg.norm(w)
    ci = cw * (1 + alpha * w)
    d[indices[:,0], indices[:,1]] = ci
    im1 = idct2(d)
    return im1, indices, cw, w

def detect(test, indices, cw, w, alpha):
    d = dct2(test)
    testc = d[indices[:,0], indices[:,1]]
    what = (testc/cw - 1) / alpha
    gamma = what@w/(np.linalg.norm(what)) #*np.linalg.norm(w))
    return gamma

k = 1000
alpha = 0.1
im = rgb2gray(imread('images/Img_03_01.jpg'))
im = (255*im).astype(np.uint8)
im1, indices, cw, w = embed(im, k=k, alpha=alpha)
print('mean difference={}, max difference={}'.format(np.mean(np.abs(im1-im)), np.max(np.abs(im1-im))))
similarity = detect(im1, indices, cw, w, alpha)
print('detected similarity={}'.format(similarity))

fig = plt.figure(figsize=(20,10))
plt.gray()
plt.subplots_adjust(0,0,1,0.925,0.05,0.05)
plt.subplot(131), plt.imshow(im), plt.axis('off'), plt.title('original image {}'.format(im.shape), size=20)
plt.subplot(132), plt.imshow(im1), plt.axis('off'), plt.title(r'watermarked image: $v_i^{\prime}=v_i.(1+\alpha x_i)$', size=20)
plt.subplot(133)
last_axes = plt.gca()
img = plt.imshow((np.abs(im1-im)).astype(np.uint8))
divider = make_axes_locatable(img.axes)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(img, cax=cax)
plt.sca(last_axes)
plt.axis('off'), plt.title('difference image', size=20)
plt.show()

