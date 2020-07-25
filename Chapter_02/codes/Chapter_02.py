# Chapter 2: Image Transformation and Facial Image Processing
# Author: Sandipan Dey

# Problem1. Implement Affine Transformation with scipy.ndimage

from skimage.io import imread
from scipy.ndimage import affine_transform
import numpy as np
import matplotlib.pylab as plt

im = imread("images/Img_02_01.jpg")
rot_mat = np.array([[np.cos(np.pi/4),np.sin(np.pi/4), 0],[-np.sin(np.pi/4),np.cos(np.pi/4), 0], [0,0,1]])
shr_mat = np.array([[1, 0.45, 0], [0, 0.75, 0], [0, 0, 1]])
transformed = affine_transform(im, rot_mat@shr_mat, offset=[-im.shape[0]/4+25, im.shape[1]/2-50, 0], output_shape=im.shape)
plt.figure(figsize=(20,10))
plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('Input image', size=20)
plt.subplot(122), plt.imshow(transformed), plt.axis('off'), plt.title('Output image', size=20)
plt.show()

# Problem2. Implement Image Transformation with Warping / Inverse Warping using scikit-image and scipy.ndimage
 
# Applying translation on an image using scikit-image warp
from skimage.io import imread
from skimage.transform import warp
import matplotlib.pylab as plt

def translate(xy, t_x, t_y):
 xy[:, 0] -= t_y
 xy[:, 1] -= t_x
 return xy
im = imread('images/Img_02_01.jpg')
im = warp(im, translate, map_args={'t_x':-250, 't_y':200}) # create a dictionary for translation parameters
plt.imshow(im)
plt.title('Translated image', size=20)
plt.show()
 
# Implementing the swirl transformation using scikit-image warp
def swirl(xy, x0, y0, R):
    r = np.sqrt((xy[:,1]-x0)**2 + (xy[:,0]-y0)**2)
    a = np.pi*r / R
    xy[:, 1] = (xy[:, 1]-x0)*np.cos(a) + (xy[:, 0]-y0)*np.sin(a) + x0
    xy[:, 0] = -(xy[:, 1]-x0)*np.sin(a) + (xy[:, 0]-y0)*np.cos(a) + y0
    return xy

im = imread('images/Img_02_02.jpg')
print(im.shape)
im1 = warp(im, swirl, map_args={'x0':220, 'y0':360, 'R':650})
plt.figure(figsize=(20,10))
plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('Input image', size=20)
plt.subplot(122), plt.imshow(im1), plt.axis('off'), plt.title('Output image', size=20)
plt.show()

# Implementing Swirl Transform using *scipy.ndimage* 
from scipy import ndimage as ndi
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pylab as plt, numpy as np

def apply_swirl(xy, x0, y0, R):                                                                        
    r = np.sqrt((xy[1]-x0)**2 + (xy[0]-y0)**2)
    a = np.pi*r / R
    return ((xy[1]-x0)*np.cos(a) + (xy[0]-y0)*np.sin(a) + x0, -(xy[1]-x0)*np.sin(a) + (xy[0]-y0)*np.cos(a) + y0)

im = rgb2gray(imread('images/Img_02_06.jpg'))
print(im.shape)
im1 = ndi.geometric_transform(im, apply_swirl, extra_arguments=(100, 100, 250))
plt.figure(figsize=(20,10))
plt.gray()
plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('Input image', size=20)
plt.subplot(122), plt.imshow(im1), plt.axis('off'), plt.title('Output image', size=20)
plt.show()

# Implementing Elastic Deformation 
import numpy as np
import matplotlib.pylab as plt
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter, map_coordinates 

def elastic_transform(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    h, w = image.shape
    dx = gaussian_filter((random_state.rand(*image.shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*image.shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

img = rgb2gray(plt.imread('images/Img_02_22.png'))
img1 = elastic_transform(img, 100, 4)
plt.figure(figsize=(20,10))
plt.subplot(121), plt.imshow(img), plt.axis('off'), plt.title('Original', size=20)
plt.subplot(122), plt.imshow(img1), plt.axis('off'), plt.title('Deformed', size=20)
plt.tight_layout()
plt.show()

# Problem3. Image Projection with Homography using scikit-image
from skimage.transform import ProjectiveTransform
from skimage.io import imread
import numpy as np
import matplotlib.pylab as plt
from matplotlib.path import Path

im_src = imread('images/Img_02_04.jpg')
im_dst = imread('images/Img_02_03.jpg')
print(im_src.shape, im_dst.shape)
pt = ProjectiveTransform()
width, height = im_src.shape[0], im_src.shape[1]
src = np.array([[   0.,    0.],
       [height-1,    0.],
       [height-1,  width-1],
       [   0.,  width-1]])
dst = np.array([[ 74.,  41.],
       [ 272.,  96.],
       [ 272.,  192.],
       [ 72.,  228.]])
pt.estimate(src, dst)
width, height = im_dst.shape[0], im_dst.shape[1]
polygon = dst
poly_path = Path(polygon)
x, y = np.mgrid[:height, :width]
coors = np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) 
mask = poly_path.contains_points(coors)
mask = mask.reshape(height, width)
dst_indices = np.array([list(x) for x in list(zip(*np.where(mask > 0)))])
src_indices = np.round(pt.inverse(dst_indices), 0).astype(int)
src_indices[:,0], src_indices[:,1] = src_indices[:,1], src_indices[:,0].copy()
im_out = np.copy(im_dst)
im_out[dst_indices[:,1], dst_indices[:,0]] = im_src[src_indices[:,0], src_indices[:,1]]
plt.figure(figsize=(30,10))
plt.subplot(131), plt.imshow(im_src, cmap='gray'), plt.axis('off'), plt.title('Source image', size=30)
plt.subplot(132), plt.imshow(im_dst, cmap='gray'), plt.axis('off'), plt.title('Destination image', size=30)
plt.subplot(133), plt.imshow(im_out, cmap='gray'), plt.axis('off'), plt.title('Output image', size=30)
plt.tight_layout()
plt.show()

# Problem4: Face morphing with dlib, scipy.spatial and opencv-python
from scipy.spatial import Delaunay
from skimage.io import imread
import scipy.misc
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt

def extract_landmarks(img, add_boundary_points=True, predictor_path = 'models/shape_predictor_68_face_landmarks.dat'):
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(predictor_path)
  try:
    #points = stasm.search_single(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    dets = detector(img, 1)
    points = np.zeros((68, 2))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        for i in range(68):
            points[i, 0] = shape.part(i).x
            points[i, 1] = shape.part(i).y
  except Exception as e:
    print('Failed finding face points: ', e)
    return []
  points = points.astype(np.int32)
  return points

src_path = 'images/Img_02_04.jpg'
dst_path = 'images/Img_02_05.jpg'
src_img = imread(src_path)
dst_img = imread(dst_path)
src_points = extract_landmarks(src_img)
dst_points = extract_landmarks(dst_img)

def weighted_average_points(start_points, end_points, percent=0.5):
  if percent <= 0:
    return end_points
  elif percent >= 1:
    return start_points
  else:
    return np.asarray(start_points*percent + end_points*(1-percent), np.int32)

def bilinear_interpolate(img, coords):
  int_coords = np.int32(coords)
  x0, y0 = int_coords
  dx, dy = coords - int_coords
  # 4 Neighour pixels
  q11 = img[y0, x0]
  q21 = img[y0, x0+1]
  q12 = img[y0+1, x0]
  q22 = img[y0+1, x0+1]
  btm = q21.T * dx + q11.T * (1 - dx)
  top = q22.T * dx + q12.T * (1 - dx)
  inter_pixel = top * dy + btm * (1 - dy)
  return inter_pixel.T

def get_grid_coordinates(points):
  xmin = np.min(points[:, 0])
  xmax = np.max(points[:, 0]) + 1
  ymin = np.min(points[:, 1])
  ymax = np.max(points[:, 1]) + 1
  return np.asarray([(x, y) for y in range(ymin, ymax)
                     for x in range(xmin, xmax)], np.uint32)

def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
  roi_coords = get_grid_coordinates(dst_points)
  # indices to vertices. -1 if pixel is not in any triangle
  roi_tri_indices = delaunay.find_simplex(roi_coords)
  for simplex_index in range(len(delaunay.simplices)):
    coords = roi_coords[roi_tri_indices == simplex_index]
    num_coords = len(coords)
    out_coords = np.dot(tri_affines[simplex_index], np.vstack((coords.T, np.ones(num_coords))))
    x, y = coords.T
    result_img[y, x] = bilinear_interpolate(src_img, out_coords)
  return None

def get_triangular_affine_matrices(vertices, src_points, dest_points):
  ones = [1, 1, 1]
  for tri_indices in vertices:
    src_tri = np.vstack((src_points[tri_indices, :].T, ones))
    dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
    mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
    yield mat

def warp_image(src_img, src_points, dest_points, dest_shape, dtype=np.uint8):
  num_chans = 3
  src_img = src_img[:, :, :3]
  rows, cols = dest_shape[:2]
  result_img = np.zeros((rows, cols, num_chans), dtype)
  delaunay = Delaunay(dest_points)
  tri_affines = np.asarray(list(get_triangular_affine_matrices(delaunay.simplices, src_points, dest_points)))
  process_warp(src_img, result_img, tri_affines, dest_points, delaunay)
  return result_img, delaunay

fig = plt.figure(figsize=(20,10))
plt.subplot(121)
plt.imshow(src_img)
for i in range(68):
    plt.plot(src_points[i,0], src_points[i,1], 'r.', markersize=20)
plt.title('Source image', size=20)
plt.axis('off')
plt.subplot(122)
plt.imshow(dst_img)
for i in range(68):
    plt.plot(dst_points[i,0], dst_points[i,1], 'g.', markersize=20)
plt.title('Destination image', size=20)
plt.axis('off')
plt.suptitle('Facial Landmarks computed for the images', size=30)
fig.subplots_adjust(wspace=0.01, left=0.1, right=0.9)
plt.show()
fig = plt.figure(figsize=(20,20))
i = 1
for percent in np.linspace(1, 0, 9):
    points = weighted_average_points(src_points, dst_points, percent)
    src_face, src_d = warp_image(src_img, src_points, points, size)
    end_face, end_d = warp_image(dst_img, dst_points, points, size)
    average_face = weighted_average(src_face, end_face, percent)
    average_face = alpha_image(average_face, points) if alpha else average_face
    plt.subplot(3, 3, i)
    plt.imshow(average_face)
    plt.title('alpha=' + str(percent), size=20)
    plt.axis('off')
    i += 1
plt.suptitle('Face morphing', size=30)
fig.subplots_adjust(top=0.92, bottom=0, left=0.075, right=0.925, wspace=0.01, hspace=0.05)
plt.show()
fig = plt.figure(figsize=(20,10))
plt.subplot(121)
plt.imshow(src_img)
plt.triplot(src_points[:,0], src_points[:,1], src_d.simplices.copy())
plt.plot(src_points[:,0], src_points[:,1], 'o', color='red')
plt.title('Source image', size=20)
plt.axis('off')
plt.subplot(122)
plt.imshow(dst_img)
plt.triplot(dst_points[:,0], dst_points[:,1], end_d.simplices.copy())
plt.plot(dst_points[:,0], dst_points[:,1], 'o')
plt.title('Destination image', size=20)
plt.axis('off')
plt.suptitle('Delaunay triangulation of the images', size=30)
fig.subplots_adjust(wspace=0.01, left=0.1, right=0.9)
plt.show()

# Problem5: Facial Landmark Detection with Deep Learning Models

# Facial Landmark Detection with Keras
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from keras.models import load_model
from keras.utils.generic_utils import custom_object_scope
from keras import backend as K
import cv2
huber_delta = 0.5

def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < huber_delta, 0.5 * x ** 2, huber_delta * (x - 0.5 * huber_delta))
    return  K.sum(x)
weights = np.empty((136,)) 
weights[0:33] = 0.5
weights[33:53] = 1
weights[53:71] = 2
weights[71:95] = 3
weights[95:] = 1

def mask_weights(y_true, y_pred):
    x = K.abs(y_true - y_pred) * weights
    return K.sum(x)

def relu6(x):
    return K.relu(x, max_value=6)

model = "models/Mobilenet_v1.hdf5"
image_color = cv2.resize(cv2.imread('images/Img_02_04.jpg'), (64, 64))
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
w, h = image_gray.shape
with custom_object_scope({'smoothL1': smoothL1, 'relu6': relu6, 
    'mask_weights': mask_weights, 'tf': tf}):
    sess = load_model(model)
    predictions = sess.predict_on_batch(np.reshape(image_gray, (1, w, h, 1)))
marks = np.array(predictions).flatten()
marks = np.reshape(marks, (-1, 2))
print(marks.shape)

image = image_color.copy()
plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Original image', size=20)
for mark in marks[:17]:
    cv2.circle(image, (int(w*mark[0]), int(h*mark[1])), 1, (0,0,255), -1, cv2.LINE_AA)
for mark in marks[17:27]:
    cv2.circle(image, (int(w*mark[0]), int(h*mark[1])), 1, (255,255,0), -1, cv2.LINE_AA)
for mark in marks[27:31]:
    cv2.circle(image, (int(w*mark[0]), int(h*mark[1])), 1, (0,255,0), -1, cv2.LINE_AA)
for mark in marks[31:36]:
    cv2.circle(image, (int(w*mark[0]), int(h*mark[1])), 1, (0,255,255), -1, cv2.LINE_AA)
for mark in marks[36:48]:
    cv2.circle(image, (int(w*mark[0]), int(h*mark[1])), 1, (255,255,255), -1, cv2.LINE_AA)
for mark in marks[48:]:
    cv2.circle(image, (int(w*mark[0]), int(h*mark[1])), 1, (255,0,255), -1, cv2.LINE_AA)
plt.subplot(122), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Detected Facial Landmarks', size=20)
plt.show()

# Facial Landmark Detection with MTCNN
from mtcnn import MTCNN
import cv2
from skimage.draw import rectangle_perimeter
import matplotlib.pylab as plt

img = cv2.cvtColor(cv2.imread("images/Img_02_04.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()
res = detector.detect_faces(img)
print(res)
plt.figure(figsize=(5,8))
box = res[0]['box']
rr, cc = rectangle_perimeter(box[0:2][::-1], end=box[2:4][::-1], shape=img.shape)
for k in range(-2,2):
    img[rr+k, cc+k] = np.array([255,0,0])
kp = res[0]['keypoints']
plt.plot(kp['left_eye'][0], kp['left_eye'][1], 'co', markersize=12, label='left_eye')
plt.plot(kp['right_eye'][0], kp['right_eye'][1], 'bo', markersize=12, label='right_eye')
plt.plot(kp['nose'][0], kp['nose'][1], 'go', markersize=12, label='nose')
plt.plot(kp['mouth_left'][0], kp['mouth_left'][1], 'mo', markersize=12, label='mouth_left')
plt.plot(kp['mouth_right'][0], kp['mouth_right'][1], 'mo', markersize=12, label='mouth_right')
plt.imshow(img)
plt.legend(loc='best')
plt.axis('off')
plt.title('Facial keypoints with MTCNN', size=20)
plt.show()

# Problem6: Face Swapping
import cv2
import dlib
import numpy as np
import matplotlib.pylab as plt
import imutils

predictor_path = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
feather_amount = 11
color_correction_blur = 0.5
keypoints = list(range(17,68))
left_eye_points, right_eye_points = list(range(42,48)), list(range(36,42))

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_landmarks(img):    
    rects = detector(img, 0)
    if len(rects) == 0:
        return -1
    return np.array([[p.x, p.y] for p in predictor(img, rects[0]).parts()])

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)
    for group in [keypoints]:
        draw_convex_hull(im, landmarks[group], color=1)
    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = cv2.GaussianBlur(im, (feather_amount, feather_amount), 0) > 0
    im = im * 1.0
    im = cv2.GaussianBlur(im, (feather_amount, feather_amount), 0)
    return im

def correct_colours(im1, im2, landmarks1):
    mean_left = np.mean(landmarks1[left_eye_points], axis=0)
    mean_right = np.mean(landmarks1[right_eye_points], axis=0)
    blur_amount = color_correction_blur * np.linalg.norm(mean_left - mean_right)
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0: # make the blur kernel size odd
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    # avoid division errors
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    return (im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64))

def plot_image_landmarks(img, img_landmarks, swap_img, swap_img_landmarks):
    img = img.copy()
    for mark in img_landmarks.tolist():
        cv2.circle(img, (mark[0], mark[1]), 1, (0,0,255), 2, cv2.LINE_AA)
    swap_img = swap_img.copy()
    for mark in swap_img_landmarks.tolist():
        cv2.circle(swap_img, (mark[0], mark[1]), 1, (0,0,255), 2, cv2.LINE_AA)        
    plt.figure(figsize=(15,10))
    plt.subplots_adjust(0,0,1,0.95,0.01,0.01)
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.axis('off')
    plt.subplot(122), plt.imshow(cv2.cvtColor(swap_img.astype(np.uint8), cv2.COLOR_BGR2RGB)), plt.axis('off')
    plt.suptitle('Facial landmarks computed for the faces to be swapped (with dlib shape-predictor)', size=20)
    plt.show()

def face_swap_filter(img_file, swap_img_file):
    
    img = imutils.resize(cv2.imread(img_file), width=400)
    swap_img = imutils.resize(cv2.imread(swap_img_file), width=400)
    
    img_landmarks = get_landmarks(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    swap_img_landmarks = get_landmarks(cv2.cvtColor(swap_img, cv2.COLOR_BGR2GRAY))
    
    plot_image_landmarks(img, img_landmarks, swap_img, swap_img_landmarks)
    
    Hmat, status = cv2.findHomography(swap_img_landmarks[keypoints], img_landmarks[keypoints])
    
    mask = get_face_mask(swap_img, swap_img_landmarks)
    warped_mask = cv2.warpPerspective(mask, Hmat, (img.shape[1], img.shape[0]))
    combined_mask = np.max([get_face_mask(img, img_landmarks), warped_mask], axis=0)    
    warped_swap = cv2.warpPerspective(swap_img, Hmat, (img.shape[1], img.shape[0]))
    output_img = np.clip(img * (1.0 - combined_mask) + warped_swap * combined_mask, 0, 255)
    warped_corrected_swap = correct_colours(img, warped_swap, img_landmarks)
    output_img_corrected = np.clip(img * (1.0 - combined_mask) + warped_corrected_swap * combined_mask, 0, 255)
    
    return (output_img, output_img_corrected)

output_img, output_img_corrected = face_swap_filter('images/Img_02_07.jpg', 'images/Img_02_08.jpg')
plt.figure(figsize=(15,10))
plt.subplots_adjust(0,0,1,0.925,0.01,0.01)
plt.subplot(121), plt.imshow(cv2.cvtColor(output_img.astype(np.uint8), cv2.COLOR_BGR2RGB)), plt.axis('off')
plt.title('Before color correction', size=15)
plt.subplot(122), plt.imshow(cv2.cvtColor(output_img_corrected.astype(np.uint8), cv2.COLOR_BGR2RGB)), plt.axis('off')
plt.title('After color correction', size=15)
plt.suptitle('Face Swapping Output', size=20)
plt.show()

# Problem7: Face Parsing

import matplotlib.pylab as plt
from mtcnn import MTCNN
from skimage.io import imread
from skimage.draw import rectangle_perimeter
from skimage.util import crop
import tensorflow as tf
import json
from keras.models import model_from_json
from skimage.transform import resize
from skimage.color import rgb2gray
import cv2
from matplotlib.colors import ListedColormap

with open("models/BiSeNet_keras.json", 'r') as json_file:
    model = json.load(json_file)
model = model_from_json(model_json, custom_objects={"tf": tf})
model.load_weights('models/BiSeNet_keras.h5')

def normalize_input(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # x should be RGB with range [0, 255]
        return ((x / 255) - mean)  / std

parsing_annos = [
    '0, background', '1, skin', '2, left eyebrow', '3, right eyebrow', 
    '4, left eye', '5, right eye', '6, glasses', '7, left ear', '8, right ear', '9, earings',
    '10, nose', '11, mouth', '12, upper lip', '13, lower lip', 
    '14, neck', '15, neck_l', '16, cloth', '17, hair', '18, hat'
]

def show_parsing_with_annos(img, f, ax):
    #get discrete colormap
    cmap = plt.get_cmap('gist_ncar', len(parsing_annos))
    new_colors = cmap(np.linspace(0, 1, len(parsing_annos)))
    new_colors[0, :] = np.array([0, 0, 0, 1.])
    new_cmap = ListedColormap(new_colors)
    # set limits .5 outside true range
    mat = ax.matshow(img, cmap=new_cmap, vmin=-0.5, vmax=18.5)
    #tell the colorbar to tick at integers    
    cbar = f.colorbar(mat, ticks=np.arange(0, len(parsing_annos)))
    cbar.ax.set_yticklabels(parsing_annos)
    ax.axis('off')

img = imread("images/Img_02_04.jpg")
h, w = img.shape[:2]
detector = MTCNN()
faces = detector.detect_faces(img)

for face in faces:
    bb = face['box']
    face = crop(img,((bb[1],h-(bb[1]+bb[3])),(bb[0],w-(bb[0]+bb[2])),(0,0)))
    rr, cc = rectangle_perimeter((bb[1], bb[0]), extent=(bb[3], bb[2]), shape=img.shape)
    for k in range(-1,2):
        img[rr+k, cc+k,:] = [255,0,0]
    # Preprocess input face for parser networks
    orig_h, orig_w = face.shape[:2]
    inp = 255*resize(face, (512,512))
    inp = normalize_input(inp)
    inp = inp[None, ...]
    # Parser networks forward pass
    # Do NOT use bilinear interp. which adds artifacts to the parsing map
    out = model.predict([inp])[0]
    parsing_map = out.argmax(axis=-1)
    parsing_map = cv2.resize(
        parsing_map.astype(np.uint8), 
        (orig_w, orig_h), 
        interpolation=cv2.INTER_NEAREST)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
    plt.subplots_adjust(0,0,1,0.95,0.01,0.01)
    ax1.imshow(img), ax1.axis('off'), ax1.set_title('original image', size=20)
    show_parsing_with_annos(parsing_map, f, ax2), ax2.set_title('parsed face', size=20)
    plt.show()
