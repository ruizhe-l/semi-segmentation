import os
import shutil
import Augmentor
import numpy as np
import scipy.io as sio
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from PIL import Image

def recale_array(array, nmin=None, nmax=None):
    array = np.array(array)
    if nmin is None:
        nmin = np.min(array)
    array = array - nmin
    if nmax is None:
        nmax = np.max(array) + 1e-10
    array = array / nmax
    array = (array * 255).astype(np.uint8)

    return array


def combine_images(x, y, pred, mask=None):
    pred = np.array(pred)
    if pred.ndim == 4:
        return combine_2d_images(x, y, pred)
    elif pred.ndim == 5:
        return combine_3d_images(x, y, mask, pred)
    else:
        raise 'Unknow dimensions of ouput image!'

def grey_to_rgb(x):
    return np.stack((x,)*3, axis=-1)


def combine_2d_images(x, y, pred):

    n_class = pred.shape[-1]
    n_y = pred.shape[2]
    x_img = recale_array(x)

    imgs = []
    # result for classes
    if x_img.shape[-1] != 3:
        x_img = grey_to_rgb(x_img)

    for i in range(n_class):
        img = np.concatenate((x_img.reshape(-1, n_y, 3),
                            grey_to_rgb(recale_array(y[..., i]).reshape(-1, n_y)),
                            grey_to_rgb(recale_array(pred[..., i]).reshape(-1, n_y))),
                            axis=1)
        imgs.append(img)

    # result for argmax
    pred_max = np.argmax(pred, axis=-1)
    y_max = np.argmax(y, axis=-1)
    img = np.concatenate((x_img.reshape(-1, n_y, 3),
                          grey_to_rgb(recale_array(y_max, nmin=0, nmax=n_class-1).reshape(-1, n_y)),
                          grey_to_rgb(recale_array(pred_max, nmin=0, nmax=n_class-1).reshape(-1, n_y))),
                         axis=1)
    imgs.append(img)

    return imgs



def combine_3d_images(x, y, mask, pred):
    n_class = pred.shape[-1]
    n_y = pred.shape[2]
    n_z = pred.shape[3]
    x_img = recale_array(x)

    imgs = []
    # result for classes
    for i in range(n_class):
        img = np.concatenate((x_img.transpose((0,1,3,2,4)).reshape(-1, n_y, order='F'),
                              recale_array(y[..., i]).transpose((0,1,3,2)).reshape(-1, n_y, order='F'),
                              recale_array(mask[..., i]).transpose((0,1,3,2)).reshape(-1, n_y, order='F'),
                              recale_array(pred[..., i]).transpose((0,1,3,2)).reshape(-1, n_y, order='F')),
                             axis=1)
        imgs.append(img)

    # result for argmax
    pred_max = np.argmax(pred, axis=-1)
    y_max = np.argmax(y, axis=-1)
    img = np.concatenate((x_img.transpose((0,1,3,2,4)).reshape(-1, n_y, order='F'),
                              recale_array(y_max).transpose((0,1,3,2)).reshape(-1, n_y, order='F'),
                              recale_array(mask[...,0]).transpose((0,1,3,2)).reshape(-1, n_y, order='F'),
                              recale_array(pred_max).transpose((0,1,3,2)).reshape(-1, n_y, order='F')),
                             axis=1)
    imgs.append(img)

    return imgs

def save_images(imgs, path):
    for i in range(len(imgs)-1):
        Image.fromarray(imgs[i]).save('%s_class_%d.png'%(path, i))
    Image.fromarray(imgs[-1]).save('%s_argmax.png'%path)

def combine_and_save_images(x, y, pred, path):
    imgs = combine_images(x, y, pred)
    save_images(imgs, path)


def eval_to_str(evaluation_dict):
    o_s = ''
    for key in evaluation_dict:
        value = evaluation_dict.get(key)
        if value.size >= 2:
            mean = np.mean(value) #[1:]
        else:
            mean = np.mean(value)
        o_s += '%s: %.4f  '%(key, mean)
    return o_s

def add_dict(dict_old, dict_new):
    if dict_old is None:
        dict_old = dict_new
    else:
        for key in dict_new:
            dict_old[key] += dict_new[key]
    return dict_old
    
def div_dict(_dict, i):
    for key in _dict:
            _dict[key] /= i
    return _dict

def multiply_dict(_dict, i):
    for key in _dict:
            _dict[key] *= i
    return _dict

def copytree(src, dst, symlinks=False, ignore=None):
    if os.path.exists(dst):
        shutil.rmtree(dst, ignore_errors=ignore)
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

# def augmentaion(img, lab, mask, times):

#     # rotate
#     if times < 1:
#         return [], [], []

#     if mask is None:
#         imgs = [[img, lab]]
#         p = Augmentor.DataPipeline(imgs)
#         # p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
#         p.flip_left_right(probability=0.5)
#         p.flip_top_bottom(probability=0.5)
#         aug_imgs = np.array(p.sample(times))
        
#         a_imgs = aug_imgs[0:, 0, ...]
#         a_labs = aug_imgs[0:, 1, ...]
#         return a_imgs, a_labs

#     imgs = [[img, lab, mask]]
#     p = Augmentor.DataPipeline(imgs)
#     p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
#     p.flip_left_right(probability=0.5)
#     p.flip_top_bottom(probability=0.5)
#     aug_imgs = np.array(p.sample(times))
    
#     a_imgs = aug_imgs[0:, 0, ...]
#     a_labs = aug_imgs[0:, 1, ...]
#     a_masks = aug_imgs[0:, 2, ...]
#     return a_imgs, a_labs, a_masks


def augmentaion(img, lab, mask, times):
    # rotate
    if times < 1:
        return [], [], []
    lab[lab>0]=1
    lab = SegmentationMapOnImage(lab, shape=img.shape, nb_classes=2)


    if mask is not None:
        mask[mask>0]=1
        mask = SegmentationMapOnImage(mask, shape=img.shape, nb_classes=2)
        

    a_imgs = []
    a_labs = []
    a_masks = []

    sometimes = lambda aug: iaa.Sometimes(0.8, aug)
    seq = iaa.Sequential([
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),
        # iaa.ElasticTransformation(alpha=(10.0,40.0), sigma=(4.0,5.0)),
        sometimes(iaa.ContrastNormalization((0.5, 2.0))),
        sometimes(iaa.PerspectiveTransform(scale=(0.02, 0.1)))
    ], random_order=True)

    for _ in range(times):
        _aug = seq._to_deterministic()
        aug_img = _aug.augment_image(img)
        aug_lab = _aug.augment_segmentation_maps(lab).get_arr_int()
        aug_lab[aug_lab>0] = 255
        a_imgs.append(aug_img)
        a_labs.append(aug_lab)

        if mask is not None:
            aug_mask = _aug.augment_segmentation_maps(mask).get_arr_int()
            aug_mask[aug_mask>0] = 255
            a_masks.append(aug_mask)

    # a_imgs = aug_imgs[0:, 0, ...]
    # a_labs = aug_imgs[0:, 1, ...]
    return a_imgs, a_labs, a_masks

def save_augmentation(imgs, labs, masks, path, img_name='_org.tif', lab_name='_manual1.gif', mask_name='_mask.gif', index_x=0):
    for i in range(len(imgs)):
        Image.fromarray(imgs[i]).save('%s/aug_%d_%d%s'%(path, index_x, i, img_name))
        Image.fromarray(labs[i]).save('%s/aug_%d_%d%s'%(path, index_x, i, lab_name))
        if masks is not None:
            Image.fromarray(masks[i]).save('%s/aug_%d_%d%s'%(path, index_x, i, mask_name))

def load_prob_map(file_list, org_suffix='_org.tif', prob_suffix='_prob.mat'):
    print('load prob map: %d'%len(file_list))
    prob_maps = []
    for f in file_list:
        prob_name = f.replace(org_suffix, prob_suffix)
        prob_map = np.array(sio.loadmat(prob_name)['prob'])
        prob_maps.append(prob_map)
    return np.array(prob_maps)

def load_labeled_map(file_list, org_suffix='_org.tif', prob_suffix='_manual1.gif'):
    print('load prob map: %d'%len(file_list))
    prob_maps = []
    for f in file_list:
        prob_name = f.replace(org_suffix, prob_suffix)
        prob_map = np.array(Image.open(prob_name))
        prob_maps.append(prob_map)
    return np.array(prob_maps)

def copyfiles(src_list, dst):
    for src in src_list:
        shutil.copy2(src, dst)


import pydensecrf.densecrf as dcrf

from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian

def DCRF(org, prob, iteration=5, w1=1, w2=2, alpha=0.5, beta=1, gamma=1):
    
    n_class = prob.shape[-1]

    org = np.array(org, np.uint8)
    lab = np.array(prob, np.float32) 

    lab = lab.reshape((-1, n_class))
    lab = np.transpose(lab, (1,0))

    d = dcrf.DenseCRF2D(prob.shape[1], prob.shape[0], prob.shape[2])
    d.setUnaryEnergy(lab.copy(order='C'))

    d.addPairwiseGaussian(sxy=gamma, compat=w2)
    d.addPairwiseBilateral(sxy=alpha, srgb=beta, rgbim=org, compat=w1)

    Q = d.inference(iteration)
    Q = np.transpose(Q, (1,0))
    Q = np.reshape(Q, prob.shape)

    return np.stack((Q[..., 1], Q[..., 0]), -1)
