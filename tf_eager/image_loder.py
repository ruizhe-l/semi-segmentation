import os
import glob
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import scipy.ndimage as nd


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def local_norm(img, sigma1, sigma2):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
    num = img - blur

    blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
    den = cv2.pow(blur, 0.5)
    den = np.clip(den, 1e-5, None)
    gray = num / den

    cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    return gray


class ImageLoder:
    def __init__(self, n_class, data_suffix, label_suffix=None, mask_suffix=None, search_path=None, filename_list=None, shuffle_data=True, resize=None, match_ref=None, prob_threshold=None, noise=False, generate_mask=None):
        self.data_suffix = data_suffix
        self.label_suffix = label_suffix
        self.mask_suffix = mask_suffix
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        self.resize = resize
        self.match_ref = match_ref
        self.prob_threshold = prob_threshold
        self.noise = noise
        self.generate_mask = generate_mask

        assert (search_path is None) ^ (filename_list is None), 'Multiple data list input!'
        if search_path is not None:
            self.filename_list = self._search_files(search_path)
        else:
            self.filename_list = filename_list
        assert len(self.filename_list) > 0, 'No data!'
        print('%d file loaded'%len(self.filename_list))
        self.file_idx = len(self.filename_list)

        if noise:
            self.angles = np.random.randint(360, size=len(self.filename_list))

    def size(self):
        return len(self.filename_list)

    def ndim(self):
        return 4

    def __call__(self, n_data):

        imgs, labs, masks = self._load_and_process_data()
        
        for i in range(1, n_data):
            img, lab, mask = self._load_and_process_data()
            imgs = np.concatenate((imgs, img), axis=0)
            if lab is not None:
                labs = np.concatenate((labs, lab), axis=0)
            if mask is not None:
                masks = np.concatenate((masks, mask), axis=0)

        return imgs, labs, masks

    def _load_and_process_data(self):
        img, lab, mask = self._next_data()

        img, lab, mask = self._pre_process(img, lab, mask)

        # assert img.shape[0:-1] == lab.shape[0:-1]
        img = img.reshape([1] + list(img.shape))# + [1])
        if lab is not None:
            lab = lab.reshape([1] + list(lab.shape))
        if mask is not None:
            mask = mask.reshape([1] + list(mask.shape))

        
        return img, lab, mask

    def _pre_process(self, img, lab, mask):

        if self.noise:
            angle = self.angles[self.file_idx]
            img = nd.rotate(img, angle, reshape=False)
            if lab is not None:
                lab = nd.rotate(lab, angle, reshape=False)
            if mask is not None:
                mask = nd.rotate(mask, angle, reshape=False) 
        
        if self.resize is not None:
            img = cv2.resize(img, self.resize)   
            if lab is not None:
                lab = cv2.resize(lab, self.resize)
            if mask is not None:
                mask = cv2.resize(mask, self.resize)

        img = self._process_img(img)
        lab = self._process_lab(lab)
        mask = self._process_mask(mask)


        return img, lab, mask

    def _process_img(self, img):

        

        if self.match_ref is not None:
            ref = self._load_file(self.match_ref)
            img = hist_match(img, ref)
            img = np.array(img, np.float32)

        # standardization (zero mean)
        # img -= np.mean(img)
        # img /= np.std(img)
        img = (img - np.median(img)) / np.std(img)
        # img = local_norm(img, 5, 30)

        # # 
        img -= np.min(img)
        img /= np.max(img)

        return img

    def _process_lab(self, lab):
        if lab is None:
            return lab

        nx = lab.shape[0]
        ny = lab.shape[1]
        labs = np.zeros((nx, ny, self.n_class), dtype=np.float32)

        if self.prob_threshold is not None:
            if lab.shape[-1] != self.n_class:
                labs[..., 0][lab<=0.1] = 1
                labs[..., 1][lab>0.1] = 1
            else:
                for i in range(self.n_class): 
                    # labs[..., i][lab[..., i]>self.prob_threshold] = 1
                    labs[..., 0][lab[..., 1]<=self.prob_threshold] = 1
                    labs[..., 1][lab[..., 1]>self.prob_threshold] = 1
        else: 
            if self.n_class == 2:
                labs[..., 0][lab<=0.1] = 1
                labs[..., 1][lab>0.1] = 1
            else:
                for i in range(self.n_class):
                    labs[..., i][lab==i] = 1
        return labs

    def _process_mask(self, mask):
        if mask is not None:
            mask[mask>0]=1
        return mask

    def _search_files(self, search_path):
        files = glob.glob(search_path)
        if self.label_suffix is None:
            return [name for name in files if self.data_suffix in name]
        return [name for name in files if self.data_suffix in name and not self.label_suffix in name]

    def _load_file(self, path, dtype=np.float32):
        if os.path.isfile(path):
            if path[-4:] == '.mat':
                return np.array(sio.loadmat(path)['prob'], dtype)
            else:
                return np.array(Image.open(path), dtype)
        else:
            if path[-8:] == 'prob.mat':
                t_path = path.replace('prob.mat', 'manual1.gif')
                if os.path.isfile(t_path):
                    path = t_path
                else:
                    t_path = path.replace('prob.mat', 'lab.tif')
                    if os.path.isfile(t_path):
                        path = t_path
                    else:
                        path = path.replace('prob.mat', '_segmentation.png')
                return np.array(Image.open(path), dtype) 
            elif path[-11:] == 'manual1.mat':
                path = path.replace('manual1.mat', 'manual1.gif')
                return np.array(Image.open(path), dtype) 
            elif path[-10:] == 'prob_s.mat':
                path = path.replace('.org.prob_s.mat', '.ah.ppm')
                return np.array(Image.open(path), dtype) 
            else:
                path = path.replace('.gif', '.mat')
                return np.array(sio.loadmat(path)['prob'], dtype)
        
        raise Exception()

    def _cycle_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.filename_list):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.filename_list)

    def _next_data(self):
        self._cycle_file()
        image_name = self.filename_list[self.file_idx]
        img = self._load_file(image_name)
        # if img.shape[-1] == 3:
        #     img = np.dot(img[...,:3], [0.299, 0.587, 0.114])

        if self.label_suffix is None:
            lab = None
        else:
            label_name = image_name.replace(self.data_suffix, self.label_suffix)
            lab = self._load_file(label_name)

        if self.mask_suffix is None:
            mask = None
        else:
            mask_name = image_name.replace(self.data_suffix, self.mask_suffix)
            mask = self._load_file(mask_name)

        if self.generate_mask is not None:
            mask = np.zeros(img.shape)
            mask[img > self.generate_mask] = 1
        return img, lab, mask

