import numbers
import random
import warnings
import numpy as np
import PIL
import skimage.transform
import torchvision
import math
import torch
import cv2
from PIL import Image
from PIL import ImageFilter

def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.
    Args:
        img (PIL Image): Image to be converted to grayscale.
    Returns:
        PIL Image: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not isinstance(img,PIL.Image.Image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img

def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'nearest':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def normalize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        print(clip.shape)
        raise TypeError('tensor is not a torch clip.')

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip


def denoramlize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        raise TypeError('tensor is not a torch clip.')

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.mul_(std[:, None, None, None]).add_(mean[:, None, None, None])

    return clip



class Compose(object):
    """Composes several transforms
    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_clip, annotation_clip=None):
        if annotation_clip is None:
            for t in self.transforms:
                data_clip = t(data_clip)
            return data_clip
        else:
            for t in self.transforms:
                data_clip, annotation_clip = t(data_clip, annotation_clip)
            return data_clip, annotation_clip

class RandomHorizontalFlip(object):
    """Horizontally flip the list of given images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def random_horizontal_flip(self, clip, chance=0.5):
        if chance < self.p:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip

    def __call__(self, data_clip, annotation_clip=None):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if annotation_clip is not None:
            chance = random.random()
            return self.random_horizontal_flip(data_clip, chance), self.random_horizontal_flip(annotation_clip, chance)
        else:
            return self.random_horizontal_flip(data_clip)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the list of given images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p
    
    def random_vertical_flip(self, clip, chance):
        if chance < self.p:
            if isinstance(clip[0], np.ndarray):
                return [np.flipud(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_TOP_BOTTOM) for img in clip
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip

    def __call__(self, data_clip, annotation_clip=None):
        """
        Args:
            img (PIL.Image or numpy.ndarray): List of images to be flipped
            in format (h, w, c) in numpy.ndarray
        Returns:
            PIL.Image or numpy.ndarray: Randomly flipped clip
        """

        if annotation_clip is not None:
            chance = random.random()
            return self.random_vertical_flip(data_clip, chance), self.random_vertical_flip(annotation_clip, chance)
        else:
            return self.random_vertical_flip(data_clip)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ClipToTensor(object):
    """Convert a clip (list of PIL images) in the range
    [0, 255] to a torch.FloatTensor of shape (T x C x H x W) in the range [0.0, 1.0].
    if mean and std are given, then normalize the clip.
    """
    def clip_to_tensor(self, clip):
        if isinstance(clip[0], np.ndarray):
            # handle numpy array
            img = torch.from_numpy(np.stack(clip, 0))
            # backward compatibility
            return img.float().div(255)
        elif isinstance(clip[0], PIL.Image.Image):
            # handle PIL Image
            return torch.stack([torchvision.transforms.ToTensor()(img) for img in clip], 0)
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(clip[0])))
    
    def __init__(self, mean=None, std=None) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, data_clip, annotation_clip=None):
        """
        Args:
            clip (PIL Image): Clip to be converted to tensor.
        Returns:
            Tensor: Converted clip.
        """
        if annotation_clip is not None:
            data_clip, annotation_clip = self.clip_to_tensor(data_clip), self.clip_to_tensor(annotation_clip)
        else:
            data_clip = self.clip_to_tensor(data_clip)
        if self.mean is not None and self.std is not None:
            mean = torch.as_tensor(self.mean, device=data_clip.device)
            std = torch.as_tensor(self.std, device=data_clip.device)
            data_clip = (data_clip - mean[None, :, None, None]) / std[None, :, None, None]
        if annotation_clip is not None:
            return data_clip, annotation_clip
        else:
            return data_clip


    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability that image should be converted to grayscale.
    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b
    """
    def __init__(self, p=0.2, per_frame=False):
        super().__init__()
        self.p = p
        self.per_frame = per_frame
    def __call__(self,clip):
        """
        Args:
            list of imgs (PIL Image or Tensor): Image to be converted to grayscale.
        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        num_output_channels = 1 if clip[0].mode == 'L' else 3
        if self.per_frame:
            for i in range(len(clip)):
                if random.random() < self.p:
                    clip[i]=to_grayscale(clip[i],num_output_channels)
        else:
            if torch.rand(1)<self.p:
                for i in range(len(clip)):
                    clip[i]=to_grayscale(clip[i],num_output_channels)
        return clip

class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = resize_clip(
            clip, new_size, interpolation=self.interpolation)
        return resized


class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, data_clip, annotaion_clip=None):
        if annotaion_clip is not None:
            return resize_clip(data_clip, self.size, self.interpolation), resize_clip(annotaion_clip, self.size, 'nearest')
        else:
            return resize_clip(data_clip, self.size, self.interpolation)


class RandomCrop(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, data_clip, annotation_clip=None):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(data_clip[0], np.ndarray):
            im_h, im_w, im_c = data_clip[0].shape
        elif isinstance(data_clip[0], PIL.Image.Image):
            im_w, im_h = data_clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(data_clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)

        if annotation_clip is not None:
            cropped_data = crop_clip(data_clip, y1, x1, h, w)
            cropped_annotation = crop_clip(annotation_clip, y1, x1, h, w)
            return cropped_data, cropped_annotation
        else:
            cropped_data = crop_clip(data_clip, y1, x1, h, w)
            return cropped_data

class RandomResizedCrop(object):
    """Crop the given list of PIL Images to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'): ## 0.4-1, 3/4-4/3
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(clip, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (list of PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        if isinstance(clip[0], np.ndarray):
            height, width, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            width, height = clip[0].size
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, data_clip, annotaion_clip=None):
        """
        Args:
            clip: list of img (PIL Image): Image to be cropped and resized.
        Returns:
            list of PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(data_clip, self.scale, self.ratio)

        if annotaion_clip is None:
            imgs = crop_clip(data_clip,i,j,h,w)
            return resize_clip(imgs,self.size,self.interpolation)
        else:
            imgs=crop_clip(data_clip,i,j,h,w)
            annotations = crop_clip(annotaion_clip,i,j,h,w)
            return resize_clip(imgs,self.size,self.interpolation), resize_clip(annotations,self.size,"nearest")
        # return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
    
class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


class CenterCrop(object):
    """Extract center crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, data_clip, annotaion_clip=None):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(data_clip[0], np.ndarray):
            im_h, im_w, im_c = data_clip[0].shape
        elif isinstance(data_clip[0], PIL.Image.Image):
            im_w, im_h = data_clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(data_clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger than '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        if annotaion_clip is None:
            return crop_clip(data_clip,y1,x1,h,w)
        else:
            return crop_clip(data_clip,y1,x1,h,w), crop_clip(annotaion_clip,y1,x1,h,w)


class RandomGaussianBlur(object):
    """Apply gaussian blur on a list of images
    Args:
    p (float): probability of applying the transformation
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2., per_frame=False):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.per_frame = per_frame

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be blurred
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Blurred list of images
        """
        if self.per_frame:
            for i in range(len(clip)):
                if random.random() < self.p:
                    radius = random.uniform(self.radius_min, self.radius_max)
                    if isinstance(clip[0], np.ndarray):
                        clip[i] = skimage.filters.gaussian(clip[i])
                    elif isinstance(clip[0], PIL.Image.Image):
                        clip[i] = clip[i].filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))
                    else:
                        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                        'but got list of {0}'.format(type(clip[0])))
            return clip
        else:
            if random.random() < self.p:
                if isinstance(clip[0], np.ndarray):
                    blurred = [skimage.filters.gaussian(img) for img in clip]
                elif isinstance(clip[0], PIL.Image.Image):
                    blurred = [img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))) for img in clip]
                else:
                    raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                    'but got list of {0}'.format(type(clip[0])))
                return blurred
            else:
                return clip

class RandomApply(object):
    """Apply a list of transformations with a probability p
    Args:
    transforms (list of Transform objects): list of transformations to compose.
    p (float): probability of applying the transformations
    """

    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be transformed
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Transformed list of images
        """
        if random.random() < self.p:
            for t in self.transforms:
                clip = t(clip)
        return clip


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, per_frame=False):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.per_frame = per_frame

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        jittered_clip = []
        if self.per_frame:
            for img in clip:
                if isinstance(clip[0], np.ndarray):
                    raise TypeError(
                        'Color jitter not yet implemented for numpy arrays')
                elif isinstance(clip[0], PIL.Image.Image):
                    brightness, contrast, saturation, hue = self.get_params(
                        self.brightness, self.contrast, self.saturation, self.hue)
                    # Create img transform function sequence
                    img_transforms = []
                    if brightness is not None:
                        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
                    if saturation is not None:
                        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
                    if hue is not None:
                        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
                    if contrast is not None:
                        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
                    random.shuffle(img_transforms)
                    for func in img_transforms:
                        jittered_img = func(img)
                    jittered_clip.append(jittered_img)
                
                else:
                    raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                    'but got list of {0}'.format(type(clip[0])))

        else:
            if isinstance(clip[0], np.ndarray):
                raise TypeError(
                    'Color jitter not yet implemented for numpy arrays')
            elif isinstance(clip[0], PIL.Image.Image):
                brightness, contrast, saturation, hue = self.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

                # Create img transform function sequence
                img_transforms = []
                if brightness is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
                if saturation is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
                if hue is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
                if contrast is not None:
                    img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
                random.shuffle(img_transforms)

                # Apply to all images
                for img in clip:
                    for func in img_transforms:
                        jittered_img = func(img)
                    jittered_clip.append(jittered_img)

            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                'but got list of {0}'.format(type(clip[0])))
        return jittered_clip


class Normalize(object):
    """Normalize a clip with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data_clip, annotation_clip=None):
        """
        Args:
            clip (list): List of PIL.Image or numpy.ndarray to be normalized
        Returns:
            list: Normalized list of PIL.Image or numpy.ndarray
        """

        ## normalize a list of tensor images accoring to mean and std
        clip = data_clip
        if isinstance(clip[0], torch.Tensor):
            mean = torch.as_tensor(self.mean, device=clip.device)
            std = torch.as_tensor(self.std, device=clip.device)
            clip = (clip - mean[None, :, None, None]) / std[None, :, None, None]
        elif isinstance(clip[0], np.ndarray):
            clip = [torchvision.transforms.functional.to_tensor(img) for img in clip]
            clip = torch.stack(clip, dim=0)
            clip = torchvision.transforms.functional.normalize(clip, self.mean, self.std)
            clip = [img.numpy() for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            clip = [torchvision.transforms.functional.to_tensor(img) for img in clip]
            clip = torch.stack(clip, dim=0)
            clip = torchvision.transforms.functional.normalize(clip, self.mean, self.std)
            clip = [torchvision.transforms.functional.to_pil_image(img) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if annotation_clip is None:
            return clip
        else:
            return clip, annotation_clip

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



class TimeTTransform:

    def __init__(self,
                 size_crops,
                 nmb_crops,
                 min_scale_crops,
                 max_scale_crops,
                 jitter_strength=0.2,
                 min_intersection=0.01,
                 blur_strength=1):
        """
        Main transform used for fine-tuning with Leopart. Implements multi-crop and calculates the corresponding
        crop bounding boxes for each crop-pair.
        :param size_crops: size of global and local crop
        :param nmb_crops: number of global and local crop
        :param min_scale_crops: the lower bound for the random area of the global and local crops before resizing
        :param max_scale_crops: the upper bound for the random area of the global and local crops before resizing
        :param jitter_strength: the strength of jittering for brightness, contrast, saturation and hue
        :param min_intersection: minimum percentage of intersection of image ares for two sampled crops from the
        same picture should have. This makes sure that we can always calculate a loss for each pair of
        global and local crops.
        :param blur_strength: the maximum standard deviation of the Gaussian kernel
        """
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        assert 0 < min_intersection < 1
        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops
        self.min_intersection = min_intersection

        # Construct color transforms
        self.color_jitter = ColorJitter(
            0.8 * jitter_strength, 0.8 * jitter_strength, 0.8 * jitter_strength,
            0.2 * jitter_strength
        )
        color_transform = [RandomApply([self.color_jitter], p=0.8),
                           RandomGrayscale(p=0.2)]
        blur = RandomGaussianBlur(radius_min=blur_strength * .1, radius_max=blur_strength * 2.)
        color_transform.append(RandomApply([blur], p=0.5))
        self.color_transform = Compose(color_transform)

        # Construct final transforms
        self.final_transform = ClipToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 

        # Construct randomly resized crops transforms
        self.rrc_transforms = []
        for i in range(len(self.size_crops)):
            random_resized_crop = RandomResizedCrop(
                self.size_crops[i],
                scale=(self.min_scale_crops[i], self.max_scale_crops[i]),
            )
            self.rrc_transforms.extend([random_resized_crop] * self.nmb_crops[i])

    def __call__(self, sample: torch.Tensor):
        multi_crops = []
        crop_bboxes = torch.zeros(len(self.rrc_transforms), 4)
        for i, rrc_transform in enumerate(self.rrc_transforms):
            # Get random crop params
            y1, x1, h, w = rrc_transform.get_params(sample, rrc_transform.scale, rrc_transform.ratio)
            if i > 0:
                y1 = (y1 // 16) * 16
                x1 = (x1 // 16) * 16
                h = (h // 16) * 16
                w = (w // 16) * 16
                # Check whether crop has min overlap with existing global crops. If not resample.
                while True:
                    # Calculate intersection between sampled crop and all sampled global crops
                    bbox = torch.Tensor([x1, y1, x1 + w, y1 + h])
                    left_top = torch.max(bbox.unsqueeze(0)[:, None, :2],
                                         crop_bboxes[:min(i, self.nmb_crops[0]), :2])
                    right_bottom = torch.min(bbox.unsqueeze(0)[:, None, 2:],
                                             crop_bboxes[:min(i, self.nmb_crops[0]), 2:])
                    wh = _upcast(right_bottom - left_top).clamp(min=0)
                    inter = wh[:, :, 0] * wh[:, :, 1]

                    # set min intersection to at least 1% of image area
                    min_intersection = int((sample[0].size[0] * sample[0].size[1]) * self.min_intersection)
                    # Global crops should have twice the min_intersection with each other
                    if i in list(range(self.nmb_crops[0])):
                        min_intersection *= 2
                    if not torch.all(inter > min_intersection):
                        y1, x1, h, w = rrc_transform.get_params(sample, rrc_transform.scale, rrc_transform.ratio)
                        y1 = (y1 // 16) * 16
                        x1 = (x1 // 16) * 16
                        h = (h // 16) * 16
                        w = (w // 16) * 16
                    else:
                        break

            # Apply rrc params and store absolute crop bounding box
            imgs = crop_clip(sample,y1,x1,h,w)
            img = resize_clip(imgs, rrc_transform.size, rrc_transform.interpolation)
            crop_bboxes[i] = torch.Tensor([x1, y1, x1 + w, y1 + h])

            # Apply color transforms
            if i == 0:
                sample = img
                crop_bboxes[i] = torch.Tensor([0, 0, rrc_transform.size[0], rrc_transform.size[1]])
            img = self.color_transform(img)
            # img = img

            # Apply final transform
            img = self.final_transform(img)
            multi_crops.append(img)

        # Calculate relative bboxes for each crop pair from aboslute bboxes
        gc_bboxes, otc_bboxes = self.calculate_bboxes(crop_bboxes)

        return multi_crops, {"gc": gc_bboxes, "all": otc_bboxes, "bbox": crop_bboxes}

    def calculate_bboxes(self, crop_bboxes):
        # 1. Calculate two intersection bboxes for each global crop - other crop pair
        gc_bboxes = crop_bboxes[:self.nmb_crops[0]]
        left_top = torch.max(gc_bboxes[:, None, :2], crop_bboxes[:, :2])  # [nmb_crops[0], sum(nmb_crops), 2]
        right_bottom = torch.min(gc_bboxes[:, None, 2:], crop_bboxes[:, 2:])  # [nmb_crops[0], sum(nmb_crops), 2]
        # Testing for non-intersecting crops. This should always be true, just as safe-guard.
        assert torch.all((right_bottom - left_top) > 0)

        # 2. Scale intersection bbox with crop size
        # Extract height and width of all crop bounding boxes. Each row contains h and w of a crop.
        ws_hs = torch.stack((crop_bboxes[:, 2] - crop_bboxes[:, 0], crop_bboxes[:, 3] - crop_bboxes[:, 1])).T[:, None]

        # Stack global crop sizes for each bbox dimension
        crops_sizes = torch.repeat_interleave(torch.Tensor([self.size_crops[0]]), self.nmb_crops[0] * 2)\
            .reshape(self.nmb_crops[0], 2)
        if len(self.size_crops) == 2:
            lc_crops_sizes = torch.repeat_interleave(torch.Tensor([self.size_crops[1]]), self.nmb_crops[1] * 2)\
                .reshape(self.nmb_crops[1], 2)
            crops_sizes = torch.cat((crops_sizes, lc_crops_sizes))[:, None]  # [sum(nmb_crops), 1, 2]

        # Calculate x1s and y1s of each crop bbox
        x1s_y1s = crop_bboxes[:, None, :2]

        # Scale top left and right bottom points by percentage of width and height covered
        left_top_scaled_gc = crops_sizes[:2] * ((left_top - x1s_y1s[:2]) / ws_hs[:2])
        right_bottom_scaled_gc = crops_sizes[:2] * ((right_bottom - x1s_y1s[:2]) / ws_hs[:2])
        left_top_otc_points_per_gc = torch.stack([left_top[i] for i in range(self.nmb_crops[0])], dim=1)
        right_bottom_otc_points_per_gc = torch.stack([right_bottom[i] for i in range(self.nmb_crops[0])], dim=1)
        left_top_scaled_otc = crops_sizes * ((left_top_otc_points_per_gc - x1s_y1s) / ws_hs)
        right_bottom_scaled_otc = crops_sizes * ((right_bottom_otc_points_per_gc - x1s_y1s) / ws_hs)

        # 3. Construct bboxes in x1, y1, x2, y2 format from left top and right bottom points
        gc_bboxes = torch.cat((left_top_scaled_gc, right_bottom_scaled_gc), dim=2)
        otc_bboxes = torch.cat((left_top_scaled_otc, right_bottom_scaled_otc), dim=2)
        return gc_bboxes, otc_bboxes


def _upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()