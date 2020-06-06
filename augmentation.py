import math

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from skimage.io import imread

from utils import checker, check_range, convert_to_absolute, create_circular_mask, skew, swap_patches, create_grid_masks

class Augmentor(object):
    """
    Modify images according to some augmenting functions. The input is a dictionary with a set of options. The keys are
    the operations and the values the range or type of operations. Another option is to pass a dictionary as value where
    values is the key for the values. This allows to pass other data such as probability that specifies the probability of
    applying the operations, otherwise is set to 1.

    Some methods are rewritten from
    https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
    """

    def __init__(self, operations, synthetic_image_creator=None, input_synthesizer_size=None, seed=None):
        '''
        Use seed if you want to apply the same transformation to different group of images. For instance, when images and
        masks need to be processed.
        :param operations: This a dictionary with all the operations. The key are the operations and the values the parameters.
                        Possible keys and the expected value
                         - brightness: (min_value, max_value) The values must for brightness must be between 0.05 and 10
                         - color_balance: (min_value, max_value) color_balance must be between 0 and 10
                         - contrast: (min_value, max_value) contrast must be between 0 and 10
                         - flip: 'horizontal' or 'hor', 'vertical' or 'ver', both
                         - greyscale: []
                         - grid_mask: (min_x_pos, max_x_pos, min_y_pos, max_y_pos, min_width_square, max_width_square,
                                        min_height_square, max_heigth_square, min_x_distance_between_squares,
                                        max_x_distance_between_squares, min_y_distance_between_squares, max_y_distance_between_squares)
                                        Values must be between 0 to 1 sinc they are relative to the size of the image.
                                        Generally, the initial position should be similar to the distance between squares
                                        This type of augmentations can be used two o three times with different parameters, since it is
                                        good to have a lot of different grids without having too much of the image covered.
                         - illumination: (min_radius, max_radius, min_magnitude, max_magnitude)  -- standard (0.05, 0.1, 100, 200)
                         - noise: (min_sigma, max_sigma) -- gaussian noise wiht mean 0
                         - occlusion: (type, min_height, max_height, min_width, max_width)  - creates a box of noise to block the image.
                                    The types are hide_and_seek and cutout so far. As extra parameter accepts 'num_patches' which can be a number or a range
                                    By default is 1.
                         - posterisation: (min_number_levels, max_number_levels) Reduce the number of levels of the image. It is assumed a 255
                                        level (at least it is going to be returned in this way). However, this will perform a reduction to less levels than 255
                         - rgb swapping: True or False. This opertion swaps the RGB channels randomly
                         - rotation: (min angle, max angle) - in degrees
                         - sharpness: (min_value, max_value) - The values must be between -5 and 5, 1 means original image not 0.
                         - shear: (type, magnitude_min, magnitude_max) types are "random", 'hor', 'ver'. The magnitude are the angles to shear in degrees
                         - skew: (type, magnitude_min, magnitude_max), where types are: "TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER", "RANDOM", "ALL"
                         - solarise: [] doing a solarisation (max(image) - image)
                         - translate: (min_x, max_x, min_y, max_y) values are relative to the size of the image (0, 0.1, 0, 0.1)
                         - whitening: (min_alpha, max_alpha)  -- new image is  alpha*white_image + (1-alpha) * image
                         - zoom: (min value, max value) - the values are relative to the current size. So, 1 is the real size image (standard 0.9, 1.1)

                         Apart from this, the values could be a dictionary where of the form {'values': [values], 'probability': 1, special_parameter: VALUE}
                         The probability is the ratio of using this operation and special_parameters are indicated in the above descriptions when they have.

        :param synthetic_image_creator (function): A model that returns an image passing a random initialisation with input_synthesizer_size
        :param input_synthesizer_size (tuple): The size of the synthesiser input.
        :param seed: A seed to initiate numpy random seed in case of need. By default, None
        '''
        self.perform_checker = True
        self.seed = seed
        self._operations = operations
        self.synthetic_image_creator = synthetic_image_creator
        self.input_synthesizer_size = input_synthesizer_size

        self.skew_types = ["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER", "RANDOM", "ALL"]
        self.flip_types = ['VERTICAL', 'VER', 'HORIZONTAL', 'HOR', 'RANDOM', 'ALL']
        self.occlusion_types = ['hide_and_seek', 'cutout']
        self.illumination_types = ['blob_positive', 'blob_negative', 'blob', 'constant_positive', 'constant_negative',
                                   'constant', 'positive', 'negative', 'all', 'random']

        self.initial_prob = {'flip': 0.5, 'solarise': 0.5, 'greyscale': 0.5, 'rgb_swapping': 0.5}

        self.numpy_fun = ['grid_mask', 'illumination', 'noise', 'occlusion', 'posterisation', 'rgb_swapping', 'sample_pairing', 'translate']

    @property
    def operations(self):
        return self._operations

    @operations.setter
    def operations(self, operations):
        self._operations = operations
        self.perform_checker = True

    def rescale(self, im):
        """
        Rescale an image between 0 and 255
        :param im (array): An image or set of images
        """
        if np.max(im) == np.min(im):
            return (im * 0).astype(np.uint8)

        return (255 * (im.astype(np.float) - np.min(im)) / (np.max(im) - np.min(im))).astype(np.uint8)

    def run(self, images, **kwargs):
        """
        Perform the augmentation on the images
        :param images(numpy arrays): A
        :param kwargs: (TODO) Extra information to pass to the transformation
        :return: The augmented image/s
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        if len(images) == 0:
            return images

        new_operations = {'numpy': {}, 'pil': {}}
        for operation, values in self.operations.items():
            if operation in self.numpy_fun:
                new_operations['numpy'][operation] = values
            else:
                new_operations['pil'][operation] = values

        get_first_value = False
        if not isinstance(images, (set, tuple, list)):
            get_first_value = True
            images = [images]

        channels = []
        for image in images:
            channel = 0
            if len(image.shape) == 3:
                channel = image.shape[2]
            channels.append(channel)
            if not isinstance(image, np.ndarray):
                raise TypeError('Images must be of type ndarray')

        norm = lambda x: np.squeeze(np.uint8(x)) if np.max(x) > 1.0 else np.uint8(x * 255)
        pil_obj = [Image.fromarray(norm(image)) for image in images]

        # If some extra data gets modified, like the labels of the images, then it must be returned. The output would be
        # a dictionary with the images in the key images and the other parameters with the same name as in the input.
        output_extra = {}
        for type_data in ['pil', 'numpy']:
            operations = new_operations[type_data]
            if type_data == 'numpy':
                output = []
                for i, image in enumerate(pil_obj):
                    # The number of channels must be preserved, operations with PIL like greyscale or greyscale images
                    # would removed the dimension and leave a 2D image
                    channel = channels[i]
                    image = np.array(image)
                    if len(image.shape) == 2 and channel == 1:
                        image = image[..., None]
                    elif (len(image.shape) == 2 and channel == 3) or (
                            len(image.shape) == 3 and image.shape[2] == 1 and channel == 3):
                        image = np.dstack([image, image, image])
                    output.append(image)


            for operation, values in operations.items():
                if not isinstance(values, dict):
                    input_data = {'values': values}
                else:
                    input_data = values

                probability = input_data.get('probability')
                probability = probability if probability else input_data.get('prob',
                                                                             self.initial_prob.get(operation, 1.0))

                extra_data = {key: value for key, value in input_data.items() if
                              key not in ['values', 'probability', 'prob']}
                for key, val in kwargs.items():
                    extra_data[key] = val

                if np.random.rand(1)[0] < probability:
                    op = getattr(self, operation, None)
                    if op is not None:
                        if type_data == 'pil':
                            pil_obj = op(pil_obj, input_data.get('values', []), **extra_data)
                        else:
                            output = op(output, input_data.get('values', []), **extra_data)
                            # If the operation returns a dictionary, means that a parameters from extra_data has been
                            # modified, when this happens, extra_data must be updated for future calls and the final
                            # output should include the modified values.
                            if isinstance(output, dict):
                                copy_dict = output
                                output = copy_dict['images']
                                for key, value in copy_dict.items():
                                    if key != 'images':
                                        output_extra[key] = value
                                        extra_data[key] = value
                    else:
                        print('The operation {} does not exist, Aborting'.format(operation))

        # output = [np.array(image) for image in pil_obj]

        if get_first_value:
            output = output[0]
        if output_extra:
            output_extra['images'] = output
            output = output_extra

        return output

    def check_images_equal_size(self, images):
        """
        Check that all the images have the same size
        :param images: A list of images
        :return: True if all images have same size otherwise False
        """
        get_size = lambda x: x.size
        if isinstance(images[0], np.ndarray):
            get_size = lambda x: x.shape[:2]

        h, w = get_size(images[0])
        for image in images:
            h1, w1 = get_size(image)
            if h1 != h or w1 != w:
                return False

        return True

    def check_groups_images_equal_size(self, images, images2):
        """
        Check that all the images in images and images2 has the same size. It assumes that both images and images2
        have at least one image
        :param images: A list of images (numpy array or PIL)
        :param images2: A list of images (numpy or PIL)
        """
        output = True
        output = output and self.check_images_equal_size(images)
        output = output and self.check_images_equal_size(images2)

        return output and self.check_images_equal_size([np.array(images[0]), np.array(images2[0])])

    def brightness(self, images, values, **kwargs):
        """
        Change the birghtness of the image, factors smaller than 0 will make the image darker and a factor greater than 1
        will make it brighter. 0 means completely black and it should be avoided
        :param images(list or array): A list or array of images, each being 3D
        :param values: 2 values. The minimum and maximum change in brightness, the values must be between 0.05 and 10
                                beynd those points the result are too dark or too bright respectively.
        :return: A list with the images with some brightness change
        """

        factor = checker('brightness', 'range of brightness', values, 2, 0.05, 10)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                image_enhancers = ImageEnhance.Brightness(image)
                output.append(image_enhancers.enhance(factor))
            else:
                output.append(image)

        return output

    def color_balance(self, images, values, **kwargs):
        """
        Change the saturation of the image, factors smaller than 1 will make the image darker and a factor greater than 1
        will make it brighter. 0 means completely black and it should be avoided
        :param images: A list of numpy arrays, each being an image
        :param values: A list with 2 values. Minimum value for colour balance (greater than 0)
                                            Maximum value for colour balance (smaller than 10)
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                         mask_positions: The positions in images that are masks.
        :return: A list of images with changed colour
        """
        factor = checker('brightness', 'range of color_balance', values, 2, 0, 10)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                image_enhancers = ImageEnhance.Color(image)
                output.append(image_enhancers.enhance(factor))
            else:
                output.append(image)

        return output

    def contrast(self, images, values, **kwargs):
        """
        Change the contrast of the image, factors smaller than 1 will make the image to have a solid color and a factor greater than 1
        will make it brighter.
        :param images: A list of numpy arrays, each being an image
        :param values: A list with 2 values. Minimum value for contrast balance (greater than 0)
                                            Maximum value for contrast balance (smaller than 10)
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                         mask_positions: The positions in images that are masks.
        :return: A list of image with changed contrast
        """

        factor = checker('contrast', 'range of contrast', values, 2, 0, 10)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                image_enhancers = ImageEnhance.Contrast(image)
                output.append(image_enhancers.enhance(factor))
            else:
                output.append(image)

        return output

    def crop(self, images, values, **kwargs):
        """
        Perform a crop of the images. The main difference with zoom is that zoom creates a crop from the center of the
        image and respects the dimension of the images, in addition it may increase the size of the image.
        This operation selects a random position in the image and extract a patch with a random height and width. Both
        the height and the width can be restricted to a given range.
        :param images: A list of numpy arrays, each being an image
        :param values: 4 values: Minimum height of the crop (or both height and width if there are only two values)
                                Maximum height of the crop (or both height and width if there are only 2 values).
                                Minimum width of the crop (optional)
                                Maximum width of the crop (optional)

                                All the values must be between 0 and 1, meaning a relative crop with respect to
                                the size of the image.
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                         mask_positions: The positions in images that are masks.
        :return:
        """

        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        name = 'crop'
        if not self.check_images_equal_size(images):
            print('For {}, the size of the images must be the same. Aborting'.format(name))
            return images

        shape = images[0].size
        name_params = ['height', 'width']
        for i in range(len(values) // 2):
            check_range(name, name_params[i], values[i * 2:(i + 1) * 2], 0, 1)

        if len(values) != 2 and len(values) != 4:
            raise ValueError('The length of values in crop must be 2 or 4')

        if len(values) == 4:
            cropped_height = np.random.uniform(values[0], values[1])
            cropped_width = np.random.uniform(values[2], values[3])
        else:
            if values[1] > 1:
                raise ValueError('When only two elements are use, the values of the crop must be relative 0 to 1')

            cropped_height = np.random.uniform(values[0], values[1])
            cropped_width = cropped_height

        cropped_height = cropped_height if cropped_height >= 1 else int(cropped_height * shape[0])
        cropped_width = cropped_width if cropped_width >= 1 else int(cropped_width * shape[1])

        center_w = \
            np.random.randint(int(np.ceil(cropped_width / 2.0)), int(np.ceil(shape[1] - cropped_width / 2.0)), 1)[0]
        center_h = \
            np.random.randint(int(np.ceil(cropped_height / 2.0)), int(np.ceil(shape[0] - cropped_height / 2.0)), 1)[0]

        width = int(np.ceil(cropped_width / 2.0))
        height = int(np.ceil(cropped_height / 2.0))

        for i, image in enumerate(images):
            if no_mask_positions[i]:
                image = image.crop((center_h - height, center_w - width, center_h + height, center_w + width))
                image = image.resize((shape[0], shape[1]))
                # image = Image.fromarray(self.rescale(image))
            output.append(image)

        return output

    def flip(self, images, values, **kwargs):
        """
        Flip the image, vertically, horizontally or both
        :param images: A list of numpy arrays, each being an image
        :param values: 1 value, the type ('horizontal', 'vertical', 'all', 'random')
        :param kwargs: None
        :return: A list with the flipped images
        """
        if isinstance(values, (tuple, list)):
            values = values[0]

        if values.upper() not in self.flip_types:
            raise ValueError('The name {} does not exist for the flip operation. Possible values are: {}'.format(values,
                                                                                                                 self.flip_types))

        if values.lower() == 'random':
            values = np.random.choice(['horizontal', 'vertical', 'all'], 1)[0]

        if values.lower() == 'horizontal' or values.lower() == 'hor' or values.lower() == 'both' or values.lower() == 'all':
            images = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]

        if values.lower() == 'vertical' or values.lower() == 'ver' or values.lower() == 'both' or values.lower() == 'all':
            images = [image.transpose(Image.FLIP_TOP_BOTTOM) for image in images]

        return images

    def greyscale(self, images, values, **kwargs):
        """
        Convert to greyscale with probability one
        :param images: A list of numpy arrays, each being an image
        :param values: None
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                         mask_positions: The positions in images that are masks.
        :return: A list with the images converted into greyscale
        """
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                output.append(ImageOps.grayscale(image))
            else:
                output.append(image)

        return output

    def grid_mask(self, images, values, **kwargs):
        """
        Add a grid mask to the images following https://arxiv.org/pdf/2001.04086.pdf
         :param images: A list of numpy arrays, each being an image
         :param values: 8 values: Minimum and maximum value for the initial x position (top left corner of the top left square)
                                     Minimum and maximum value for the initial y position (top left corner of the top left square)
                                     Minimum and maximum value (range) for the width of the square
                                     Minimum and maximum value (range) for the height of the square
                                     Minimum and maximum value (range) for the x distance between square
                                     Minimum and maximum value (range) for the y distance between square

                                     All the values must be between 0 and 1 since they are relative to the image size.
         :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                         mask_positions: The positions in images that are masks.
                         use_colour: The colour to use. If the colour is not passed or it is a negative value or greater
                        than 255, gaussian noise will be used instead.
        :return: List of images with occlusions by a grid of masks
        """
        use_colour = kwargs.get('use_colour', -1)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        if not hasattr(values, '__len__') or len(values) != 12:
            raise ValueError(
                'The number of values for the grid_mask operation must be a list or tuple with 12 values. The range of the initial point, square size and distance between square in x and y for the three of them')

        if not self.check_images_equal_size(images):
            print('For grid masks, the size of the images must be the same. Aborting')
            return images

        h, w = images[0].shape[:2]
        params = []
        name = 'grid_mask'
        name_params = ['initial x position', 'initial y position', 'width square', 'height square',
                       'x distance between squares', 'y distance between squares']
        for i in range(len(values) // 2):
            param = checker(name, name_params[i], values[i * 2:(i + 1) * 2], 2, 0, 1)
            if i % 2 == 0:
                param = int(np.ceil(param * w))
            else:
                param = int(np.ceil(param * h))
            params.append(param)

        images_to_use = []
        for ii in range(len(images)):
            if no_mask_positions[ii]:
                if use_colour < 0 or use_colour > 255:
                    im = 30 * np.random.randn(*(images[ii].shape)) + 127.5
                    im[im < 0] = 0
                    im[im > 255] = 255
                else:
                    im = use_colour * np.ones(tuple(images[ii].shape))
            else:
                im = np.zeros(tuple(images[ii].shape))
            images_to_use.append(im)

        return create_grid_masks(images, params[:2], params[2:4], params[4:], images_to_use,
                                 no_mask_positions.tolist())

    def illumination(self, images, values, **kwargs):
        """
        Add illumination circles to the image following paper: https://arxiv.org/pdf/1910.08470.pdf
        :param images: A list of numpy arrays, each being an image
        :param values: 4 values: Minimum and maximum radius (float). The values must be between 0 and 1
                                Minimum and maximum intensity to add (int). This value cannot be larger than 255 and lower than 0.
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                        mask_positions: The positions in images that are masks.
        :return: A list with the images with some blob of changes in the illumination
        """
        name = 'illumination'  # inspect.currentframe().f_code.co_name
        if not self.check_images_equal_size(images):
            print('For {}, the size of the images must be the same. Aborting'.format(name))
            return images

        param_values = [('radius', 1), ('intensity', 255)]
        if not hasattr(values, '__len__') or len(values) != 5:
            raise ValueError(
                'The number of values for the illumination operation must be a list or tuple with 5 values'.format(
                    name))

        if values[0].lower() not in self.illumination_types:
            raise ValueError(
                'The name {} does not exist for the flip operation. Possible values are: {}'.format(values[0],
                                                                                                    self.illumination_types))

        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        shape = images[0].shape
        for i in range(len(values) // 2):
            check_range(name, param_values[i][0], values[i * 2 + 1:(i + 1) * 2 + 1], 0, param_values[i][1])

        aux = convert_to_absolute(values[1:3], shape)
        values[1] = aux[0]
        values[2] = aux[1]

        type_illumination = values[0].lower()
        if type_illumination == 'random':
            type_illumination = np.random.choice(self.illumination_types[:-1], 1)

        blob = np.zeros(shape[:2])
        if 'constant' not in type_illumination:
            radius = np.random.uniform(values[1], values[2])
            intensity = np.random.uniform(values[3], values[4])

            yc = np.random.randint(0, shape[0], 1)
            xc = np.random.randint(0, shape[1], 1)

            _, blob = create_circular_mask(shape[0], shape[1], (xc, yc), radius, get_distance_map=True)
            min_val = np.min(blob[blob > 0])
            blob = (blob - min_val) / (1 - min_val)
            blob[blob < 0] = 0

            if 'positive' in type_illumination:
                sign = 1
            elif 'negative' in type_illumination:
                sign = -1
            else:
                sign = int(np.random.rand(1) < 0.5)
            blob = sign * (intensity * blob)

        if 'blob' not in type_illumination:
            intensity = np.random.uniform(values[3], values[4])
            if 'positive' in type_illumination:
                sign = 1
            elif 'negative' in type_illumination:
                sign = -1
            else:
                sign = int(np.random.rand(1) < 0.5)
            blob += sign * intensity

        if len(shape) == 3:
            blob = blob[:, :, np.newaxis]

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                image = image.astype(np.float)
                image += blob
                image = self.rescale(image)
            output.append(image)

        return output

    def noise(self, images, values, **kwargs):
        """
        Add noise to the images
        :param images: A list of numpy arrays, each being an image
        :param values: 2 values:
                        int: Minimum number for the std of the noise. Values are between 0 and 255. Recommendation: not higher than 30
                        int: Maximum number for the std of the noise. Values are between 0 and 255. Recommendation: not higher than 30
                        Selection is done by a uniform distribution between minimum and maximum values.
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                        mask_positions: The positions in images that are masks.
        :return: A list with the images plus noise
        """
        std = checker('noise', 'standard deviation of the noise', values, 2, 0, 100)

        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                if len(image.shape) == 2:
                    image = image[:, :, np.newaxis]
                row, col, ch = image.shape
                gauss = std * np.clip(np.random.randn(row, col, ch), -3, 3)
                noisy = image + gauss
                output.append(self.rescale(noisy))  # Image.fromarray(self.rescale(noisy)))
            else:
                output.append(image)

        return output

    def occlusion(self, images, values, **kwargs):
        """
        Perform hide and seek and cutout occlusions on some images
        :param images: A list of numpy arrays, each being an image
        :param values: 5 values:
                        str: type of occlusion: at this moment - 'hide_and_seek', 'cutout'
                        int: Minimum number of boxes columns directions
                        int: Maximum number of boxes columns directions
                        int: Minimum number of boxes rows directions
                        int: Maximum number of boxes rows directions
                        Selection is done by a uniform distribution between minimum and maximum values.
        :param kwargs: For this operation, there are two extra parameters:
                        mask_positions: The positions in images that are masks.
                        use_colour: The colour to use. If the colour is not passed or it is a negative value or greater
                        than 255, gaussian noise will be used instead.
        :return: List of images with occlusion
        """
        use_colour = kwargs.get('use_color', -1)
        if not self.check_images_equal_size(images):
            print('For occlusions, the size of the images must be the same. Aborting')
            return images

        if not hasattr(values, '__len__') or len(values) != 5:
            raise ValueError('The number of values for the occlusion operation must be a list or tuple with 5 values')
        if values[0] not in self.occlusion_types:
            raise ValueError(
                'The name {} does not exist for the skew operation. Possible values are: {}'.format(values[0],
                                                                                                    self.skew_types))

        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        swapped_images = []
        for ii in range(len(images)):
            if no_mask_positions[ii]:
                if use_colour < 0 and use_colour > 255:
                    im = 30 * np.random.randn(*(images[ii].shape)) + 127.5
                    im[im < 0] = 0
                    im[im > 255] = 255
                else:
                    im = use_colour * np.ones(tuple(images[ii].shape))
            else:
                im = np.zeros(tuple(images[ii].shape))
            swapped_images.append(im)

        new_images = swap_patches(images, values, 'occlusion', swapped_images, **kwargs)

        return [self.rescale(image) for image in new_images]

    def posterisation(self, images, values, **kwargs):
        """
        Reduce the number of levels of the image. It is assumed a 255 level (at least it is going to be returned in this way).
        However, this will perform a reduction to less levels than 255
        :param images: A list of numpy arrays, each being an image
        :param values: Two values, representing the minimum value and maximum value of levels to apply the posterisation.
                        An uniform distribution will be used to select the value
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                        mask_positions: The positions in images that are masks.
        :return: A list with the posterised images.
        """

        if not hasattr(values, '__len__') or len(values) != 2:
            raise ValueError(
                'The number of values for the posterisation operation must be a list or tuple with 2 values')

        levels = checker('Posterisation', 'levels', values, 2, 1, 256)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        # the idea is to reduce the number of levels in the image, so if we need to get 128 levels, means that we need
        # to get the pixels to be between 0 and 128 and then multiply them by 2. So we need to first divide them between
        # 256 /128 = 2
        levels = 256 // levels

        outputs = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                image = (self.rescale(image) / levels).astype(np.uint8)
                image = self.rescale(image)

            outputs.append(image)

        return outputs

    def rgb_swapping(self, images, values, **kwargs):
        """
        Swap the rgb components in the images randomly
        :param images: A list of numpy arrays, each being an image
        :param values: Not used
        :param kwargs: mask_positions, in case one or more of images are masks, then this transformation is not applied
        :return: A list with the images after swapping
        """
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        outputs = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                c = np.random.choice(np.arange(3), 2, replace=False)
                image = np.copy(image)
                image[:, :, c[0]] = image[:, :, c[1]]
                image[:, :, c[1]] = image[:, :, c[0]]

            outputs.append(image)

        return outputs

    def rotate(self, images, values, **kwargs):
        """
        Rotate an image by selecting an angle from a range
        :param images: A list of PIL arrays, each being an image
        :param values: 2 values the minimum and maximum angle. Values between -360 and 360
        :param kwargs: Not used
        :return: The same as images
        """
        if not hasattr(values, '__len__') or len(values) != 2:
            raise ValueError('The number of values for the rotate operation must be a list or tuple with 2 values')
        if min(values) < -360 or max(values) > 360:
            raise ValueError("The range of the angles must be between {} and {}.".format(-360, 360))

        angle = checker('Rotate', 'range of rotation', values, 2, -360, 360)
        output = []
        for image in images:
            # Get size before we rotate
            x = image.size[0]
            y = image.size[1]

            # Rotate, while expanding the canvas size
            image = image.rotate(angle, expand=True, resample=Image.BICUBIC)

            # Return the image, re-sized to the size of the image passed originally
            output.append(image.resize((x, y), resample=Image.BICUBIC))

        return output

    def sample_pairing(self, images, values, **kwargs):
        """
        This augmentation performs a weighted average of the image being augmented an another one that must be included
        in the values. The paper recommends to use this augmentation in three steps for better convergence:
        1. SamplePairing is completely disabled for the first epochs.
        2. SamplePairing is enabled for a few epochs and then disabled again. This alternating process should be used
            for the majority of the epochs.
        3. At the end of training SamplePairing is completely disabled again. This is named fine-tuning in the paper.

        :param images: A list of numpy arrays, each being an image
        :param values: 4 values. - Minimum value for weight to mix the images (the original image is multiplied by weight
                                    where the new image is multiplied by 1 - weight). The value must be >= 0
                                 - Maximum value for the weight to mix images. The value must be <= 1
                                 - A list with the images to use for the mixing with the original ones. For each image one
                                    of them would be selected automatically.
                                 - A list with the labels of the images that are going to be used for mixing up. The
                                    labels must be one hot vectors.
        :param kwargs: It will check whether a mask exists, if it does the process will not continue
                        - mask_positions: The positions in images that are masks.
                        - labels: It requires the labels of the images as one hot encoding, otherwise the result will not be good
                        imaging that when mixing a an image of label 0 and 2, new_label = weight*0 + (1-weight)*2. This
                        means that if we have a weight of 0.5, the new label is 1 by mixing a 0 and 2 label images,
                        which does not make sense.
        :return: A list of numpy arrays with t

        """
        labels = np.array(kwargs.get('labels', None))
        if len(labels) == 0:
            raise ValueError('For the operation sample_pairing the labels of the images must be passed in the run function with the key labels')
        if not hasattr(labels[0], '__len__'):
            raise ValueError('In the operation sample_pairing the labels must be a 2D array or list of lists. Only one hot encoding vectors')
        if not hasattr(values, '__len__') or len(values) != 4:
            raise ValueError('The number of values for the sample_pairing operation must be a list or tuple with 4 values. The minimum and maximum weights, a list with images to mix and a their respective labels')

        if kwargs.get('mask_positions', None):
            print('The operation sample_pairing does not allow masks to be passed. Aborting')

        if len(values[2]) != len(values[3]):
            raise ValueError('In the operation sample_pairing, the number of images and labels used for mixing must be the same, since they correspondent')
        if len(values[3]) == 0:
            raise ValueError('In the operation sample_pairing, at least one image and one labels must be passed to mix with the original images')
        if not hasattr(values[3][0], '__len__'):
            raise ValueError('In the operation sample_pairing the labels for mixing must be a 2D array or list of lists. Only one hot encoding vectors')

        output = {'images': [], 'labels': []}
        image_labels = list(zip(values[2], values[3]))
        num_images = len(image_labels)

        name = 'sample_pairing'
        if not self.check_groups_images_equal_size(images, values[2]):
            print('For {}, the size of all the images, including the ones for mixing must be the same. Aborting'.format(
                name))
            return images

        weight = checker(name, 'weights for averaging', values[:2], 2, 0, 1)

        for image, label in zip(images, labels):
            pos = np.random.randint(0, num_images - 1, 1)[0]
            image_mixing, label_mixing = image_labels[pos]
            new_image = self.rescale(weight * image + (1 - weight) * image_mixing)
            output['images'].append(new_image)
            output['labels'].append(weight * label + (1 - weight) * np.array(label_mixing))

        return output

    def sharpness(self, images, values, **kwargs):
        """
        Blurred or sharp an image
        :param images: A list of numpy arrays, each being an image
        :param values: 2 values: minimum value for the sharpness. It cannot be smaller than -5
                                 maximum value for the sharpness. It cannot be greater than 5.

                                The standard sharpness value is between 0 and 2, whereas 0 means blurred images,
                                1 means original image and 2 sharp image. However, negative values can be used to get
                                very blurry images and values greater than 2. The restrictions are -5 to 5 since
                                beyond those boundaries the fourier coefficients fail. It is recommended to use values
                                from -1 to 3.
        :param kwargs: For this operation, the only extra parameter is the whether an image is a mask.
                         mask_positions: The positions in images that are masks.
        :return: A list of image with changed contrast
        """

        factor = checker('sharpness', 'range of sharpness', values, 2, -5, 5)
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                image_enhancers = ImageEnhance.Sharpness(image)
                output.append(image_enhancers.enhance(factor))
            else:
                output.append(image)

        return output

    def shear(self, images, values, **kwags):
        """
        Shear transformation of an image from https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
        :param images: A list of numpy arrays, each being an image
        :param values: 3 values: type (random, both, all, horizontal, vertical) - both and all produces the same result
                                minimum value for shear,
                                maximum value.
        :param kwags: Not used
        :return: A list with the images after a shear transformation
        """

        width, height = images[0].size
        if not self.check_images_equal_size(images):
            print('The shear operation can only be performed when the images have the same dimensions. Aborting')
            return images

        if not isinstance(values, (tuple, list)) or len(values) != 3:
            raise ValueError('The number of values for the shear operation must be a list or tuple with 3 values')
        if values[0] not in self.flip_types:
            raise ValueError(
                'The name {} does not exist for the shear operation. Possible values are: {}'.format(values[0],
                                                                                                     self.flip_types))
        if values[1] > values[2]:
            values = [values[2], values[1]]
        if values[1] < 0 or values[2] > 360:
            raise ValueError("The magnitude range of the shear operation must be greater than 0 and less than 360.")

        direction = values[0]
        if values[0].lower() == 'random' or values[0].lower() == 'both' or values[0].lower() == 'all':
            direction = np.random.choice(['hor', 'ver'], 1)[0]

        angle_to_shear = int(np.random.uniform((abs(values[1]) * -1) - 1, values[2] + 1))
        if angle_to_shear != -1: angle_to_shear += 1

        # We use the angle phi in radians later
        phi = math.tan(math.radians(angle_to_shear))

        outputs = []
        if direction.lower() == "hor" or direction.lower() == 'horizontal':
            # Here we need the unknown b, where a is
            # the height of the image and phi is the
            # angle we want to shear (our knowns):
            # b = tan(phi) * a
            shift_in_pixels = phi * height

            if shift_in_pixels > 0:
                shift_in_pixels = math.ceil(shift_in_pixels)
            else:
                shift_in_pixels = math.floor(shift_in_pixels)

            # For negative tilts, we reverse phi and set offset to 0
            # Also matrix offset differs from pixel shift for neg
            # but not for pos so we will copy this value in case
            # we need to change it
            matrix_offset = shift_in_pixels
            if angle_to_shear <= 0:
                shift_in_pixels = abs(shift_in_pixels)
                matrix_offset = 0
                phi = abs(phi) * -1

            # Note: PIL expects the inverse scale, so 1/scale_factor for example.
            transform_matrix = (1, phi, -matrix_offset,
                                0, 1, 0)

            for image in images:
                image = image.transform((int(round(width + shift_in_pixels)), height),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)

                image = image.crop((abs(shift_in_pixels), 0, width, height))

                outputs.append(image.resize((width, height), resample=Image.BICUBIC))

        elif direction.lower() == "ver" or direction.lower() == 'vertical':
            shift_in_pixels = phi * width

            matrix_offset = shift_in_pixels
            if angle_to_shear <= 0:
                shift_in_pixels = abs(shift_in_pixels)
                matrix_offset = 0
                phi = abs(phi) * -1

            transform_matrix = (1, 0, 0,
                                phi, 1, -matrix_offset)

            for image in images:
                image = image.transform((width, int(round(height + shift_in_pixels))),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)

                image = image.crop((0, abs(shift_in_pixels), width, height))

                outputs.append(image.resize((width, height), resample=Image.BICUBIC))

        return outputs

    def skew(self, images, values, **kwags):
        """
        Skew images
        :param images(list or array): A list or array of images, each being 3D. This method requires all the images
                                    to be of the same size
        :param values: First value the skew type: TILT, TILT_TOP_BOTTOM, TILT_LEFT_RIGHT, CORNER or RANDOM.
                        The other two are the minimum and maximum skew (0 to 1 values).

                     - ``TILT`` will randomly skew either left, right, up, or down.
                       Left or right means it skews on the x-axis while up and down
                       means that it skews on the y-axis.
                     - ``TILT_TOP_BOTTOM`` will randomly skew up or down, or in other
                       words skew along the y-axis.
                     - ``TILT_LEFT_RIGHT`` will randomly skew left or right, or in other
                       words skew along the x-axis.
                     - ``CORNER`` will randomly skew one **corner** of the image either
                       along the x-axis or y-axis. This means in one of 8 different
                       directions, randomly.
        :param kwags: Extra parameters. They are not used in this method but it is required for consistency
        :return: A list with the skew images
        """
        if not self.check_images_equal_size(images):
            print('The skew operation can only be performed when the images have the same dimensions. Aborting')
            return images

        if not isinstance(values, (tuple, list)) or len(values) != 3:
            raise ValueError('The number of values for the skew operation must be a list or tuple with 3 values')
        if values[0] not in self.skew_types:
            raise ValueError(
                'The name {} does not exist for the skew operation. Possible values are: {}'.format(values[0],
                                                                                                    self.skew_types))

        magnitude = checker('Skew', 'range of skewness', values[1:], 2, 0, 1)

        return skew(images, values[0], magnitude)

    def solarise(self, images, values, **kwargs):
        """
        Perform the solarisation of the image. This operation is computed by creating an inverse of the image, where
        high intensity pixels are changed to low and viceversa (255 - image normally).
        :param images: A list of numpy arrays, each being an image
        :param values: None
        :param kwargs: The parameter mask_positions can be used to avoid using this operation over masks.
        :return: A list wit the images after the solarisation
        """
        no_mask_positions = np.ones(len(images)).astype(bool)
        for pos in kwargs.get('mask_positions', []): no_mask_positions[pos] = False

        output = []
        for i, image in enumerate(images):
            if no_mask_positions[i]:
                output.append(ImageOps.invert(image))
            else:
                output.append(image)

        return output

    def translate(self, images, values, **kwargs):
        """
        Translate an image along x, y or both. The way to move is given as in flipping
        :param images: A list of numpy arrays, each being an image
        :param values: A set of 3 or 5 values.
                        1. Type of translation: 'VERTICAL', 'VER', 'HORIZONTAL', 'HOR', 'RANDOM', 'ALL'
                        2. Minimum translation at x position (or both if only 3 values).
                        3. Maximum translation at x position (or both if only 3 values).
                        4. Minimum translation at y position (optional).
                        5. Maximum translation at y position (optional).
                        Values 2 - 5 are relative to the size of the image, so values are between -1 and 1.
        :param kwargs: For this operation, there is only one extra parameters:
                        use_color: The colour to use. If the colour is not passed or it is a negative value or greater
                        than 255, gaussian noise will be used instead.
        :return:
        """
        use_colour = kwargs.get('use_colour', -1)

        if not self.check_images_equal_size(images):
            print('The skew operation can only be performed when the images have the same dimensions. Aborting')
            return images

        values = list(values)
        if not isinstance(values, (tuple, list)) or (len(values) != 3 and len(values) != 5):
            raise ValueError(
                'The number of values for the translation operation must be a list or tuple with 3 or 5 values')
        if values[0].upper() not in self.flip_types:
            raise ValueError(
                'The name {} does not exist for the translate operation. Possible values are: {}'.format(values,
                                                                                                         self.flip_types))
        for i, v in enumerate(values[1:]):
            if len(values) == 5:
                j = 1 - i // 2  # There are four values (two ranges) and every two we use the same values of the shape. The inversion is because PIL uses x,y instead of height, width
            else:
                j = 1 - i

            if isinstance(v, float) and (v > 1.0 or v < -1.0):
                raise ValueError('When float is used, the values must be between -1 and 1 inclusive.')
            if isinstance(v, float):
                values[i + 1] = int(images[0].shape[j] * v)
            elif isinstance(v, int):
                if v > images[0].shape[j] or v < -images[0].shape[j]:
                    raise ValueError(
                        'When integers are used, the values for translation must be within the size of the image.')
            else:
                raise TypeError('Only float and integers are allowed for translate.')

        if values[1] > values[2]:
            values = [values[2], values[1]]

        if len(values) == 3:
            tx = int(np.random.uniform(values[1], values[2]))
            ty = int(np.random.uniform(values[1], values[2]))
        else:
            tx = int(np.random.uniform(values[1], values[2]))
            ty = int(np.random.uniform(values[3], values[4]))

        if values[0].lower() == 'random':
            values = np.random.choice(['horizontal', 'vertical', 'all'], 1)[0]

        output = []
        for image in images:
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
            h, w, c = image.shape
            if use_colour < 0 or use_colour > 255:
                im = 30 * np.abs(np.random.randn(h, w, c)) + 127.5
                im[im < 0] = 0
                im[im > 255] = 255
            else:
                im = use_colour * np.ones((h, w, c))

            if values[0].lower() == 'horizontal' or values[0].lower() == 'hor':
                ty = 0

            if values[0].lower() == 'vertical' or values[0].lower() == 'ver':
                tx = 0

            im[max(ty, 0): min(h + ty, h), max(tx, 0): min(w + tx, w), ...] = image[max(-ty, 0): min(h - ty, h),
                                                                              max(-tx, 0): min(w - tx, w), ...]

            output.append(self.rescale(np.squeeze(im)))
        return output

    def zoom(self, images, values, **kwargs):
        """
        Zoom an image. This means to resize the image and then cropping it if the new size is larger or adding noise
        padding if it is smaller.
        :param images: A list of images
        :param values: Tuple with the range of values of the zoom factor
        :param kwargs: Not used
        :return: A list with the zoomed images
        """
        h, w = images[0].size
        c = len(images[0].getbands())
        if not self.check_images_equal_size(images):
            print('The zoom operation can only be performed when the images have the same dimensions. Aborting')
            return images

        factor = checker('zoom', 'the range of zoom', values, 2, 0.1, 10)
        h_new, w_new = int(factor * h), int(factor * w)

        output = []
        for image in images:
            image = image.resize((h_new, w_new))
            dif_h = int(np.round(np.abs(h_new - h) / 2))
            dif_w = int(np.round(np.abs(w_new - w) / 2))
            if factor < 1:
                if c == 1:
                    im = 20 * np.random.randn(w, h) + 127.5
                else:
                    im = 20 * np.random.randn(w, h, c) + 127.5
                im[dif_w: w_new + dif_w, dif_h:h_new + dif_h, ...] = np.array(image)
                image = Image.fromarray(self.rescale(im))
            if factor > 1:
                image = image.crop((dif_h, dif_w, h + dif_h, w + dif_w))
            output.append(image)

        return output
