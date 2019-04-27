"""
Image metas are used to store image information related to its
original state and its transformations (to adapt it to the network
input).

Licensed under The MIT License
Written by Jean Da Rolt
"""
import numpy as np


class ImageMetas():
    """Stores image metas."""
    def __init__(self, original_shape, window=None, scale=1,
                 padding=((0, 0), (0, 0), (0, 0)),
                 crop=(-1, -1, -1, -1), image_id=-1):
        self.original_shape = original_shape
        if window is None:
            self.window = (0, 0, original_shape[0], original_shape[1])
        else:
            self.window = window
        self.scale = (scale, scale) if isinstance(scale, int) else scale
        self.padding = padding
        self.crop = crop
        self.image_id = image_id

    def to_numpy(self):
        """Takes attributes of an image and puts them in one 1D array. Use
        parse_image_meta() to parse the values back.

        image_id: An int ID of the image. Useful for debugging.
        image_shape: [height, width, channels]
        window: (y1, x1, y2, x2) in pixels. The area of the image where the real
                image is (excluding the padding)
        active_class_ids: List of class_ids available in the dataset from which
            the image came. Useful if training on images from multiple datasets
            where not all classes are present in all datasets.
        """
        padding_flat = [element for tupl in self.padding for element in tupl]
        meta = np.array(
            [self.image_id]                 # size=1
            + list(self.original_shape)     # size=3
            + list(self.window)             # size=4 (y1, x1, y2, x2) in image coordinates
            + list(self.scale)              # size=2 (vertical, horizontal)
            + list(padding_flat)            # size=6
            + list(self.crop),              # size=4
            dtype=np.float32)
        return meta

    def __str__(self):
        return (f"image_id: {self.image_id}, "
                f"original_shape: {self.original_shape}, "
                f"window: {self.window}, "
                f"scale: {self.scale}, "
                f"padding: {self.padding}', "
                f"crop: {self.crop}")


def build_metas_from_numpy(meta):
    """Parses an image info Numpy array to its components.
    See to_numpy() for more details.
    """
    metas = ImageMetas(meta[1:4],                    # original_shape
                       meta[4:8],                    # window
                       meta[8:10],                   # scale
                       meta[10:16].reshape((3, 2)),  # padding
                       meta[16:],                     # crop
                       meta[0])                      # image_id
    return metas
