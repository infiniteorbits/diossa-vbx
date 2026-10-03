"""Keras (TensorFlow 2) MobilePose module.

Model implemented using Functional API.

Notes
-----
The implementation loosely follows the original paper [1]_,
to match the off-the-shelf model [2]_.
- The backbone is the same, but unmodified.
- The deconvolutions are replaced with dense upsampling convolutions.
- The heatmaps to keypoints computation is performed with
a differentiable spatial to numerical transform.

References
----------
.. [1] "MobilePose: Real-Time Pose Estimation for Unseen Objects with Weak Shape Supervision", https://arxiv.org/abs/2003.03522
.. [2] "GitHub: MobilePose", https://github.com/YuliangXiu/MobilePose

"""

from typing import Tuple, List
from abc import ABC

import tensorflow as tf
from tensorflow import keras

from ..mobilenet import mobilenet_v2
from ..duc import duc
from ..nn import dsnt

from .keypoints_regression import KeypointsRegression


def mobilepose(tensor, keypoints: int):
    """MobilePose model constructor.

    Submodels:
    1. MobileNetV2
    2. DUC

    Parameters
    ----------
    tensor : tensor
        Input tensor.
    keypoints : int
        Number of keypoints to be produced.

    Returns
    -------
    tensor
        Output of the model.
    """
    x = mobilenet_v2(tensor, output_layers=True)[-1]
    x = keras.layers.Conv2D(256, kernel_size=1, use_bias=False,
                            name="conv_compress")(x)
    x = duc(x, 512, name="duc1")
    x = duc(x, 256, name="duc2")
    x = duc(x, 128, name="duc3")
    x = keras.layers.Conv2D(keypoints, kernel_size=1, use_bias=False,
                                name="conv_heatmap")(x)
    return x


class MobilePose_ABC(ABC):
    """MobilePose model abstract base class."""

    def post(self, unnormalized_heatmaps: tf.Tensor) -> List[tf.Tensor]:
        """Normalizes the heatmaps and calculates the coordinates.

        Parameters
        ----------
        unnormalized_heatmaps : tensor
            Unnormalized heatmaps.

        Returns
        -------
        coords : tensor
            (Keypoint) coordinates, with batch as first dimension.
            Range dependes on `normalized_coordinates` boolean.
        heatmaps : tensor
            (Keypoint) heatmaps, with batch as first dimension.
        """
        x = tf.transpose(unnormalized_heatmaps, (0, 3, 1, 2))
        heatmaps = dsnt.flat_softmax(x)
        coords = dsnt.dsnt(heatmaps, normalized_coordinates=self.normalized_coordinates)
        return [coords, heatmaps]


class MobilePose_Edge(keras.Model, MobilePose_ABC):
    """(Partial) MobilePose model class, non quantizable (post-processing) operations.

    Source code can be embedded as post-processing of the edge model.

    Parameters
    ----------
    normalized_coordinates : bool, optional
        Selects normalized coordinates in range ]-1; 1[ or coordinates on the heatmap size.
    """

    def __init__(self, normalized_coordinates: bool = False) -> None:
        self.normalized_coordinates = normalized_coordinates

        super().__init__(name="MobilePose")

    def call(self, inputs, training=None, mask=None):
        return self.post(inputs)


class MobilePose(keras.Model, MobilePose_ABC, KeypointsRegression):
    """MobilePose model class.

    Submodels:
    1. MobileNetV2
    2. DUC
    3. (DSNT)

    Parameters
    ----------
    img_size : tuple of int, optional
        Image size in (width, height) format.
    channels : int, optional
        Number of channels on the image.
    keypoints : int, optional
        Number of keypoints to be produced.
    normalized_coordinates : bool, optional
        Selects normalized coordinates in range ]-1; 1[ or coordinates on the heatmap size.
        For training must be `normalized_coordinates = True`
        due to limitation in the `compute_loss` method.
    ignore_invisible_keypoints : bool, optional
        If True ignores the training loss of the invisible keypoints.
    """

    def __init__(self, img_size: Tuple[int, int] = (224, 224),
                 channels: int = 3, keypoints: int = 8,
                 normalized_coordinates: bool = True,
                 ignore_invisible_keypoints: bool = True) -> None:
        self.normalized_coordinates = normalized_coordinates
        self.ignore_invisible_keypoints = ignore_invisible_keypoints

        inp = keras.Input(shape=(*reversed(img_size), channels), name="img")
        out = mobilepose(inp, keypoints)
        super().__init__(inputs=inp, outputs=out, name="MobilePose")

    def compute_loss(self,
                     coords: tf.Tensor,
                     heatmaps: tf.Tensor,
                     ground_truth_keypoints: tf.Tensor,
                     ground_truth_visibility: tf.Tensor
                    ) -> tf.Tensor:
        """Computes the inference loss related to the ground truth.

        Parameters
        ----------
        coords : tf.Tensor
            Predicted keypoints coords (N,num_keypoints,2) shape.
            Only supports normalized coordinates in range ]-1; 1[.
        heatmaps: tf.Tensor
            Predicted keypoints heatmaps (N,num_keypoints,56,56) shape.
        ground_truth_keypoints: tf.Tensor
            Ground truth keypoints coords (N,num_keypoints,2) shape
        ground_truth_visibility: tf.Tensor
            Ground truth visibility flags (N,num_keypoints) shape.
            Ignored if model was created with ignore_invisible_keypoints = True

        Returns
        -------
        losses : Dict[str, tf.Tensor]
            Dictionary with all loss components. Coordinates loss and heatmaps loss.
        """
        coordinates_loss = dsnt.euclidean_losses(coords, ground_truth_keypoints)
        heatmaps_loss = dsnt.js_reg_losses(heatmaps, ground_truth_keypoints, sigma_t=1.0)

        if self.ignore_invisible_keypoints:
            coordinates_loss = coordinates_loss[ground_truth_visibility == 1]
            heatmaps_loss = heatmaps_loss[ground_truth_visibility == 1]

        return {'coords': coordinates_loss, 'heatmaps': heatmaps_loss}


class MobilePose_Full(keras.Model, MobilePose_ABC):
    """MobilePose complete model class.

    Submodels:
    1. MobileNetV2
    2. DUC
    3. DSNT

    Parameters
    ----------
    img_size : tuple of int, optional
        Image size in (width, height) format.
    channels : int, optional
        Number of channels on the image.
    keypoints : int, optional
        Number of keypoints to be produced.
    normalized_coordinates : bool, optional
        Selects normalized coordinates in range ]-1; 1[ or coordinates on the heatmap size.
        For training must be `normalized_coordinates = True`
        due to limitation in the `compute_loss` method.
    ignore_invisible_keypoints : bool, optional
        If True ignores the training loss of the invisible keypoints.
    """

    def __init__(self, img_size: Tuple[int, int] = (224, 224),
                 channels: int = 3, keypoints: int = 8,
                 normalized_coordinates: bool = True,
                 ignore_invisible_keypoints: bool = True) -> None:
        self.normalized_coordinates = normalized_coordinates
        self.ignore_invisible_keypoints = ignore_invisible_keypoints

        inp = keras.Input(shape=(*reversed(img_size), channels), name="img")
        out = mobilepose(inp, keypoints)
        super().__init__(inputs=inp, outputs=out, name="MobilePose")

    def call(self, inputs, training=None, mask=None):
        return self.post(super().call(inputs, training, mask))

