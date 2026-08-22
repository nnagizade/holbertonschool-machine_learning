#!/usr/bin/env python3
"""Defines the Yolo class for object detection using Yolo v3."""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor.

        model_path: path to where a Darknet Keras model is stored
        classes_path: path to where the list of class names used for the
            Darknet model, listed in order of index, can be found
        class_t: float representing the box score threshold for the
            initial filtering step
        nms_t: float representing the IOU threshold for non-max
            suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
                outputs: number of outputs (predictions) made by the
                    Darknet model
                anchor_boxes: number of anchor boxes used for each
                    prediction
                2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Applies the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs for a single image.

        outputs: list of numpy.ndarray containing the predictions from
            the Darknet model for a single image
        image_size: numpy.ndarray containing the image's original size
            [image_height, image_width]

        Returns: tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

            # build grid of cell coordinates
            cx = np.arange(grid_width).reshape(1, grid_width, 1)
            cx = np.tile(cx, (grid_height, 1, anchor_boxes))
            cy = np.arange(grid_height).reshape(grid_height, 1, 1)
            cy = np.tile(cy, (1, grid_width, anchor_boxes))

            bx = (self.sigmoid(t_xy[..., 0]) + cx) / grid_width
            by = (self.sigmoid(t_xy[..., 1]) + cy) / grid_height

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            input_h = self.model.input.shape[2]
            input_w = self.model.input.shape[1]

            bw = (anchor_w * np.exp(t_wh[..., 0])) / input_w
            bh = (anchor_h * np.exp(t_wh[..., 1])) / input_h

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.zeros(output[..., 0:4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

        return boxes, box_confidences, box_class_probs
