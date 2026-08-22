#!/usr/bin/env python3
"""Defines the Yolo class for object detection using Yolo v3."""
import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


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
            containing all of the anchor boxes
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes based on their objectness score and class probability.

        boxes: list of numpy.ndarray of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for
            each output, respectively
        box_confidences: list of numpy.ndarray of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the processed box
            confidences for each output, respectively
        box_class_probs: list of numpy.ndarray of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the processed
            box class probabilities for each output, respectively

        Returns: tuple of (filtered_boxes, box_classes, box_scores)
        """
        box_scores_list = []
        box_classes_list = []
        filtered_boxes_list = []

        for box, confidence, class_prob in zip(
                boxes, box_confidences, box_class_probs):
            scores = confidence * class_prob
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            box_scores_list.append(class_scores.reshape(-1))
            box_classes_list.append(classes.reshape(-1))
            filtered_boxes_list.append(box.reshape(-1, 4))

        box_scores = np.concatenate(box_scores_list)
        box_classes = np.concatenate(box_classes_list)
        boxes_all = np.concatenate(filtered_boxes_list)

        mask = box_scores >= self.class_t

        filtered_boxes = boxes_all[mask]
        box_classes = box_classes[mask]
        box_scores = box_scores[mask]

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply Non-max suppression to filtered boxes.

        filtered_boxes: numpy.ndarray of shape (?, 4) containing all of
            the filtered bounding boxes
        box_classes: numpy.ndarray of shape (?,) containing the class
            number for the class that filtered_boxes predicts,
            respectively
        box_scores: numpy.ndarray of shape (?) containing the box scores
            for each box in filtered_boxes, respectively

        Returns: tuple of (box_predictions, predicted_box_classes,
            predicted_box_scores)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            order = np.argsort(-cls_scores)
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            keep = []
            indices = list(range(len(cls_boxes)))

            while len(indices) > 0:
                current = indices[0]
                keep.append(current)
                rest = indices[1:]

                new_indices = []
                for idx in rest:
                    iou = self.iou(cls_boxes[current], cls_boxes[idx])
                    if iou <= self.nms_t:
                        new_indices.append(idx)
                indices = new_indices

            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(
                np.full(len(keep), cls, dtype=box_classes.dtype))
            predicted_box_scores.append(cls_scores[keep])

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def iou(box1, box2):
        """Computes the Intersection over Union of two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        intersection = inter_w * inter_h

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union

    @staticmethod
    def load_images(folder_path):
        """
        Load images from a folder.

        folder_path: string representing the path to the folder holding
            all the images to load

        Returns: tuple of (images, image_paths)
            images: a list of images as numpy.ndarray
            image_paths: a list of paths to the individual images in
                images
        """
        image_paths = glob.glob(folder_path + '/*')
        images = [cv2.imread(path) for path in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images for the Darknet model.

        images: a list of images as numpy.ndarray

        Resizes the images with inter-cubic interpolation and rescales
        all images to have pixel values in the range [0, 1]

        Returns: tuple of (pimages, image_shapes)
            pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                containing all of the preprocessed images
            image_shapes: a numpy.ndarray of shape (ni, 2) containing the
                original height and width of the images
        """
        input_h = self.model.input.shape[2]
        input_w = self.model.input.shape[1]

        pimages_list = []
        image_shapes_list = []

        for img in images:
            image_shapes_list.append([img.shape[0], img.shape[1]])

            resized = cv2.resize(
                img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            rescaled = resized / 255

            pimages_list.append(rescaled)

        pimages = np.array(pimages_list)
        image_shapes = np.array(image_shapes_list)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Display the image with all boundary boxes, class names, and
        box scores.

        image: a numpy.ndarray containing an unprocessed image
        boxes: a numpy.ndarray containing the boundary boxes for the
            image
        box_classes: a numpy.ndarray containing the class indices for
            each box
        box_scores: a numpy.ndarray containing the box scores for each
            box
        file_name: the file path where the original image is stored

        If the `s` key is pressed, the image is saved in the directory
        `detections`, located in the current directory. If `detections`
        does not exist, it is created. Otherwise, if any other key is
        pressed, the image window is closed without saving.
        """
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            class_name = self.class_names[box_classes[i]]
            score = round(box_scores[i], 2)
            text = "{} {}".format(class_name, score)

            cv2.putText(
                image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            cv2.imwrite(os.path.join('detections', file_name), image)

        cv2.destroyAllWindows()
