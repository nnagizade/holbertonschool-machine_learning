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
