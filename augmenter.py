import pandas
import numpy as np
import augment_matrices
import time
import cv2


def augment(images, labels):
    unique_labels = np.unique(labels)

    label_counts = dict()

    for label in unique_labels:
        label_counts[label] = np.count_nonzero(labels == label)

    most_frequent_label = max(label_counts, key=label_counts.get)
    most_frequent_label_count = label_counts[most_frequent_label]

    for label, amount in label_counts.items():
        if label == most_frequent_label:
            continue

        image_target_count = min(most_frequent_label_count, amount * 8)

        print("augmenting label: {}", label)
        while amount < image_target_count:
            for image, label2 in zip(images, labels):
                if label2 != label:
                    continue

                M = np.eye(3)
                M = np.matmul(M, augment_matrices.get_translate_matrix(0.5))
                M = np.matmul(M, augment_matrices.get_zoom_matrix())
                M = np.matmul(M, augment_matrices.get_stretch_matrix())
                M = np.matmul(M, augment_matrices.get_shear_matrix())
                M = np.matmul(M, augment_matrices.get_rotate_matrix(False))
                M = np.matmul(M, augment_matrices.get_flip_matrix())

                imageMatrix = np.eye(3)

                height, width = image.shape[:2]

                imageMatrix = np.matmul(np.float32(
                    [[1, 0, width / 2], [0, 1, height / 2], [0, 0, 1]]), M)  # type: ignore
                imageMatrix = np.matmul(imageMatrix, np.float32(
                    [[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]]))  # type: ignore

                image = cv2.warpAffine(
                    image, imageMatrix[:2], (width, height), borderMode=cv2.BORDER_REPLICATE)

                images = np.append(images, [image], axis=0)
                labels = np.append(labels, label)

                amount += 1

                if amount >= image_target_count:
                    break
