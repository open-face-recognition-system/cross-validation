import os
from random import sample

from numpy import mean
from sklearn.metrics import accuracy_score, recall_score, precision_score

import eigenfaces
import fisherfaces
import lbph
from training import training


def create_kfolds(k=6):
    paths = [os.path.join("dataset/tmp/", i) for i in os.listdir("dataset/tmp/")]
    dataset = []

    for student_id_folder in paths:
        for photo_type_folder in os.listdir(student_id_folder):
            photo_type_path = os.path.join(student_id_folder + "/" + photo_type_folder)
            for photo_path in os.listdir(photo_type_path):
                dataset.append(photo_type_path + "/" + photo_path)

    train_test_split = []
    size = len(dataset)
    num_of_elements = int(size / k)

    for i in range(k):
        new_sample = sample(dataset, num_of_elements)
        train_test_split.append(new_sample)
        for row in new_sample:
            dataset.remove(row)
    if len(dataset) != 0:
        for rows in range(len(dataset)):
            train_test_split[rows].append(dataset[rows])
        dataset.clear()
    return train_test_split


def training_data(groups, folder_name, test_size):
    main_folder = "classifiers/" + test_size + "/" + folder_name
    final_paths = []
    for group in groups:
        for paths_to_training in group:
            final_paths.append(paths_to_training)

    training(final_paths, main_folder)


def recognize_data(group, folder_name, test_size):
    main_folder = "classifiers/" + test_size + "/" + folder_name
    final_paths = []
    y_true = []
    for paths_to_recognize in group:
        final_paths.append(paths_to_recognize)
        student_id = int(os.path.split(paths_to_recognize)[-1].split("-")[0])
        y_true.append(student_id)

    y_pred_eigenfaces = eigenfaces.recognize(final_paths, main_folder)
    y_pred_fisherfaces = fisherfaces.recognize(final_paths, main_folder)
    y_pred_lbph = lbph.recognize(final_paths, main_folder)

    eigen_accuracy = accuracy_score(y_true, y_pred_eigenfaces)
    eigen_recall = recall_score(y_true, y_pred_eigenfaces, average='macro')
    eigen_precision = precision_score(y_true, y_pred_eigenfaces, average='macro')

    fisher_accuracy = accuracy_score(y_true, y_pred_fisherfaces)
    fisher_recall = recall_score(y_true, y_pred_fisherfaces, average='macro')
    fisher_precision = precision_score(y_true, y_pred_fisherfaces, average='macro')

    lbph_accuracy = accuracy_score(y_true, y_pred_lbph)
    lbph_recall = recall_score(y_true, y_pred_lbph, average='macro')
    lbph_precision = precision_score(y_true, y_pred_lbph, average='macro')

    return (eigen_accuracy, eigen_recall, eigen_precision, fisher_accuracy, fisher_recall, fisher_precision,
            lbph_accuracy, lbph_recall, lbph_precision)


if __name__ == '__main__':
    test_size = 30
    data = create_kfolds()
    eigen_accuracy_values = []
    eigen_recall_values = []
    eigen_precision_values = []

    fisher_accuracy_values = []
    fisher_recall_values = []
    fisher_precision_values = []

    lbph_accuracy_values = []
    lbph_recall_values = []
    lbph_precision_values = []

    for index, paths in enumerate(data):
        aux_data = data.copy()
        data_to_recognize = aux_data[index]
        aux_data.pop(index)
        training_data(aux_data, str(index), str(test_size))
        (eigen_accuracy,
         eigen_recall,
         eigen_precision,
         fisher_accuracy,
         fisher_recall,
         fisher_precision,
         lbph_accuracy,
         lbph_recall,
         lbph_precision) = recognize_data(data_to_recognize, str(index), str(test_size))
        eigen_accuracy_values.append(eigen_accuracy)
        eigen_recall_values.append(eigen_recall)
        eigen_precision_values.append(eigen_precision)

        fisher_accuracy_values.append(fisher_accuracy)
        fisher_recall_values.append(fisher_recall)
        fisher_precision_values.append(fisher_precision)

        lbph_accuracy_values.append(lbph_accuracy)
        lbph_recall_values.append(lbph_recall)
        lbph_precision_values.append(lbph_precision)

    print("Eigenfaces accuracy: ", mean(eigen_accuracy_values))
    print("Eigenfaces recall: ", mean(eigen_recall_values))
    print("Eigenfaces precision: ", mean(eigen_precision_values))

    print("Fisherfaces accuracy: ", mean(fisher_accuracy_values))
    print("Fisherfaces recall: ", mean(fisher_recall_values))
    print("Fisherfaces precision: ", mean(fisher_precision_values))

    print("LBPH accuracy: ", mean(lbph_accuracy_values))
    print("LBPH recall: ", mean(lbph_recall_values))
    print("LBPH precision: ", mean(lbph_precision_values))
