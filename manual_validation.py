from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

y_true = [15, 15, 15, 15, 15, 22, 22, 22, 22, 22, 19, 19, 19, 19, 19,
          11, 11, 11, 11, 11, 8, 8, 8, 8, 8]

y_pred_test_1 = [15, 15, 15, 15, 15, 15, 15, 15, 15, 22, 15, 19, 19, 19, 19,
                 22, 22, 22, 22, 22, 22, 22, 19, 11, 22]

y_pred_test_2 = [11, 15, 15, -1, 15, 15, 15, 15, 15, 15, 22, 22, 15, 15, 11,
                 11, 22, 19, 11, 19, 19, 22, 8, 15, 15]

y_pred_test_3 = [-1, -1, 11, -1, -1, -1, -1, -1, 11, 11, 19, 11, 19, 19, 22,
                 22, 22, 22, 22, 11, 11, 11, 11, 11, 11]

y_pred_test_4 = [-1, -1, -1, 22, 22, 22, 15, -1, 22, 22, 22, 19, 22, 19, 19,
                 22, 11, 22, 11, 22, 22, 19, 19, 19, 19]

y_pred_test_5_with_update = [15, 15, 15, 15, 15, 15, 22, 22, 11, 22, 22, 22, 19, 19, 19,
                             22, 11, 11, 11, 11, 22, 22, 22, 22, 22]

y_pred_test_5 = [15, 15, 15, 15, 15, 22, 22, 15, 11, 11, 19, 19, 19, 19, 19, 11, 11, 11, 11, 11,
                 22, 22, 22, 22, 22]

y_pred_test_6_with_update = [-1, 22, 11, 15, 11, 22, 11, 22, 22, 22, 22, 22, 11, 22, 19,
                             22, 22, 22, 22, 22, 22, 22, 11, 22, 22]

y_pred_test_6 = [15, -1, 15, 15, 15, 22, 22, 22, 22, 11, 19, 19, 19, 22, 19,
                 22, 11, 22, 22, 22, 22, 22, 22, 11, 11]

if __name__ == '__main__':
    accuracy = accuracy_score(y_true, y_pred_test_1)
    precision = precision_score(y_true, y_pred_test_1, average="weighted")
    recall = recall_score(y_true, y_pred_test_1, average="weighted")

    print("Reconhecedor estático")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    accuracy = accuracy_score(y_true, y_pred_test_2)
    precision = precision_score(y_true, y_pred_test_2, average='weighted')
    recall = recall_score(y_true, y_pred_test_2, average='weighted')

    print("Reconhecedor estático apontando para uma foto no computador")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    accuracy = accuracy_score(y_true, y_pred_test_3)
    precision = precision_score(y_true, y_pred_test_3, average='weighted')
    recall = recall_score(y_true, y_pred_test_3, average='weighted')

    print("Reconhecedor dinâmico parado, sem qualquer movimento")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    accuracy = accuracy_score(y_true, y_pred_test_4)
    precision = precision_score(y_true, y_pred_test_4, average='weighted')
    recall = recall_score(y_true, y_pred_test_4, average='weighted')

    print("Reconhecedor dinâmico com o celular em mãos")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    accuracy = accuracy_score(y_true, y_pred_test_5)
    precision = precision_score(y_true, y_pred_test_5, average='weighted')
    recall = recall_score(y_true, y_pred_test_5, average='weighted')

    print("Reconhecedor estático com update das fotos")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    accuracy = accuracy_score(y_true, y_pred_test_5_with_update)
    precision = precision_score(y_true, y_pred_test_5_with_update, average='weighted')
    recall = recall_score(y_true, y_pred_test_5_with_update, average='weighted')

    print("Reconhecedor dinâmico sem update das fotos")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    accuracy = accuracy_score(y_true, y_pred_test_6)
    precision = precision_score(y_true, y_pred_test_6, average='weighted')
    recall = recall_score(y_true, y_pred_test_6, average='weighted')

    print("Reconhecedor dinâmico com update das fotos")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    accuracy = accuracy_score(y_true, y_pred_test_6_with_update)
    precision = precision_score(y_true, y_pred_test_6_with_update, average='weighted')
    recall = recall_score(y_true, y_pred_test_6_with_update, average='weighted')

    print("Reconhecedor dinâmico sem update das fotos")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
