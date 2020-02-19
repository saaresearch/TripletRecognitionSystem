
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def knn_acc(test_embeddings, test_labels, train_embeddings,
            train_labels, n_neighbors, metric):
    knn = KNeighborsClassifier(n_neighbors, metric=metric)
    knn.fit(train_embeddings, train_labels)
    y_pred = knn.predict(test_embeddings)
    msg = f'Knn acc: {accuracy_score(test_labels,y_pred)} count neighbors: {n_neighbors}'
    print(msg)
