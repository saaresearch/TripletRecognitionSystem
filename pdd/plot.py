import matplotlib.pyplot as plt
import numpy as np


def plot(embeddings, labels, class_names, name, save_img_path):
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(class_names)):
        inds = np.where(labels == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1])
    plt.legend(class_names, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(name)
    plt.show()
    fig.canvas.draw()
    fig.savefig(save_img_path)
