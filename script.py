import pickle
import torch

from torchvision import transforms
from PIL import Image

from pdd.model import PDDModel
from pdd.data_utils import (
    load_config,
    write_json_file
)


def load_image(infilename):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = Image.open(infilename)
    img = transform(img)
    img = img.reshape(1, 3, 256, 256)
    return img


def load_class_names(filename):
    with open(filename) as f:
        return f.read().splitlines()


def show_predict(pred):
    print("Network prediction:")
    item_prediction = pred['prediction']
    print(
        f"\tClass label: {item_prediction['label']}\n"
        f"\tClass name: {item_prediction['class_name']}\n"
    )

    item_knn = pred['knn_parameters']
    print("KNN parameters:")
    print(
        f"\tCount neighbors: {item_knn['n_neighbors']}\n"
        f"\tMetric: {item_knn['metrics']}\n"
    )

    print(f"TOP {len(pred['topn'])} predictions:")
    for predict in pred['topn']:
        print(f"\tLabel: {predict['label']} "
              f"Image index: {predict['index']}",
              f"Distance: {predict['distance']:.4f}"
              )


def create_response(label, class_names, knn_model, indices, distances):
    response = {
        "prediction": {
            "label": int(label),
            "class_name": str(class_names[label])
        },
        "knn_parameters": {
            "n_neighbors": knn_model.n_neighbors,
            "metrics": knn_model.metric.__name__
        },
        "topn": []
    }

    for idx, dist in zip(indices, distances):
        response["topn"].append({
            "label": int(knn_model._y[idx]),
            "index": int(idx),
            "distance": float(dist)
        })

    return response


def get_predict(img_name, feature_extractor, classifier, class_names, topn):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PDDModel(1280, 15, True)
    model.load_state_dict(
        torch.load(
            feature_extractor,
            map_location=device))
    inputs = load_image(img_name)
    model.eval()
    embedding = model(inputs).detach().cpu().numpy()
    knn = pickle.load(open(classifier, 'rb'))

    classes_name = load_class_names(class_names)
    y_pred = knn.predict(embedding)
    distances, indices = knn.kneighbors(embedding[[0]], n_neighbors=topn)
    return create_response(y_pred[0], classes_name,
                           knn, indices[0], distances[0])


def main():
    config = load_config('config/script_parameters.yaml')
    pred = get_predict(
        config['img_path'],
        config['feature_extractor'],
        config['classifier'],
        config['class_names'],
        config['topn']
    )
    show_predict(pred)
    if config['prediction_savefile']:
        write_json_file(config['prediction_savefile'], pred, 4)


if __name__ == '__main__':
    main()
