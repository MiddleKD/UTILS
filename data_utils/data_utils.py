import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import os


def load_templates_features():
    id_arr, colors_arr, weights_arr = [], [], []
    
    total_json = []

    fns = glob("./features/*.json")
    for fn in fns:
        with open(fn, 'r') as rf:
            total_json.extend(json.load(rf))
    
    for data in total_json:
        id_arr.append(data['id'])
        colors_arr.append(np.array(data['colors']).flatten())
        weights_arr.append(np.array(data['weights']).flatten())
    
    return np.array(id_arr), np.array(colors_arr), np.array(weights_arr)


def load_templates_dict():
    fns = glob("/media/mlfavorfit/sdb/Favorfit_templates_confirmed/**/*.jpg")
    fns_dict = {fn[-26:]:fn for fn in fns}

    with open("./analize/background_colors_json/meta_datas/all_templates_meta_data.json", 'r') as rf:
        datas_dict = json.load(rf)

    sorted_list = sorted(list(datas_dict.items()), key= lambda x:int(x[0].replace("_","").replace(".jpg","")))

    templates_dict = {}
    for idx, (key, value) in enumerate(sorted_list):
        templates_dict[idx+1] = fns_dict[key]
        # result["colors"] = value["features"][0]  # rgb
        # result["weights"] = value["features"][1]

    return templates_dict


def calculate_cos_similarity(target, data_arr):
    target = np.array(target) / np.linalg.norm(target)
    data_arr = np.array(data_arr) / np.linalg.norm(data_arr, axis=1)[:,None]
    cos_sim = np.matmul(target, data_arr.T)[0]

    cos_sim[np.isnan(cos_sim)] = 0
    return cos_sim


def get_close_index(similarity, id_arr, max_num=None):
    top_n_index = similarity.argsort()[::-1]
    if max_num is not None and max_num > 0:
        top_n_index = top_n_index[:max_num]
    return [int(id) for id in np.array(id_arr)[top_n_index]]


def extract_euclidien_similarity(data_arr):
    data_arr = np.array(data_arr)
    norm_data = np.sum(data_arr ** 2, axis=1).reshape(-1, 1)
    squared_distances = norm_data + norm_data.T - 2 * np.dot(data_arr, data_arr.T)
    squared_distances = np.maximum(squared_distances, 0)
    distances = np.sqrt(squared_distances)
    similarities = 1 / (1 + distances)
    np.fill_diagonal(similarities, 1)
    
    return similarities


def get_template_dict():
    template_paths = glob("/media/mlfavorfit/sdb/template/*/*.jpg")
    template_dict = {int(os.path.basename(path).split(".")[0]):path for path in template_paths}
    return template_dict


def visualize_templates(template_ids, template_dict):
    for x, template_id in enumerate(template_ids):
        template_path = template_dict[template_id]
        template_image = Image.open(template_path)

        plt.subplot(1, len(template_ids), x + 1)
        plt.axis('off')
        plt.imshow(template_image)

    plt.show()


def concat_array(arr1, arr2, axis=0):
    return np.concatenate([arr1, arr2], axis=axis)


if __name__ == "__main__":
    colors = np.array([
            [
                204,
                189,
                203
            ],
            [
                170,
                149,
                176
            ],
            [
                142,
                123,
                148
            ],
            [
                86,
                64,
                95
            ]
        ])
    weights = np.array([
            0.37982177734375,
            0.3221435546875,
            0.20721435546875,
            0.0908203125
        ])
    
    id_arr, colors_arr, weights_arr = load_templates_features()
    
    similarity = calculate_cos_similarity([concat_array(colors.flatten(), weights*30, axis=0)], concat_array(colors_arr, weights_arr*30, axis=1))
    print(get_close_index(similarity, id_arr, max_num=30))

    similarity = calculate_cos_similarity([colors.flatten()], colors_arr)
    print(get_close_index(similarity, id_arr, max_num=30))