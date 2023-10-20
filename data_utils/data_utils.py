import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import os

def load_background_data():
    id_arr, colors_arr = [], []
    with open('template_colors_bgr.json', 'r') as rf:
        for data in json.load(rf):
            id_arr.append(data['id'])
            colors_arr.append(np.array(data['colors']).flatten())
    return id_arr, colors_arr


def extract_similarity(data_arr):
    data_arr = np.array(data_arr)
    data_arr = data_arr / np.linalg.norm(data_arr, axis=1)[:, None]
    matrix = np.einsum('ik,jk->ij', data_arr, data_arr)
    return matrix


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

