import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

def bgr_to_hsv(bgr_2dim):
    normalized_bgr_2dim = np.array(bgr_2dim) / 255.0
    b, g, r = normalized_bgr_2dim[:,0], normalized_bgr_2dim[:,1], normalized_bgr_2dim[:,2]
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    diff = max_val - min_val

    # Initialize h, s, and v with zeros
    h = np.zeros_like(r)
    s = np.zeros_like(r)
    v = max_val * 100  # Value

    # Compute Hue
    cond_r = max_val == r
    cond_g = max_val == g
    cond_b = max_val == b
    cond_mx_mn = max_val == min_val

    h = np.where(cond_r, 60 * ((g - b) / diff % 6), h)
    h = np.where(cond_g, 60 * ((b - r) / diff + 2), h)
    h = np.where(cond_b, 60 * ((r - g) / diff + 4), h)
    h = np.where(cond_mx_mn, 0, h)

    # Compute Saturation
    s = np.where(max_val != 0, (diff / max_val) * 100, 0)

    # Stack them into a 2D array
    hsv_2dim = np.stack([h, s, v], axis=1)
    return hsv_2dim

def hsv_to_rgb(hsv_2dim):
    hsv_2dim = np.array(hsv_2dim).astype(np.float32)
    normalized_hsv_2dim = hsv_2dim / np.array([360.0,100.0,100.0])
    return mcolors.hsv_to_rgb(normalized_hsv_2dim) * 255.0
    
def hsv_to_bgr(hsv_2dim):
    rgb_2dim = hsv_to_rgb(hsv_2dim)
    bgr_2dim = np.array([list(reversed(rgb_1dim)) for rgb_1dim in rgb_2dim])
    return bgr_2dim


def visualize_rgb_colors(rgb_colors):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Loop through the list of BGR colors and plot each one
    for i, color in enumerate(rgb_colors):
        # Convert BGR to RGB and normalize to [0, 1]
        rgb_color = [x / 255.0 for x in color]

        # Create a rectangle filled with the normalized RGB color
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=rgb_color)
        
        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, len(rgb_colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()

def visualize_bgr_colors(bgr_colors):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Loop through the list of BGR colors and plot each one
    for i, color in enumerate(bgr_colors):
        # Convert BGR to RGB and normalize to [0, 1]
        rgb_color = [x / 255.0 for x in reversed(color)]

        # Create a rectangle filled with the normalized RGB color
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=rgb_color)
        
        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, len(bgr_colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()


def visualize_hsv_colors(hsv_colors):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    rgb_colors = hsv_to_rgb(hsv_colors)
    # Loop through the list of HSV colors and plot each one
    for i, rgb_color in enumerate(rgb_colors):
        # Create a rectangle filled with the RGB color
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=rgb_color/255)

        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, len(hsv_colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()


from sklearn.decomposition import PCA

def plot_3D_PCA(data, labels=None, label_names=None):
    
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        indices = np.where(labels == label)
        ax.scatter(reduced_data[indices, 0], reduced_data[indices, 1], reduced_data[indices, 2], 
                   label=label_names[label] if label_names else f'Label {label}')

    ax.set_title('3D PCA')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # Adding the legend
    ax.legend(loc='upper right')

    plt.show()



# clustering kmeans
def img_none_flatten(img, mask):
    if mask is not None:
        img = np.where(mask > 127, img, np.nan)
    return img[~np.isnan(img[:, :, :3]).any(axis=2)]

class Point:
    def __init__(self, data, K=4):
        self.data = data
        self.k = np.random.randint(0, K)

    def __repr__(self):
        return str({"data": self.data, "k": self.k})

def make_k_mapping(points):
    point_dict = defaultdict(list)
    for p in points:
        point_dict[p.k] = point_dict[p.k] + [p.data]
    return point_dict


def calc_k_means(point_dict, K=4):
    means = [np.mean(point_dict[k], axis=0) for k in range(K)]
    return means


def update_k(points, means, K=4):
    for p in points:
        dists = [np.linalg.norm(means[k] - p.data) for k in range(K)]
        p.k = np.argmin(dists)


def find_color_type_fixed(img, mask=None, n_clusters=4, epochs=1):
    
    img_flatten = img_none_flatten(img.copy(), mask.copy())
    points = [Point(d, K=n_clusters) for d in img_flatten]
    point_dict = make_k_mapping(points)
    colors = calc_k_means(point_dict, K=n_clusters)
    update_k(points, colors, K=n_clusters)
    for e in range(epochs):
        point_dict = make_k_mapping(points)
        colors = calc_k_means(point_dict, K=n_clusters)
        update_k(points, colors, K=n_clusters)

    try:
        colors = [[int(c) for c in color] if isinstance(color, np.ndarray) or isinstance(color, list) else [0, 0, 0] for color in colors]
    except TypeError:
        print(colors)
        raise TypeError

    percentage = [0 for _ in range(n_clusters)]
    for p in points:
        percentage[p.k] += 1
    percentage = [p / len(img_flatten) for p in percentage]

    return colors, percentage


def preprocess_input_image(input_image):
    return input_image

def sort_colors(colors):
    colors.sort(key=lambda c: c[2])
    colors.sort(key=lambda c: c[1])
    colors.sort(key=lambda c: c[0])
    return colors

def sort_colors_hsv(colors):
    if not isinstance(colors, list):
        colors = colors.tolist()
        
    colors.sort(key=lambda c: c[2]**2 + c[1]**2, reverse=True)
    return colors

def sort_color_feature_mean_dist(colors):
    colors = np.array(colors)
    mean_color = np.mean(colors, axis=0)
    dist = np.sum((colors - np.array([mean_color for _ in range(len(colors))])) ** 2, axis=1)/len(colors)
    
    return colors[np.argsort(dist)[::-1]].tolist()


if __name__ == "__main__":

    bgr_2dim = np.array([[28,42,42], [0,0,255], [10,20,200]]).astype(np.float32)
    hsv_2dim = bgr_to_hsv(bgr_2dim)
    rgb_2dim = hsv_to_rgb(hsv_2dim)
    bgr_2dim_2 = hsv_to_bgr(hsv_2dim)

    print("BGR Colors")
    print(bgr_2dim)
    visualize_bgr_colors(bgr_2dim)

    print("HSV Colors")
    print(hsv_2dim)
    visualize_hsv_colors(hsv_2dim)

    print("RGB Colors")
    print(rgb_2dim)
    visualize_rgb_colors(rgb_2dim)

    print("BGR Colors2")
    print(bgr_2dim_2)
    visualize_bgr_colors(bgr_2dim)