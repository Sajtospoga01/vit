import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def plot_channel_distribution(path:str):
    #plot brightness distribution

    image = np.load(path)
    n = 10
    print(n)
    image = image[n]
    brightnesses = np.array(image)

    brightnesses = np.bincount(brightnesses.reshape(-1, brightnesses.shape[-1])[:,2])
    brightness_dist = np.zeros(256)
    for i in range(min(brightnesses.shape[0],256)):
        brightness_dist[i] = brightnesses[i]
    print(brightness_dist)
    print(brightness_dist.shape)

    plt.bar(np.arange(256), brightness_dist)
    plt.title('Brightness Distribution')
    plt.xlabel('Brightness')
    plt.ylabel('Count')
    plt.show()



if __name__ == "__main__":
    print("hello world")
    plot_channel_distribution("C:\\Users\\andra\\Downloads\\bhsi\\pb_tr\\image\\batch_0.npy")