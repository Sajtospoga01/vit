import rasterio
from rasterio.plot import show
from matplotlib import pyplot as plt
import numpy as np

BACKGROUND_CONSTANT = 78

def prune_too_much_background(image, treshold = 0.8):
    image = image.copy()
    image = np.mean(image, axis=0).astype(np.uint8)
    image[image == BACKGROUND_CONSTANT] = 0
    print(image)

    if np.count_nonzero(image) / image.size < treshold:
        return True
    return False

def view_hsi(path:str) -> None:
    src = rasterio.open(path)
    image = src.read()
    normalize_fn = HSINormalize(
        mean=[  
            136.43702139, 136.95781982, 136.70735693, 136.91850906, 137.12465157,
            137.26050865, 137.37743316, 137.24835798, 137.04779119, 136.9453704,
            136.79646442, 136.68328908, 136.28231996, 136.02395119, 136.01146934,
            136.72767901, 137.38975674, 137.58604882, 137.61197314, 137.46675538,
            137.57319831, 137.69239868, 137.72318172, 137.76894864, 137.74861655,
            137.77535075, 137.80038781, 137.85482571, 137.88595859, 137.9490434,
            138.00128494, 138.17846624
            ],
        std=[
            33.48886853, 33.22482796, 33.4670978, 33.53758141, 33.48675988, 33.33348355,
            33.35096189, 33.63958817, 33.85081288, 34.08314358, 34.37542553, 34.60344274,
            34.80732573, 35.17761688, 35.1956623, 34.43121367, 33.76600779, 33.77061146,
            33.92844916, 34.0370747, 34.0285642, 33.87601205, 33.81035869, 33.66611756,
            33.74440912, 33.69755911, 33.69845938, 33.6707364, 33.62571536, 33.44615438,
            33.27907802, 32.90732107
            ],
    )
    image = np.moveaxis(image, 0, -1)

    print(normalize_fn)

    image = image + 32768
    image = image // 256
    
    image = image * 256

    image = image - 32768
    image = image/10000
    

    
    print(image.shape)
    print(src.count)
    print(np.min(image))
    print(np.max(image))
    print(prune_too_much_background(image))
    show(src)

def plot_distribution(path:str):
    # brihgtness distribution at channel 0
    with rasterio.open(path) as src:
        img = src.read()

    
    print(img.min(), img.max())
    img = np.moveaxis(img,0, -1)


    mask = np.where(img < 0 , True, False)

    print(img.min())
    print(img.max())
    brightnesses = img.copy()
    brightness_dist = np.bincount(brightnesses.reshape(512*512, 32)[:,2])
    print(brightness_dist)
    #plot brightness distribution
    plt.bar(np.arange(256), brightness_dist)
    plt.title('Brightness Distribution')
    plt.xlabel('Brightness')
    plt.ylabel('Count')
    plt.show()


class HSINormalize(object):
    """Normalize the HSI image with 32 bands.

    Args:
        mean (sequence): Mean values of 32 channels.
        std (sequence): Std values of 32 channels.
    """

    def __init__(self, mean, std):
        if len(mean) != 32 or len(std) != 32:
            raise ValueError("mean and std must have 32 elements each")
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        print("called")
        img = results['img'].astype(np.float32)
   
        for i in range(img.shape[2]):  # Assuming img is in shape [H, W, C]
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        
        results['img'] = img
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std)
        return results

if __name__ == "__main__":
    # view_hsi("D:\\data\\tr\\image\\O1_0002.tif")
    view_hsi("D:\\data\\tr\\image\\O1_0010.tif")