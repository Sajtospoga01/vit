#%%
import os
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm
from utilities.transform_utils.image_cutting import image_cut


# Assuming you have a 3D array 'min_values_per_class' and 'max_values_per_class'
# of shape (num_classes, num_bands) containing the min and max values for each class and each band
def resize_single_image(image, new_width, new_height):
    old_height, old_width, _ = image.shape

    # Create a set of indices for the new image
    x_indices = (np.arange(new_height) * (old_height / new_height)).astype(int)
    y_indices = (np.arange(new_width) * (old_width / new_width)).astype(int)

    # Use numpy's advanced indexing to pull out the correct pixels from the original image
    x_indices_mesh, y_indices_mesh = np.meshgrid(x_indices, y_indices, indexing='ij')

    # Index into the original image to get the resized image
    resized_image = image[x_indices_mesh, y_indices_mesh]

    return resized_image


def main():
    directories = [["tr", "f_tr"], ["val", "f_val"],["ts", "f_ts"]]
    for directory in directories:
        im_path = "D:\\data\\" + directory[0] + "\\image"
        label_path = "D:\\data\\" + directory[0] + "\\label"
        e_im_path = "D:\\data\\" + directory[1] + "\\image"
        e_label_path = "D:\\data\\" + directory[1] + "\\label"

        if not os.path.exists(e_im_path):
            os.makedirs(e_im_path)
        if not os.path.exists(e_label_path):
            os.makedirs(e_label_path)

        files = sorted(os.listdir(im_path))
        image_path_names = [os.path.join(im_path, f) for f in files]
        label_path_names = [os.path.join(label_path, f) for f in files]
        e_image_path_names = [os.path.join(e_im_path, f) for f in files]
        e_label_path_names = [os.path.join(e_label_path, f) for f in files]
        for im_filename, label_filename, e_im_filename, e_label_filename in tqdm(
            zip(
                image_path_names,
                label_path_names,
                e_image_path_names,
                e_label_path_names,
            ),
            total=len(image_path_names),
        ):
            with rasterio.open(im_filename) as src:
                image = src.read()


            image = np.moveaxis(image, 0, -1)
            image = image + 32768
            image = image / 256

            image = image.astype(np.float32)


            label = Image.open(label_filename)
            label = np.array(label)

            image = resize_single_image(image, 64, 64)
            # cut_ims = np.expand_dims(cut_ims, axis=0)

            label = np.squeeze(resize_single_image(np.expand_dims(label,axis=-1), 64, 64))
            # cut_labels = np.expand_dims(cut_labels, axis=0)

            # for i,(im,la) in enumerate(zip(cut_ims, cut_labels)):
            #     im = im.astype(np.uint8)
            #     im = np.moveaxis(im, -1, 0)
                
            #     with rasterio.open(
            #         e_im_filename[:-4] + "_" + str(i) + ".tif",
            #         "w",
            #         driver="GTiff",
            #         width=im.shape[1],
            #         height=im.shape[2],
            #         count=im.shape[0],
            #         dtype=rasterio.uint8,
            #     ) as dst:

            #         dst.write(im)
            image = np.moveaxis(image, -1, 0)
            with rasterio.open(
                e_im_filename,
                "w",
                driver="GTiff",
                width=image.shape[2],
                height=image.shape[1],
                count=image.shape[0],
                dtype=rasterio.uint8,
            ) as dst:

                dst.write(image)
  

            label = label.astype(np.uint8)
            Image.fromarray(label).save(e_label_filename)
 #%%
def get_pixel_min_max_range():
    #%%
    directories = [["tr", "f_tr"], ["val", "f_val"], ["ts", "f_ts"]]

    for directory in directories:

        im_path = "D:\\data\\" + directory[0] + "\\image"
        label_path = "D:\\data\\" + directory[0] + "\\label"

        files = sorted(os.listdir(im_path))
        image_path_names = [os.path.join(im_path, f) for f in files]
        label_path_names = [os.path.join(label_path, f) for f in files]

        mean = None
        M2 = None
        n = 0
        i = 0

        # Track class distribution
        class_distribution = {}

        for im_filename, label_filename in tqdm(
                zip(image_path_names, label_path_names), total=len(image_path_names)
        ):
            with rasterio.open(im_filename) as src:
                image = src.read()

            with rasterio.open(label_filename) as src:
                label = src.read(1)  # Assuming labels in band 1

            # Update class distribution
            unique_classes = np.unique(label)
            for cls in unique_classes:
                if cls in class_distribution:
                    class_distribution[cls] += np.count_nonzero(label == cls)
                else:
                    class_distribution[cls] = np.count_nonzero(label == cls)

            if mean is None:
                mean = np.zeros(image.shape[0])
                M2 = np.zeros(image.shape[0])

            n += image.shape[1] * image.shape[2]
            delta = image - mean[:, None, None]
            mean += delta.sum(axis=(1, 2)) / n
            delta2 = image - mean[:, None, None]
            M2 += (delta * delta2).sum(axis=(1, 2))
            i += 1

        variance_n = M2 / n
        std_dev = np.sqrt(variance_n)

        print(f"Directory: {directory[0]}")
        print("Mean per band:", mean)
        print("Standard deviation per band:", std_dev)
        print("Class Distribution:", class_distribution)  # Print class distribution
        #%%
        # plot the distribution of the classes
        import matplotlib.pyplot as plt

        plt.bar(class_distribution.keys(), class_distribution.values())
        plt.show()

        #without background
        class_distribution.pop(0, None)
        plt.bar(class_distribution.keys(), class_distribution.values())
        plt.show()

        #%%


if __name__ == "__main__":
    get_pixel_min_max_range()
    # main()

# %%
