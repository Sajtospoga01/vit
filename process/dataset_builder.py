import numpy as np
import os
from tqdm import tqdm
import rasterio
import pandas as pd


import numpy as np

BACKGROUND_CONSTANT = 78

def prune_too_much_background(image, treshold = 0.9):
    image = image.copy()
    image = np.mean(image, axis=0).astype(np.uint8)
    image[image == BACKGROUND_CONSTANT] = 0

    if np.count_nonzero(image) / image.size < treshold:
        return True
    return False

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



class S3DatasetOptimizer:
    """
    Builds image batches from a dataset providing optimised IO performance
    :batch_size int: batch size to group images in
    :path string: path to the dataset folder
    :return tuple(list(string),list(string)): tuple of filenames for in and out
    """

    def __init__(self, batch_size: int, path: str):
        self.batch_size = batch_size
        self.path = path
        self.x_names = self.__build_dataset(batch_size, path)

    def __build_dataset(self, batch_size: int, path: str):
        """
        Build dataset from the given path
        :batch_size int: batch size to group images in
        :path string: path to the dataset folder
        :return tuple(list(string),list(string)): tuple of filenames for in and out
        """

        X = sorted(os.listdir(os.path.join(path, "image")))
        # y = sorted(os.listdir(os.path.join(path, "label")))
        # for X_val in tqdm(zip(X), total=len(X)):
        #     if X_val.split(".")[0] != y_val.split(".")[0]:
        #         raise ValueError("Image and label names do not match")

        pruned = []
        tqdm_instance = tqdm(enumerate(X), total=len(X))
        total_pruned = 0
        tqdm_instance.set_description(f"Pruned {total_pruned} images")
        for i,im_name in tqdm_instance:
            with rasterio.open(os.path.join(path, "image", im_name)) as im:
                im = im.read()
                if prune_too_much_background(im):
                    pruned.append(i)
                    total_pruned += 1
                    tqdm_instance.set_description(f"Pruned {total_pruned} images")

        for i in pruned[::-1]:
            X.pop(i)
            # y.pop(i)


        n_batch = len(X) // batch_size
        diff = len(X) % batch_size

        X_batches = np.array_split(X[:-diff], n_batch)
        # y_batches = np.array_split(y[:-diff], n_batch)

        # add diff number of unknown to last batch
        X_last = X[-diff:]
        # y_last = y[-diff:]

        for _ in range(batch_size - diff):
            X_last.append("#")
        # y_last.append("#")

        X_last = np.array(X_last)
        # y_last = np.array(y_last)

        X_batches.append(X_last)
        # y_batches.append(y_last)

        # append last not ful batch to end of list

        return X_batches  # , y_batches

    def export_dataset(
            self,
            output_path: str,
            image_size:tuple[int,int],
            n_channels:int,
            dtype=np.uint8,
    ):
        """
        Export dataset in batches to the given path

        Parameters
        ----------
        :input_path string: path to the dataset folder
        :output_path string: path to the output folder
        :batch_size int: batch size to group images in

        Keyword Arguments
        -----------------
        :dtype numpy.dtype: data type of the images, defaults to np.int8
        """
        number_of_images_exported = 0
        batch_size_ex = self.batch_size

        for i, X in tqdm(enumerate(self.x_names), total=len(self.x_names)):
            X_batch = np.zeros((batch_size_ex, n_channels, *image_size), dtype=dtype)
            # y_batch = np.zeros((batch_size_ex, 1, 512, 512), dtype=dtype)

            for j, X_image_name in enumerate(X):
                if X_image_name == "#":
                    continue
                X_input_path = os.path.join(self.path, "image", X_image_name)
                # y_input_path = os.path.join(self.path, "label", y_image_name)

                with rasterio.open(X_input_path) as X_image:
                    X_batch[j, ...] = np.moveaxis(resize_single_image(np.moveaxis(X_image.read(),0,-1), image_size[1], image_size[0]),-1,0)
                # with rasterio.open(y_input_path) as y_image:
                #     y_batch[j, ...] = y_image.read()
                number_of_images_exported += 1
            X_output_path = os.path.join(output_path, "image", f"batch_{str(i)}")
            # y_output_path = os.path.join(output_path, "label", f"batch_{str(i)}")

            X_batch = np.moveaxis(X_batch, 1, -1)
            # y_batch = np.moveaxis(y_batch, 1, -1)

            np.save(X_output_path, X_batch)
            # np.save(y_output_path, y_batch)
        df = pd.DataFrame([number_of_images_exported, batch_size_ex])
        df.to_csv(os.path.join(output_path, "info.csv"))


def load_image_batch(path: str, og_path: list[str], batch_number: int):
    """
    Load image batch from the given path
    :param path: path to the image batch
    :param batch_number: batch number
    :return: image batch
    """
    X_path = os.path.join(path, "image", str(batch_number) + ".npy")
    y_path = os.path.join(path, "label", str(batch_number) + ".npy")

    X_batch = np.load(X_path)
    y_batch = np.load(y_path)

    print(X_batch.shape)
    print(X_batch[0])
    with rasterio.open("C:\\Users\\andra\\guorbit\\dataset\\e_tr\\image\\" + og_path[0]) as og_image:
        og_image = og_image.read()
        print(og_image.shape)
        print(og_image[0])

    print(np.array_equal(X_batch[0], og_image))


if __name__ == "__main__":
    folder_to_process = [["p_tr", "pb_tr"], ["p_val", "pb_val"]]
    BATCH_SIZE = 256
    for in_folder, out_folder in folder_to_process:
        in_path = os.path.join("F:\\retiled", in_folder)
        out_path = os.path.join("C:\\Users\\andra\\Downloads\\bchsi", out_folder)
        exporter = S3DatasetOptimizer(BATCH_SIZE, in_path)
        exporter.export_dataset(out_path,(64,64),32,)
