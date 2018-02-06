import tensorflow as tf
import os
from sklearn.utils import shuffle as sklearn_shuffle


class CDiscountDataset:
    def __init__(self, folder):
        self.dataset_folder = folder
        self.train_folder = os.path.join(self.dataset_folder, 'train')
        self.test_folder = os.path.join(self.dataset_folder, 'test')
        self.val_folder = os.path.join(self.dataset_folder, 'val')

        if not os.path.exists(self.train_folder) or \
                not os.path.exists(self.test_folder) or \
                not os.path.exists(self.val_folder):
            raise ValueError('Folder should consist `train`, `test` and `val` folders with images')

        self.class_mapping = self._read_class_mapping()
        self.num_classes = len(self.class_mapping)

    def _read_class_mapping(self):
        """
        Creating dict where key is a class label which is a name of directory and value
        is a numeric integer value from 0 to n_classes - 1.
        """
        classes = set()
        for cls in os.listdir(self.train_folder) + os.listdir(self.val_folder):
            classes.add(cls)
        n_classes = len(classes)
        sorted_names_of_classes = sorted(list(classes))
        return dict(zip(sorted_names_of_classes, range(n_classes)))

    def _get_files_and_labels(self, subfolder):
        if subfolder == 'test':
            filenames = [os.path.join(self.test_folder, img) for img in os.listdir(self.test_folder)]
            labels = [-1] * len(filenames)
        else:
            filenames, labels = [], []
            dataset_subfolder = os.path.join(self.dataset_folder, subfolder)
            for folder in os.listdir(dataset_subfolder):
                for img in os.listdir(os.path.join(dataset_subfolder, folder)):
                    filenames.append(os.path.join(dataset_subfolder, folder, img))
                    labels.append(self.class_mapping[folder])

        return filenames, labels

    def get_train_iterator(self, epochs, batch_size, preprocessing_function, shuffle=True):
        return self._get_iterator('train', epochs, batch_size, preprocessing_function, shuffle)

    def get_val_iterator(self, epochs, batch_size, preprocessing_function, shuffle=True):
        return self._get_iterator('val', epochs, batch_size, preprocessing_function, shuffle)

    def get_test_iterator(self, epochs, batch_size, preprocessing_function, shuffle=False):
        return self._get_iterator('test', epochs, batch_size, preprocessing_function, shuffle)

    def _get_iterator(self, subfolder, epochs, batch_size, preprocessing_function, shuffle):
        filenames, labels = self._get_files_and_labels(subfolder)
        if shuffle:
            filenames, labels = sklearn_shuffle(filenames, labels)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.repeat(epochs)
        dataset = dataset.map(_parse_function, num_parallel_calls=20)
        if preprocessing_function:
            dataset = dataset.map(preprocessing_function, num_parallel_calls=20)
        if batch_size:
            dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(10)
        iterator = dataset.make_initializable_iterator()
        return iterator


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    return image_decoded, label
