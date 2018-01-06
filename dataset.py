import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle

data_dir = os.getcwd() + '/data/'


class Dataset:

    def __init__(
        self,
        batch_size,
        valid_split=0.2,
        test_split=0.2,
        ):
        self.batch_size = batch_size
        self.valid_split = valid_split
        self.test_split = test_split

    # Read data

    def read_data(self):
        vehicle_folder = data_dir + '/vehicles/'
        non_vehicles_folder = data_dir + '/non-vehicles/'

        vehicles = glob.glob('{}*/*.png'.format(vehicle_folder),
                             recursive=True)
        vehicles_labels = np.ones(len(vehicles))

        non_vehicles = \
            glob.glob('{}*/*.png'.format(non_vehicles_folder),
                      recursive=True)
        non_vehicles_labels = np.zeros(len(non_vehicles))

        x = vehicles + non_vehicles
        y = np.concatenate((vehicles_labels, non_vehicles_labels))

        data = shuffle(list(zip(x, y)))
        (train, test) = train_test_split(data,
                test_size=self.test_split)
        (train, valid) = train_test_split(train,
                test_size=self.valid_split)
        return (train, valid, test)

    def read_augmented_data(self):
        augmented_data_folder = data_dir + '/augmented_data'
        augmented = \
            glob.glob('{}/*.png'.format(augmented_data_folder),
                      recursive=True)
        augmented_labels = np.ones(len(augmented))

        data = shuffle(list(zip(augmented, augmented_labels)))
        (train, test) = train_test_split(data,
                test_size=self.test_split)
        (train, valid) = train_test_split(train,
                test_size=self.valid_split)
        return (train, valid, test)

    def get_generators(
        self,
        train_data,
        validation_data,
        test_data,
        use_test_set=True,
        ):

        def generator(data, batch_size):
            while True:
                shuffle(data)
                for offset in range(0, len(data), batch_size):
                    batch = data[offset:offset + batch_size]

                    (x, y) = ([], [])
                    for item in batch:
                        _y = item[1]
                        _x = cv2.cvtColor(cv2.imread(item[0]),
                                cv2.COLOR_BGR2RGB)
                        x.append(_x)
                        y.append(_y)

                        x.append(cv2.flip(_x, 1))
                        y.append(_y)

                    x = np.array(x)
                    y = np.array(y)

                    yield shuffle(x, np.expand_dims(y, axis=1))

        test_generator = None
        if use_test_set:
            train_data = train_data + test_data
        else:
            test_generator = generator(test_data, self.batch_size)

        train_generator = generator(train_data, self.batch_size)
        validation_generator = generator(validation_data,
                self.batch_size)
        return (train_generator, validation_generator, test_generator)

