import numpy as np
import os
import tensorflow as tf
from scipy.interpolate import splprep, splev
from Utils import show_positions_mus


class Mnist_Image2Image_Dataloader():
    L = 28
    BUFFER_SIZE = 8192
    BATCH_SIZE = 256
    EPOCH = 5
    DATASET_DIR = "Dataset/Mnist/"
    TYPE = 'train'
    INITIAL_LENGTH = 10

    def __init__(self, type=TYPE, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, epoch=EPOCH, dataset_dir=DATASET_DIR,
                 initial_length=INITIAL_LENGTH):
        assert type == 'train' or type == 'test'
        self.DATASET_DIR = dataset_dir
        self.TYPE = type
        if type == 'train':
            self.N = 60000
        else:
            self.N = 10000

        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        self.EPOCH = epoch
        self.INITIAL_LENGTH = initial_length
        self.load_sequence_list()
        self.create_dataset()

    def generator(self):
        for i, sequence_points in enumerate(self.sequence_list):
            data = np.zeros([self.L, self.L])
            l = 0
            for p1, p2 in zip(sequence_points[:-1], sequence_points[1:]):
                l += 1
                data[p1[1], p1[0]] = 1.0
                y = (p2[0]) + (p2[1]) * self.L
                if l >= self.INITIAL_LENGTH:
                    yield (data.copy(), y)
                else:
                    pass

    def validation_generator(self):
        for sequence_points in self.validation_sequence_list:
            data = np.zeros([self.L, self.L])
            l = 0
            for p1, p2 in zip(sequence_points[:-1], sequence_points[1:]):
                l += 1
                data[p1[1], p1[0]] = 1.0
                y = (p2[0]) + (p2[1]) * self.L
                if l >= self.INITIAL_LENGTH:
                    yield (data.copy(), y)
                else:
                    pass

    def create_dataset(self):
        self.dataset = tf.data.Dataset.from_generator(
            self.generator, (tf.float32, tf.int64),
            (tf.TensorShape([self.L, self.L]), tf.TensorShape([]))
        )
        self.dataset = self.dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=42).repeat().batch(
            self.BATCH_SIZE)
        self.dataset_iterator = self.dataset.make_one_shot_iterator()
        self.get_batch = self.dataset_iterator.get_next()
        if self.TYPE == 'train':
            self.validation_dataset = tf.data.Dataset.from_generator(
                self.validation_generator, (tf.float32, tf.int64),
                (tf.TensorShape([self.L, self.L]), tf.TensorShape([])))
            self.validation_dataset = self.validation_dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=42).repeat(
                self.EPOCH).batch(self.BATCH_SIZE)

            self.validation_dataset_iterator = self.validation_dataset.make_one_shot_iterator()
            self.get_validation_batch = self.validation_dataset_iterator.get_next()

    def load_sequence_list(self):
        try:
            self.sequence_list = np.load(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"))
            if self.TYPE == 'train':
                self.validation_sequence_list = np.load(
                    os.path.join(self.DATASET_DIR + "cache/", "validation_points.npy"))
        except:
            self.sequence_list = []
            print("Creating Cache file for mnist dataset")
            for i in range(self.N):
                if i % 100 == 0:
                    print("{}/{}".format(i, self.N))
                file_name = self.DATASET_DIR + "sequences/{}img-{}-points.txt".format(self.TYPE, i)
                sequence_points = open(file_name, 'r').readlines()[1:]
                sequence_points = [tuple(map(lambda x: int(x) - 1, s[:-1].split(","))) for s in sequence_points]
                sequence_points = [s for s in sequence_points if s[0] > -1]
                self.sequence_list.append(sequence_points)
            if self.TYPE == 'train':
                np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"),
                        self.sequence_list[:-10000])
                np.save(os.path.join(self.DATASET_DIR + "cache/", "validation_points.npy"), self.sequence_list[-10000:])
                self.validation_sequence_list = self.sequence_list[-10000:]
                self.sequence_list = self.sequence_list[:-10000]
            else:
                np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"), self.sequence_list)


class Mnist_Temporal_Image2Image_Dataloader():
    L = 28
    BUFFER_SIZE = 8192
    BATCH_SIZE = 256
    EPOCH = 5
    DATASET_DIR = "Dataset/Mnist/"
    TYPE = 'train'
    INITIAL_LENGTH = 10
    MAX_LEN = 120

    def __init__(self, type=TYPE, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, epoch=EPOCH, dataset_dir=DATASET_DIR,
                 initial_length=INITIAL_LENGTH, max_len=MAX_LEN):
        assert type == 'train' or type == 'test'
        self.DATASET_DIR = dataset_dir
        self.TYPE = type
        if type == 'train':
            self.N = 60000
        else:
            self.N = 10000
        assert max_len > initial_length
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        self.EPOCH = epoch
        self.INITIAL_LENGTH = initial_length
        self.MAX_LEN = max_len
        self.load_sequence_list()
        self.create_dataset()

    def generator(self):
        for i, sequence_points in enumerate(self.sequence_list):
            # if len(sequence_points) <= self.INITIAL_LENGTH:
            if len(sequence_points) <= 10:
                continue
            x = np.zeros([self.L, self.L, self.MAX_LEN])
            y = np.zeros([self.L, self.L, self.MAX_LEN])
            for j, p in enumerate(sequence_points[:self.MAX_LEN]):
                if j > 0:
                    x[:, :, j] = x[:, :, j - 1]
                x[p[1], p[0], j] = 1.0
                if j >= self.INITIAL_LENGTH:
                    y[p[1], p[0], j - self.INITIAL_LENGTH] = 1.0
            l = len(sequence_points) - self.INITIAL_LENGTH
            yield (x, y, l)

    def validation_generator(self):
        for i, sequence_points in enumerate(self.validation_sequence_list):
            # if len(sequence_points) <= self.INITIAL_LENGTH:
            if len(sequence_points) <= 10:
                continue
            x = np.zeros([self.L, self.L, self.MAX_LEN])
            y = np.zeros([self.L, self.L, self.MAX_LEN])
            for j, p in enumerate(sequence_points[:self.MAX_LEN]):
                if j > 0:
                    x[:, :, j] = x[:, :, j - 1]
                x[p[1], p[0], j] = 1.0
                if j >= self.INITIAL_LENGTH:
                    y[p[1], p[0], j - self.INITIAL_LENGTH] = 1.0
            l = len(sequence_points) - self.INITIAL_LENGTH
            yield (x, y, l)

    def create_dataset(self):
        self.dataset = tf.data.Dataset.from_generator(
            self.generator, (tf.float32, tf.float32, tf.int64),
            (tf.TensorShape([self.L, self.L, self.MAX_LEN]), tf.TensorShape([self.L, self.L, self.MAX_LEN]),
             tf.TensorShape([]))
        )
        self.dataset = self.dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=42).repeat(
            -1 if self.TYPE == 'train' else 1)
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.BATCH_SIZE))
        self.dataset_iterator = self.dataset.make_one_shot_iterator()
        self.get_batch = self.dataset_iterator.get_next()
        self.get_validation_batch = (None, None, None)
        if self.TYPE == 'train':
            self.validation_dataset = tf.data.Dataset.from_generator(
                self.validation_generator, (tf.float32, tf.float32, tf.int64),
                (tf.TensorShape([self.L, self.L, self.MAX_LEN]), tf.TensorShape([self.L, self.L, self.MAX_LEN]),
                 tf.TensorShape([]))
            )
            self.validation_dataset = self.validation_dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=42).repeat()
            self.validation_dataset = self.validation_dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(self.BATCH_SIZE))

            self.validation_dataset_iterator = self.validation_dataset.make_one_shot_iterator()
            self.get_validation_batch = self.validation_dataset_iterator.get_next()

    def load_sequence_list(self):
        try:
            self.sequence_list = np.load(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"))
            if self.TYPE == 'train':
                self.validation_sequence_list = np.load(
                    os.path.join(self.DATASET_DIR + "cache/", "validation_points.npy"))
        except:
            self.sequence_list = []
            print("Creating Cache file for mnist dataset")
            for i in range(self.N):
                if i % 100 == 0:
                    print("{}/{}".format(i, self.N))
                file_name = self.DATASET_DIR + "sequences/{}img-{}-points.txt".format(self.TYPE, i)
                sequence_points = open(file_name, 'r').readlines()[1:]
                sequence_points = [tuple(map(lambda x: int(x) - 1, s[:-1].split(","))) for s in sequence_points]
                sequence_points = [s for s in sequence_points if s[0] > -1]
                self.sequence_list.append(sequence_points)
            if self.TYPE == 'train':
                np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"),
                        self.sequence_list[:-10000])
                np.save(os.path.join(self.DATASET_DIR + "cache/", "validation_points.npy"), self.sequence_list[-10000:])
                self.validation_sequence_list = self.sequence_list[-10000:]
                self.sequence_list = self.sequence_list[:-10000]
            else:
                np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"), self.sequence_list)


class Mnist_Relative_Sequence_Dataloader():
    L = 28
    BUFFER_SIZE = 8192
    BATCH_SIZE = 256
    EPOCH = 5
    DATASET_DIR = "Dataset/Mnist/"
    TYPE = 'train'
    MAX_LEN = 120

    def __init__(self, type=TYPE, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, epoch=EPOCH, dataset_dir=DATASET_DIR,
                 max_len=MAX_LEN):
        assert type == 'train' or type == 'test'
        self.DATASET_DIR = dataset_dir
        self.TYPE = type
        if type == 'train':
            self.N = 60000
        else:
            self.N = 10000

        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        self.EPOCH = epoch
        self.MAX_LEN = max_len
        self.load_sequence_list()
        self.create_dataset()

    def generator(self):
        for sequence_points in self.sequence_list:
            x = np.zeros([self.MAX_LEN, 4])
            y = np.zeros([self.MAX_LEN, 4])
            x1 = sequence_points[:-1]
            y1 = sequence_points[1:]
            l = len(x1)
            if l <= self.MAX_LEN:
                x[:len(x1)] = x1
                y[:len(y1)] = y1
            else:
                x[:self.MAX_LEN] = x1[:self.MAX_LEN]
                y[:self.MAX_LEN] = y1[:self.MAX_LEN]
                l = self.MAX_LEN
            yield (x, y, l)

    def validation_generator(self):
        for sequence_points in self.validation_sequence_list:
            x = np.zeros([self.MAX_LEN, 4])
            y = np.zeros([self.MAX_LEN, 4])
            x1 = sequence_points[:-1]
            y1 = sequence_points[1:]
            l = len(x1)
            if l <= self.MAX_LEN:
                x[:len(x1)] = x1
                y[:len(y1)] = y1
            else:
                x[:self.MAX_LEN] = x1[:self.MAX_LEN]
                y[:self.MAX_LEN] = y1[:self.MAX_LEN]
                l = self.MAX_LEN
            yield (x, y, l)

    def create_dataset(self):
        self.dataset = tf.data.Dataset.from_generator(
            self.generator, (tf.float32, tf.float32, tf.int32),
            (tf.TensorShape([self.MAX_LEN, 4]), tf.TensorShape([self.MAX_LEN, 4]), tf.TensorShape([]))
        )
        self.dataset = self.dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=42).repeat().batch(
            self.BATCH_SIZE)
        self.dataset_iterator = self.dataset.make_one_shot_iterator()
        self.get_batch = self.dataset_iterator.get_next()
        if self.TYPE == 'train':
            self.validation_dataset = tf.data.Dataset.from_generator(
                self.validation_generator, (tf.float32, tf.float32, tf.int32),
                (tf.TensorShape([self.MAX_LEN, 4]), tf.TensorShape([self.MAX_LEN, 4]), tf.TensorShape([]))
            )
            self.validation_dataset = self.validation_dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=42).repeat(
                self.EPOCH).batch(self.BATCH_SIZE)

            self.validation_dataset_iterator = self.validation_dataset.make_one_shot_iterator()
            self.get_validation_batch = self.validation_dataset_iterator.get_next()

    def load_sequence_list(self):
        try:
            self.sequence_list = np.load(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_input_sequence.npy"))
            if self.TYPE == 'train':
                self.validation_sequence_list = np.load(
                    os.path.join(self.DATASET_DIR + "cache/", "validation_input_sequence.npy"))
        except:
            self.sequence_list = []
            print("Creating Cache file for mnist dataset")
            for i in range(self.N):
                if i % 100 == 0:
                    print("{}/{}".format(i, self.N))
                file_name = self.DATASET_DIR + "sequences/{}img-{}-inputdata.txt".format(self.TYPE, i)
                input_sequence = open(file_name, 'r').readlines()
                input_sequence = [list(map(float, s[:-1].split())) for s in input_sequence]
                self.sequence_list.append(input_sequence)
            if self.TYPE == 'train':
                np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_input_sequence.npy"),
                        self.sequence_list[:-10000])
                np.save(os.path.join(self.DATASET_DIR + "cache/", "validation_input_sequence.npy"),
                        self.sequence_list[-10000:])
                self.validation_sequence_list = self.sequence_list[-10000:]
                self.sequence_list = self.sequence_list[:-10000]
            else:
                np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_input_sequence.npy"),
                        self.sequence_list)

        # self.MAX_LEN = max(self.MAX_LEN, max([len(s) for s in self.sequence_list]))
        # if type == 'train':
        #     self.MAX_LEN = max(self.MAX_LEN, max([len(s) for s in self.validation_sequence_list]))


class Mnist_Absolute_Sequence_Dataloader():
    L = 28
    BUFFER_SIZE = 8192
    BATCH_SIZE = 256
    DATASET_DIR = "Dataset/Mnist/"
    TYPE = 'train'
    MAX_LEN = 120
    INITIAL_LENGTH = 10
    SEP = False

    def __init__(self, type=TYPE, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, dataset_dir=DATASET_DIR,
                 max_len=MAX_LEN, initial_length=INITIAL_LENGTH, sep=SEP):
        assert type == 'train' or type == 'test'
        self.DATASET_DIR = dataset_dir
        self.TYPE = type
        if type == 'train':
            self.N = 60000
        else:
            self.N = 10000

        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        self.MAX_LEN = max_len
        self.INITIAL_LENGTH = initial_length
        self.SEP = sep
        self.load_sequence_list()
        self.create_dataset()
        if self.TYPE == 'test':
            self.get_validation_batch = self.get_batch

    def generator(self):
        for sequence_points in self.sequence_list:

            x = np.zeros([self.MAX_LEN, 2])
            y = np.zeros([self.MAX_LEN, 2])
            x1 = sequence_points[:-1]
            y1 = sequence_points[1:]
            l = len(x1)
            if l < self.INITIAL_LENGTH:
                continue
            if l <= self.MAX_LEN:
                x[:len(x1)] = x1
                y[:len(y1)] = y1
            else:
                x[:self.MAX_LEN] = x1[:self.MAX_LEN]
                y[:self.MAX_LEN] = y1[:self.MAX_LEN]
                l = self.MAX_LEN
            yield (x, y, l)

    def validation_generator(self):
        for sequence_points in self.validation_sequence_list:
            x = np.zeros([self.MAX_LEN, 2])
            y = np.zeros([self.MAX_LEN, 2])
            x1 = sequence_points[:-1]
            y1 = sequence_points[1:]
            l = len(x1)
            if l < self.INITIAL_LENGTH:
                continue
            if l <= self.MAX_LEN:
                x[:len(x1)] = x1
                y[:len(y1)] = y1
            else:
                x[:self.MAX_LEN] = x1[:self.MAX_LEN]
                y[:self.MAX_LEN] = y1[:self.MAX_LEN]
                l = self.MAX_LEN
            yield (x, y, l)

    def sep_generator(self):
        for sequence_points in self.sequence_list:

            x = np.zeros([self.INITIAL_LENGTH, 2])
            y = np.zeros([self.MAX_LEN, 2])
            x1 = sequence_points[:self.INITIAL_LENGTH]
            y1 = sequence_points[self.INITIAL_LENGTH:]
            x[:len(x1)] = x1
            l = len(y1)
            if len(x1) < self.INITIAL_LENGTH:
                continue
            if l <= self.MAX_LEN:
                y[:len(y1)] = y1
            else:
                y[:self.MAX_LEN] = y1[:self.MAX_LEN]
                l = self.MAX_LEN
            yield (x, y, l)

    def sep_validation_generator(self):
        for sequence_points in self.validation_sequence_list:
            x = np.zeros([self.INITIAL_LENGTH, 2])
            y = np.zeros([self.MAX_LEN, 2])
            x1 = sequence_points[:self.INITIAL_LENGTH]
            y1 = sequence_points[self.INITIAL_LENGTH:]
            x[:len(x1)] = x1
            l = len(y1)
            if len(x1) < self.INITIAL_LENGTH:
                continue
            if l <= self.MAX_LEN:
                y[:len(y1)] = y1
            else:
                y[:self.MAX_LEN] = y1[:self.MAX_LEN]
                l = self.MAX_LEN
            yield (x, y, l)

    def create_dataset(self):
        if self.SEP:
            self.dataset = tf.data.Dataset.from_generator(
                self.sep_generator, (tf.float32, tf.float32, tf.int32),
                (tf.TensorShape([self.INITIAL_LENGTH, 2]), tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([]))
            )
        else:
            self.dataset = tf.data.Dataset.from_generator(
                self.generator, (tf.float32, tf.float32, tf.int32),
                (tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([]))
            )

        self.dataset = self.dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=42).repeat(
            -1 if self.TYPE == 'train' else 1).batch(
            self.BATCH_SIZE)
        self.dataset_iterator = self.dataset.make_one_shot_iterator()
        self.get_batch = self.dataset_iterator.get_next()
        if self.TYPE == 'train':
            if self.SEP:
                self.validation_dataset = tf.data.Dataset.from_generator(
                    self.sep_validation_generator, (tf.float32, tf.float32, tf.int32),
                    (tf.TensorShape([self.INITIAL_LENGTH, 2]), tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([]))
                )
            else:
                self.validation_dataset = tf.data.Dataset.from_generator(
                    self.validation_generator, (tf.float32, tf.float32, tf.int32),
                    (tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([]))
                )
            self.validation_dataset = self.validation_dataset.shuffle(buffer_size=self.BUFFER_SIZE,
                                                                      seed=42).repeat().batch(self.BATCH_SIZE)

            self.validation_dataset_iterator = self.validation_dataset.make_one_shot_iterator()
            self.get_validation_batch = self.validation_dataset_iterator.get_next()

    def load_sequence_list(self):
        try:
            self.sequence_list = np.load(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"))
            if self.TYPE == 'train':
                self.validation_sequence_list = np.load(
                    os.path.join(self.DATASET_DIR + "cache/", "validation_points.npy"))
        except:
            self.sequence_list = []
            print("Creating Cache file for mnist dataset")
            for i in range(self.N):
                if i % 100 == 0:
                    print("{}/{}".format(i, self.N))
                file_name = self.DATASET_DIR + "sequences/{}img-{}-points.txt".format(self.TYPE, i)
                sequence_points = open(file_name, 'r').readlines()[1:]
                sequence_points = [tuple(map(lambda x: int(x) - 1, s[:-1].split(","))) for s in sequence_points]
                sequence_points = [s for s in sequence_points if s[0] > -1]
                self.sequence_list.append(sequence_points)
            if self.TYPE == 'train':
                np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"),
                        self.sequence_list[:-10000])
                np.save(os.path.join(self.DATASET_DIR + "cache/", "validation_points.npy"), self.sequence_list[-10000:])
                self.validation_sequence_list = self.sequence_list[-10000:]
                self.sequence_list = self.sequence_list[:-10000]
            else:
                np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"), self.sequence_list)


class Mnist_Spline_Sequence_Dataloader():
    L = 28
    BUFFER_SIZE = 8192
    BATCH_SIZE = 256
    DATASET_DIR = "Dataset/Mnist/"
    TYPE = 'train'
    MAX_LEN = 120
    INITIAL_LENGTH = 10

    def __init__(self, type=TYPE, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, dataset_dir=DATASET_DIR,
                 max_len=MAX_LEN, initial_length=INITIAL_LENGTH):
        assert type == 'train' or type == 'test'
        self.DATASET_DIR = dataset_dir
        self.TYPE = type
        if type == 'train':
            self.N = 60000
        else:
            self.N = 10000

        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        self.MAX_LEN = max_len
        self.INITIAL_LENGTH = initial_length
        self.load_sequence_list()
        self.create_dataset()

    def generator(self):
        for sequence_points, spline_sequence_points in zip(self.sequence_list, self.spline_sequence_list):
            x = np.zeros([self.MAX_LEN, 2])
            y = np.zeros([self.MAX_LEN, 2])
            y_spline = np.zeros([self.MAX_LEN, 2])
            x1 = sequence_points[:-1]
            y1 = sequence_points[1:]
            y1_spline = spline_sequence_points[1:]
            l = len(x1)
            if l < self.INITIAL_LENGTH:
                continue
            if l <= self.MAX_LEN:
                x[:len(x1)] = x1
                y[:len(y1)] = y1
                y_spline[:len(y1_spline)] = y1_spline
            else:
                x[:self.MAX_LEN] = x1[:self.MAX_LEN]
                y[:self.MAX_LEN] = y1[:self.MAX_LEN]
                y_spline[:self.MAX_LEN] = y1_spline
                l = self.MAX_LEN
            yield (x, y, y_spline, l)

    def validation_generator(self):
        for sequence_points, spline_sequence_points in zip(self.validation_sequence_list,
                                                           self.validation_spline_sequence_list):
            x = np.zeros([self.MAX_LEN, 2])
            y = np.zeros([self.MAX_LEN, 2])
            y_spline = np.zeros([self.MAX_LEN, 2])
            x1 = sequence_points[:-1]
            y1 = sequence_points[1:]
            y1_spline = spline_sequence_points[1:]
            l = len(x1)
            if l < self.INITIAL_LENGTH:
                continue
            if l <= self.MAX_LEN:
                x[:len(x1)] = x1
                y[:len(y1)] = y1
                y_spline[:len(y1_spline)] = y1_spline
            else:
                x[:self.MAX_LEN] = x1[:self.MAX_LEN]
                y[:self.MAX_LEN] = y1[:self.MAX_LEN]
                y_spline[:self.MAX_LEN] = y1_spline
                l = self.MAX_LEN
            yield (x, y, y_spline, l)

    def create_dataset(self):
        self.dataset = tf.data.Dataset.from_generator(
            self.generator, (tf.float32, tf.float32, tf.float32, tf.int32),
            (tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([self.MAX_LEN, 2]),
             tf.TensorShape([]))
        )
        self.dataset = self.dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=42).repeat().batch(
            self.BATCH_SIZE)
        self.dataset_iterator = self.dataset.make_one_shot_iterator()
        self.get_batch = self.dataset_iterator.get_next()
        if self.TYPE == 'train':
            self.validation_dataset = tf.data.Dataset.from_generator(
                self.validation_generator, (tf.float32, tf.float32, tf.float32, tf.int32),
                (
                    tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([self.MAX_LEN, 2]),
                    tf.TensorShape([self.MAX_LEN, 2]),
                    tf.TensorShape([]))
            )
            self.validation_dataset = self.validation_dataset.shuffle(buffer_size=self.BUFFER_SIZE,
                                                                      seed=42).repeat().batch(self.BATCH_SIZE)

            self.validation_dataset_iterator = self.validation_dataset.make_one_shot_iterator()
            self.get_validation_batch = self.validation_dataset_iterator.get_next()

    def load_sequence_list(self):
        try:
            self.sequence_list = np.load(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"))
            if self.TYPE == 'train':
                self.validation_sequence_list = np.load(
                    os.path.join(self.DATASET_DIR + "cache/", "validation_points.npy"))
        except:
            self.sequence_list = []
            print("Creating Cache file for mnist dataset")
            for i in range(self.N):
                if i % 100 == 0:
                    print("{}/{}".format(i, self.N))
                file_name = self.DATASET_DIR + "sequences/{}img-{}-points.txt".format(self.TYPE, i)
                sequence_points = open(file_name, 'r').readlines()[1:]
                sequence_points = [tuple(map(lambda x: int(x) - 1, s[:-1].split(","))) for s in sequence_points]
                sequence_points = [s for s in sequence_points if s[0] > -1]
                self.sequence_list.append(sequence_points)
            if self.TYPE == 'train':
                np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"),
                        self.sequence_list[:-10000])
                np.save(os.path.join(self.DATASET_DIR + "cache/", "validation_points.npy"), self.sequence_list[-10000:])
                self.validation_sequence_list = self.sequence_list[-10000:]
                self.sequence_list = self.sequence_list[:-10000]
            else:
                np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"), self.sequence_list)

        try:
            self.spline_sequence_list = np.load(
                os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_spline_points.npy"))
        except:
            self.spline_sequence_list = []
            for i, s in enumerate(self.sequence_list):
                print(i)
                spline_s = self.fit_spline_to_sequence(s)
                self.spline_sequence_list.append(spline_s)
            np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_spline_points.npy"), self.sequence_list)

        if self.TYPE == 'train':
            try:
                self.validation_spline_sequence_list = np.load(
                    os.path.join(self.DATASET_DIR + "cache/", "validation_spline_points.npy"))
            except:
                self.validation_spline_sequence_list = []
                for i, s in enumerate(self.validation_sequence_list):
                    spline_s = self.fit_spline_to_sequence(s)
                    if len(spline_s) == 0:
                        print("Sag")
                    self.validation_spline_sequence_list.append(spline_s)
                np.save(os.path.join(self.DATASET_DIR + "cache/", "validation_spline_points.npy"),
                        self.validation_spline_sequence_list)

    def fit_spline_to_sequence(self, sequence):
        predicted_sequence = sequence[:self.INITIAL_LENGTH]

        x = np.array(sequence)[:, 0]
        y = np.array(sequence)[:, 1]
        l = len(sequence)
        if l <= self.INITIAL_LENGTH:
            return []

        for i in range(self.INITIAL_LENGTH, l):
            w = [0.99 ** j for j in range(i)]
            tck, u = splprep([x[:i], y[:i]], s=7.7, w=w)
            x_p, y_p = splev([1 + 1 / (i + 1)], tck)
            predicted_sequence.append([x_p[0], y_p[0]])

        return predicted_sequence


class Mnist_Hybrid_Image2Image_Dataloader():
    L = 28
    BUFFER_SIZE = 8192
    BATCH_SIZE = 256
    EPOCH = 5
    DATASET_DIR = "Dataset/Mnist/"
    TYPE = 'train'
    INITIAL_LENGTH = 10
    MAX_LEN = 120

    def __init__(self, type=TYPE, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, epoch=EPOCH, dataset_dir=DATASET_DIR,
                 initial_length=INITIAL_LENGTH, max_len=MAX_LEN):
        assert type == 'train' or type == 'test'
        self.DATASET_DIR = dataset_dir
        self.TYPE = type
        if type == 'train':
            self.N = 60000
        else:
            self.N = 10000
        assert max_len > initial_length
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        self.EPOCH = epoch
        self.INITIAL_LENGTH = initial_length
        self.MAX_LEN = max_len
        self.load_sequence_list()
        self.create_dataset()

    def generator(self):
        for i, sequence_points in enumerate(self.sequence_list):
            if len(sequence_points) <= self.INITIAL_LENGTH:
                continue
            x_image = np.zeros([self.L, self.L, self.INITIAL_LENGTH])
            x_coordinate = []
            y_image = np.zeros([self.L, self.L, self.MAX_LEN])
            y_coordinate = np.zeros([self.MAX_LEN, 2])
            for j, p in enumerate(sequence_points[:self.MAX_LEN]):
                if j < self.INITIAL_LENGTH:
                    if j > 0:
                        x_image[:, :, j] = x_image[:, :, j - 1]
                    x_image[p[1], p[0], j] = 1.0
                    x_coordinate.append(p)
                if j >= self.INITIAL_LENGTH:
                    y_image[p[1], p[0], j - self.INITIAL_LENGTH] = 1.0
                    y_coordinate[j - self.INITIAL_LENGTH] = p
            l = len(sequence_points) - self.INITIAL_LENGTH
            yield (x_image, y_image, x_coordinate, y_coordinate, l)

    def validation_generator(self):
        for i, sequence_points in enumerate(self.validation_sequence_list):
            if len(sequence_points) <= self.INITIAL_LENGTH:
                continue
            x_image = np.zeros([self.L, self.L, self.INITIAL_LENGTH])
            x_coordinate = []
            y_image = np.zeros([self.L, self.L, self.MAX_LEN])
            y_coordinate = np.zeros([self.MAX_LEN, 2])
            for j, p in enumerate(sequence_points[:self.MAX_LEN]):
                if j < self.INITIAL_LENGTH:
                    if j > 0:
                        x_image[:, :, j] = x_image[:, :, j - 1]
                    x_image[p[1], p[0], j] = 1.0
                    x_coordinate.append(p)
                if j >= self.INITIAL_LENGTH:
                    y_image[p[1], p[0], j - self.INITIAL_LENGTH] = 1.0
                    y_coordinate[j - self.INITIAL_LENGTH] = p
            l = len(sequence_points) - self.INITIAL_LENGTH
            yield (x_image, y_image, x_coordinate, y_coordinate, l)

    def create_dataset(self):
        self.dataset = tf.data.Dataset.from_generator(
            self.generator, (tf.float32, tf.float32, tf.float32, tf.float32, tf.int64),
            (tf.TensorShape([self.L, self.L, self.INITIAL_LENGTH]), tf.TensorShape([self.L, self.L, self.MAX_LEN]),
             tf.TensorShape([self.INITIAL_LENGTH, 2]), tf.TensorShape([self.MAX_LEN, 2]),
             tf.TensorShape([]))
        )
        self.dataset = self.dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=42).repeat()
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.BATCH_SIZE))
        self.dataset_iterator = self.dataset.make_one_shot_iterator()
        self.get_batch = self.dataset_iterator.get_next()
        if self.TYPE == 'train':
            self.validation_dataset = tf.data.Dataset.from_generator(
                self.validation_generator, (tf.float32, tf.float32, tf.float32, tf.float32, tf.int64),
                (tf.TensorShape([self.L, self.L, self.INITIAL_LENGTH]), tf.TensorShape([self.L, self.L, self.MAX_LEN]),
                 tf.TensorShape([self.INITIAL_LENGTH, 2]), tf.TensorShape([self.MAX_LEN, 2]),
                 tf.TensorShape([]))
            )
            self.validation_dataset = self.validation_dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=42).repeat()
            self.validation_dataset = self.validation_dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(self.BATCH_SIZE))

            self.validation_dataset_iterator = self.validation_dataset.make_one_shot_iterator()
            self.get_validation_batch = self.validation_dataset_iterator.get_next()

    def load_sequence_list(self):
        try:
            self.sequence_list = np.load(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"))
            if self.TYPE == 'train':
                self.validation_sequence_list = np.load(
                    os.path.join(self.DATASET_DIR + "cache/", "validation_points.npy"))
        except:
            self.sequence_list = []
            print("Creating Cache file for mnist dataset")
            for i in range(self.N):
                if i % 100 == 0:
                    print("{}/{}".format(i, self.N))
                file_name = self.DATASET_DIR + "sequences/{}img-{}-points.txt".format(self.TYPE, i)
                sequence_points = open(file_name, 'r').readlines()[1:]
                sequence_points = [tuple(map(lambda x: int(x) - 1, s[:-1].split(","))) for s in sequence_points]
                sequence_points = [s for s in sequence_points if s[0] > -1]
                self.sequence_list.append(sequence_points)
            if self.TYPE == 'train':
                np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"),
                        self.sequence_list[:-10000])
                np.save(os.path.join(self.DATASET_DIR + "cache/", "validation_points.npy"), self.sequence_list[-10000:])
                self.validation_sequence_list = self.sequence_list[-10000:]
                self.sequence_list = self.sequence_list[:-10000]
            else:
                np.save(os.path.join(self.DATASET_DIR + "cache/", self.TYPE + "_points.npy"), self.sequence_list)


class Oracle_Dataloader():
    DATASET_DIR = "Mnist_Saved_Models/Oracle/Oracle1/dataset.npy"
    BUFFER_SIZE = 8192
    BATCH_SIZE = 256
    TYPE = 'train'
    INITIAL_LENGTH = 10

    def __init__(self, type=TYPE, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, dataset_dir=DATASET_DIR):
        assert type == 'train' or type == 'test'
        self.DATASET_DIR = dataset_dir
        self.TYPE = type
        if type == 'train':
            self.N = 60000
        else:
            self.N = 10000

        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = buffer_size
        self.load_sequence_list()
        self.create_dataset()

    def load_sequence_list(self):
        sequence_list = np.load(self.DATASET_DIR)
        n = len(sequence_list)
        n_test = n // 5
        if self.TYPE == 'test':
            self.sequence_list = sequence_list[-n_test:]

        if self.TYPE == 'train':
            self.validation_sequence_list = sequence_list[:n_test]
            self.sequence_list = sequence_list[n_test:-n_test]

        self.MAX_LEN = len(sequence_list[0])

    def generator(self):
        for sequence_points in self.sequence_list:
            x = sequence_points.copy()
            x[1:] = x[:-1]
            x[0] = 0
            y = sequence_points
            l = len(sequence_points)
            yield (x, y, l)

    def validation_generator(self):
        for sequence_points in self.validation_sequence_list:
            x = sequence_points.copy()
            x[1:] = x[:-1]
            x[0] = 0
            y = sequence_points
            l = len(sequence_points)
            yield (x, y, l)

    def create_dataset(self):
        self.dataset = tf.data.Dataset.from_generator(
            self.generator, (tf.float32, tf.float32, tf.int32),
            (tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([]))
        )
        self.dataset = self.dataset.shuffle(buffer_size=self.BUFFER_SIZE, seed=42).repeat().batch(
            self.BATCH_SIZE)
        self.dataset_iterator = self.dataset.make_one_shot_iterator()
        self.get_batch = self.dataset_iterator.get_next()
        if self.TYPE == 'train':
            self.validation_dataset = tf.data.Dataset.from_generator(
                self.validation_generator, (tf.float32, tf.float32, tf.int32),
                (tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([self.MAX_LEN, 2]), tf.TensorShape([]))
            )
            self.validation_dataset = self.validation_dataset.shuffle(buffer_size=self.BUFFER_SIZE,
                                                                      seed=42).repeat().batch(self.BATCH_SIZE)

            self.validation_dataset_iterator = self.validation_dataset.make_one_shot_iterator()
            self.get_validation_batch = self.validation_dataset_iterator.get_next()


if __name__ == "__main__":
    pass
    # A = Mnist_Relative_Sequence_Dataloader('test', 8192, 10, 1, max_len=15)
    # B = Mnist_Image2Image_Dataloader('train', 8192, 10, 1)
    # C = Mnist_Absolute_Sequence_Dataloader('test', 8192, 10, 1, max_len=15)
    # D = Oracle_Dataloader('train', 8192, 4, "Mnist_Saved_Models/Oracle/Oracle1/dataset.npy")
    # E = Mnist_Temporal_Image2Image_Dataloader(batch_size=2, initial_length=3, max_len=10)
    # F = Mnist_Spline_Sequence_Dataloader(batch_size=20, initial_length=10, max_len=120)
    G = Mnist_Hybrid_Image2Image_Dataloader(batch_size=2, initial_length=10, max_len=110)

    ses = tf.InteractiveSession()
    # a = ses.run(A.get_batch)
    # b = ses.run(B.get_batch)
    # c = ses.run(C.get_batch)
    # d = ses.run(D.get_batch)
    # print(a[0].shape)
    # print(b[0].shape)
    # print(c[0].shape)
    # print(d[0][0][:5], d[1][0][:5], d[2][0])
    # e = ses.run(E.get_batch)
    # print(e[0].shape)
    # x = e[0][0]
    # y = e[1][0]
    # import matplotlib.pyplot as plt
    #
    # print(x[:, :, 0].shape)
    # plt.imshow(x[:, :, 3])
    # plt.figure()
    # plt.imshow(y[:, :, 1])
    # plt.show()

    # f = ses.run(F.get_validation_batch)
    # f1, f2, f3, f4 = f
    # i = 0
    # print(f3[i] - f2[i])
    #
    # show_positions_mus(f2[i][:f4[i]], f3[i][:f4[i]])

    g = ses.run(G.get_batch)
    print(g[0].shape)

    ses.close()
