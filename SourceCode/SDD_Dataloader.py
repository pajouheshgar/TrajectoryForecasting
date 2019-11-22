import tensorflow as tf
import numpy as np
from PIL import Image
from Utils import get_discrete_line
from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict

unstable_videos = [['nexus', 3], ['nexus', 4], ['nexus', 5], ['bookstore', 1], ['deathCircle', 3], ['hyang', 3]]
FPS = 30
PAST_LENGTH = 60
FUTURE_LENGTH = 120
N_SPLITS = 5


class Abosulute_Single_Object_Position_Dataloader:
    SDD_DATASET_DIR = "Dataset/SDD/"
    FPS = 30
    PAST_LENGTH = 60
    FUTURE_LENGTH = 120
    N_SPLITS = 5

    FUTURE_RESOLUTION = 30
    PAST_RESOLUTION = 60

    SHUFFLE_BUFFER_SIZE = 1024
    BATCH_SIZE = 128
    TEST_SPLIT_ID = 0
    NORMALIZE = True

    def __init__(self, future_resolution=FUTURE_RESOLUTION, past_resolution=PAST_RESOLUTION,
                 shuffle_buffer_size=SHUFFLE_BUFFER_SIZE, batch_size=BATCH_SIZE, test_split_id=TEST_SPLIT_ID,
                 sdd_dataset_dir=SDD_DATASET_DIR, normalize=NORMALIZE):
        self.SDD_DATASET_DIR = sdd_dataset_dir
        sdd_split_file = self.SDD_DATASET_DIR + "sdd_data_split.npy"
        sdd_split = dict(np.load(sdd_split_file).item())
        self.videos = sdd_split['videos']
        self.splits = sdd_split['splits']
        assert 30 % future_resolution == 0
        self.FUTURE_RESOLUTION = future_resolution
        assert 60 % past_resolution == 0
        self.PAST_RESOLUTION = past_resolution
        self.N_FUTURE_POINTS = FUTURE_LENGTH // self.FUTURE_RESOLUTION
        self.N_PAST_POINTS = PAST_LENGTH // self.PAST_RESOLUTION

        self.SHUFFLE_BUFFER_SIZE = shuffle_buffer_size
        self.BATCH_SIZE = batch_size
        self.TEST_SPLIT_ID = test_split_id

        self.NORMALIZE = normalize

        self.create_dataset()

    def load_object_dict(self, scene_name, video_id):
        object_dict = np.load(
            self.SDD_DATASET_DIR + "annotations/{}/video{}/object_dict.npy".format(scene_name, video_id)).item()
        return object_dict

    def load_frame_dict(self, scene_name, video_id):
        frame_dict = np.load(
            self.SDD_DATASET_DIR + "annotations/{}/video{}/frame_dict.npy".format(scene_name, video_id)).item()
        return frame_dict

    def load_reference_image(self, scene_name, video_id):
        reference_image = Image.open(
            self.SDD_DATASET_DIR + "annotations/" + scene_name + "/video{}/reference.jpg".format(video_id))
        return reference_image

    def unstable_generator(self):
        for scene_name, video_id in unstable_videos:
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.PAST_RESOLUTION)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.FUTURE_RESOLUTION)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0
                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    l_past = len(past_trajectory_points)
                    l_future = len(future_trajectory_points)
                    # yield (past_trajectory_points, future_trajectory_points, l_past, l_future, object_type)
                    yield (past_trajectory_points, future_trajectory_points, object_type, width, height)

    def split_generator0(self):
        split = self.splits[0]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.PAST_RESOLUTION)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.FUTURE_RESOLUTION)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0
                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    l_past = len(past_trajectory_points)
                    l_future = len(future_trajectory_points)
                    # yield (past_trajectory_points, future_trajectory_points, l_past, l_future, object_type)
                    yield (past_trajectory_points, future_trajectory_points, object_type, width, height)

    def split_generator1(self):
        split = self.splits[1]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.PAST_RESOLUTION)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.FUTURE_RESOLUTION)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0
                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    l_past = len(past_trajectory_points)
                    l_future = len(future_trajectory_points)
                    # yield (past_trajectory_points, future_trajectory_points, l_past, l_future, object_type)
                    yield (past_trajectory_points, future_trajectory_points, object_type, width, height)

    def split_generator2(self):
        split = self.splits[2]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.PAST_RESOLUTION)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.FUTURE_RESOLUTION)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0
                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    l_past = len(past_trajectory_points)
                    l_future = len(future_trajectory_points)
                    # yield (past_trajectory_points, future_trajectory_points, l_past, l_future, object_type)
                    yield (past_trajectory_points, future_trajectory_points, object_type, width, height)

    def split_generator3(self):
        split = self.splits[3]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.PAST_RESOLUTION)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.FUTURE_RESOLUTION)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0
                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    l_past = len(past_trajectory_points)
                    l_future = len(future_trajectory_points)
                    # yield (past_trajectory_points, future_trajectory_points, l_past, l_future, object_type)
                    yield (past_trajectory_points, future_trajectory_points, object_type, width, height)

    def split_generator4(self):
        split = self.splits[4]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.PAST_RESOLUTION)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.FUTURE_RESOLUTION)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0
                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    l_past = len(past_trajectory_points)
                    l_future = len(future_trajectory_points)
                    # yield (past_trajectory_points, future_trajectory_points, l_past, l_future, object_type)
                    yield (past_trajectory_points, future_trajectory_points, object_type, width, height)

    def create_dataset(self):
        generator_list = [self.split_generator0, self.split_generator1, self.split_generator2, self.split_generator3,
                          self.split_generator4]

        def create_dataset_from_generator(generator):
            dataset = tf.data.Dataset.from_generator(
                generator, (tf.float32, tf.float32, tf.int64, tf.int64, tf.int64),
                (tf.TensorShape([self.N_PAST_POINTS, 2]), tf.TensorShape([self.N_FUTURE_POINTS, 2]),
                 tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])))
            return dataset

        self.train_dataset_list = [create_dataset_from_generator(generator_list[index]) for index in
                                   range(N_SPLITS) if index != self.TEST_SPLIT_ID]
        # self.test_dataset = create_dataset_from_generator(self.generator_list[self.TEST_SPLIT_ID])
        self.test_dataset = create_dataset_from_generator(generator_list[self.TEST_SPLIT_ID])
        self.unstable_dataset = create_dataset_from_generator(self.unstable_generator)
        self.train_dataset = self.train_dataset_list[0]
        for train_dataset in self.train_dataset_list[1:]:
            self.train_dataset = self.train_dataset.concatenate(train_dataset)

        self.whole_dataset = self.train_dataset.concatenate(self.test_dataset)

        def shuffle_repeat_batch_get(dataset, epoch=-1):
            return dataset.shuffle(buffer_size=self.SHUFFLE_BUFFER_SIZE, seed=42).repeat(epoch).batch(
                self.BATCH_SIZE).make_one_shot_iterator().get_next()

        self.get_train_batch = shuffle_repeat_batch_get(self.train_dataset)
        self.get_test_batch = shuffle_repeat_batch_get(self.test_dataset, epoch=1)
        self.get_unstable_batch = shuffle_repeat_batch_get(self.unstable_dataset)
        self.get_whole_batch = shuffle_repeat_batch_get(self.whole_dataset, epoch=1)


class Abosulute_Single_Object_Position_REF_IMG_Dataloader:
    SDD_DATASET_DIR = "Dataset/SDD/"
    FPS = 30
    PAST_LENGTH = 60
    FUTURE_LENGTH = 120
    N_SPLITS = 5

    FUTURE_RESOLUTION = 30
    PAST_RESOLUTION = 60

    SHUFFLE_BUFFER_SIZE = 1024
    BATCH_SIZE = 128
    TEST_SPLIT_ID = 0
    NORMALIZE = True

    REF_IMAGE_WIDTH_HEIGHT = 512

    def __init__(self, future_resolution=FUTURE_RESOLUTION, past_resolution=PAST_RESOLUTION,
                 shuffle_buffer_size=SHUFFLE_BUFFER_SIZE, batch_size=BATCH_SIZE, test_split_id=TEST_SPLIT_ID,
                 sdd_dataset_dir=SDD_DATASET_DIR, normalize=NORMALIZE):
        self.SDD_DATASET_DIR = sdd_dataset_dir
        sdd_split_file = self.SDD_DATASET_DIR + "sdd_data_split.npy"
        sdd_split = dict(np.load(sdd_split_file).item())
        self.videos = sdd_split['videos']
        self.splits = sdd_split['splits']
        assert 30 % future_resolution == 0
        self.FUTURE_RESOLUTION = future_resolution
        assert 60 % past_resolution == 0
        self.PAST_RESOLUTION = past_resolution
        self.N_FUTURE_POINTS = FUTURE_LENGTH // self.FUTURE_RESOLUTION
        self.N_PAST_POINTS = PAST_LENGTH // self.PAST_RESOLUTION

        self.SHUFFLE_BUFFER_SIZE = shuffle_buffer_size
        self.BATCH_SIZE = batch_size
        self.TEST_SPLIT_ID = test_split_id

        self.NORMALIZE = normalize

        self.create_dataset()

    def load_object_dict(self, scene_name, video_id):
        object_dict = np.load(
            self.SDD_DATASET_DIR + "annotations/{}/video{}/object_dict.npy".format(scene_name, video_id)).item()
        return object_dict

    def load_frame_dict(self, scene_name, video_id):
        frame_dict = np.load(
            self.SDD_DATASET_DIR + "annotations/{}/video{}/frame_dict.npy".format(scene_name, video_id)).item()
        return frame_dict

    def load_reference_image(self, scene_name, video_id):
        reference_image = Image.open(
            self.SDD_DATASET_DIR + "annotations/" + scene_name + "/video{}/reference.jpg".format(video_id))
        reference_image = reference_image.convert('RGB')
        return reference_image

    def unstable_generator(self):
        for scene_name, video_id in unstable_videos:
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.PAST_RESOLUTION)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.FUTURE_RESOLUTION)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0
                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    l_past = len(past_trajectory_points)
                    l_future = len(future_trajectory_points)
                    # yield (past_trajectory_points, future_trajectory_points, l_past, l_future, object_type)
                    yield (past_trajectory_points, future_trajectory_points, object_type, width, height,
                           resized_reference_image)

    def split_generator0(self):
        split = self.splits[0]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.PAST_RESOLUTION)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.FUTURE_RESOLUTION)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0
                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    l_past = len(past_trajectory_points)
                    l_future = len(future_trajectory_points)
                    # yield (past_trajectory_points, future_trajectory_points, l_past, l_future, object_type)
                    yield (past_trajectory_points, future_trajectory_points, object_type, width, height,
                           resized_reference_image)

    def split_generator1(self):
        split = self.splits[1]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.PAST_RESOLUTION)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.FUTURE_RESOLUTION)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0
                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    l_past = len(past_trajectory_points)
                    l_future = len(future_trajectory_points)
                    # yield (past_trajectory_points, future_trajectory_points, l_past, l_future, object_type)
                    yield (past_trajectory_points, future_trajectory_points, object_type, width, height,
                           resized_reference_image)

    def split_generator2(self):
        split = self.splits[2]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.PAST_RESOLUTION)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.FUTURE_RESOLUTION)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0
                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    l_past = len(past_trajectory_points)
                    l_future = len(future_trajectory_points)
                    # yield (past_trajectory_points, future_trajectory_points, l_past, l_future, object_type)
                    yield (past_trajectory_points, future_trajectory_points, object_type, width, height,
                           resized_reference_image)

    def split_generator3(self):
        split = self.splits[3]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.PAST_RESOLUTION)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.FUTURE_RESOLUTION)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0
                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    l_past = len(past_trajectory_points)
                    l_future = len(future_trajectory_points)
                    # yield (past_trajectory_points, future_trajectory_points, l_past, l_future, object_type)
                    yield (past_trajectory_points, future_trajectory_points, object_type, width, height,
                           resized_reference_image)

    def split_generator4(self):
        split = self.splits[4]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.PAST_RESOLUTION)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.FUTURE_RESOLUTION)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0
                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    l_past = len(past_trajectory_points)
                    l_future = len(future_trajectory_points)
                    # yield (past_trajectory_points, future_trajectory_points, l_past, l_future, object_type)
                    yield (past_trajectory_points, future_trajectory_points, object_type, width, height,
                           resized_reference_image)

    def create_dataset(self):
        generator_list = [self.split_generator0, self.split_generator1, self.split_generator2, self.split_generator3,
                          self.split_generator4]

        def create_dataset_from_generator(generator):
            dataset = tf.data.Dataset.from_generator(
                generator, (tf.float32, tf.float32, tf.int64, tf.int64, tf.int64, tf.float32),
                (tf.TensorShape([self.N_PAST_POINTS, 2]), tf.TensorShape([self.N_FUTURE_POINTS, 2]),
                 tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]),
                 tf.TensorShape([self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT, 3])))
            return dataset

        self.train_dataset_list = [create_dataset_from_generator(generator_list[index]) for index in
                                   range(N_SPLITS) if index != self.TEST_SPLIT_ID]
        # self.test_dataset = create_dataset_from_generator(self.generator_list[self.TEST_SPLIT_ID])
        self.test_dataset = create_dataset_from_generator(generator_list[self.TEST_SPLIT_ID])
        self.unstable_dataset = create_dataset_from_generator(self.unstable_generator)
        self.train_dataset = self.train_dataset_list[0]
        for train_dataset in self.train_dataset_list[1:]:
            self.train_dataset = self.train_dataset.concatenate(train_dataset)

        def shuffle_repeat_batch_get(dataset, epoch=-1):
            return dataset.shuffle(buffer_size=self.SHUFFLE_BUFFER_SIZE, seed=42).repeat(epoch).batch(
                self.BATCH_SIZE).make_one_shot_iterator().get_next()

        self.get_train_batch = shuffle_repeat_batch_get(self.train_dataset)
        self.get_test_batch = shuffle_repeat_batch_get(self.test_dataset, epoch=1)
        self.get_unstable_batch = shuffle_repeat_batch_get(self.unstable_dataset)


class IMG_Single_Object_Position_REF_IMG_Dataloader:
    SDD_DATASET_DIR = "Dataset/SDD/"
    FPS = 30
    PAST_LENGTH = 60
    FUTURE_LENGTH = 120
    N_SPLITS = 5

    STRIDE = 30

    SHUFFLE_BUFFER_SIZE = 32
    BATCH_SIZE = 128
    TEST_SPLIT_ID = 0
    NORMALIZE = True
    BLUR_STD = 1.0

    SIZE = 128
    REF_IMAGE_WIDTH_HEIGHT = SIZE
    TRAJECTORY_IMAGE_WIDTH_HEIGHT = SIZE

    def __init__(self, stride=STRIDE, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE, batch_size=BATCH_SIZE,
                 test_split_id=TEST_SPLIT_ID,
                 sdd_dataset_dir=SDD_DATASET_DIR, normalize=NORMALIZE, size=SIZE, blur_std=BLUR_STD):
        self.SDD_DATASET_DIR = sdd_dataset_dir
        sdd_split_file = self.SDD_DATASET_DIR + "sdd_data_split.npy"
        sdd_split = dict(np.load(sdd_split_file).item())
        self.videos = sdd_split['videos']
        self.splits = sdd_split['splits']
        assert 30 % stride == 0
        self.STRIDE = stride
        self.N_FUTURE_POINTS = FUTURE_LENGTH // self.STRIDE
        self.N_PAST_POINTS = PAST_LENGTH // self.STRIDE

        self.SHUFFLE_BUFFER_SIZE = shuffle_buffer_size
        self.BATCH_SIZE = batch_size
        self.TEST_SPLIT_ID = test_split_id

        self.NORMALIZE = normalize

        self.BLUR_STD = blur_std
        self.SIZE = size
        self.REF_IMAGE_WIDTH_HEIGHT = self.SIZE
        self.TRAJECTORY_IMAGE_WIDTH_HEIGHT = self.SIZE

        self.create_dataset()

    def load_object_dict(self, scene_name, video_id):
        object_dict = np.load(
            self.SDD_DATASET_DIR + "annotations/{}/video{}/object_dict.npy".format(scene_name, video_id)).item()
        return object_dict

    def load_frame_dict(self, scene_name, video_id):
        frame_dict = np.load(
            self.SDD_DATASET_DIR + "annotations/{}/video{}/frame_dict.npy".format(scene_name, video_id)).item()
        return frame_dict

    def load_reference_image(self, scene_name, video_id):
        reference_image = Image.open(
            self.SDD_DATASET_DIR + "annotations/" + scene_name + "/video{}/reference.jpg".format(video_id))
        reference_image = reference_image.convert('RGB')
        return reference_image

    def unstable_generator(self):
        for scene_name, video_id in unstable_videos:
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.STRIDE)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.STRIDE)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0

                    past_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS])
                    future_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])
                    future_points_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])

                    prev_point = None
                    for i, p in enumerate(past_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                            past_trajectory_temporal_images[:, :, i] = past_trajectory_temporal_images[:, :, i - 1]
                            for p1, p2 in line:
                                past_trajectory_temporal_images[p2, p1, i] = 1.0
                        if i == 0:
                            # past_trajectory_temporal_images[x_scaled_discretized, y_scaled_discretized, i] = 1.0
                            past_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        prev_point = (x_scaled_discretized, y_scaled_discretized)

                    last_past_point = prev_point
                    future_trajectory_temporal_images[:, :, 0] = past_trajectory_temporal_images[:, :, -1]
                    for i, p in enumerate(future_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            future_trajectory_temporal_images[:, :, i] = future_trajectory_temporal_images[:, :, i - 1]

                        line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                        for p1, p2 in line:
                            future_trajectory_temporal_images[p2, p1, i] = 1.0
                        future_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0

                        prev_point = (x_scaled_discretized, y_scaled_discretized)
                        future_points_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        future_points_temporal_images[:, :, i] = gaussian_filter(
                            future_points_temporal_images[:, :, i], sigma=self.BLUR_STD)

                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    yield (
                        past_trajectory_temporal_images,
                        future_trajectory_temporal_images,
                        future_points_temporal_images,
                        future_trajectory_points,
                        object_type, width, height,
                        resized_reference_image,
                        last_past_point
                    )

    def split_generator0(self):
        split = self.splits[0]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.STRIDE)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.STRIDE)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0

                    past_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS])
                    future_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])
                    future_points_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])

                    prev_point = None
                    for i, p in enumerate(past_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                            past_trajectory_temporal_images[:, :, i] = past_trajectory_temporal_images[:, :, i - 1]
                            for p1, p2 in line:
                                past_trajectory_temporal_images[p2, p1, i] = 1.0
                        if i == 0:
                            # past_trajectory_temporal_images[x_scaled_discretized, y_scaled_discretized, i] = 1.0
                            past_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        prev_point = (x_scaled_discretized, y_scaled_discretized)

                    last_past_point = prev_point
                    future_trajectory_temporal_images[:, :, 0] = past_trajectory_temporal_images[:, :, -1]
                    for i, p in enumerate(future_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            future_trajectory_temporal_images[:, :, i] = future_trajectory_temporal_images[:, :, i - 1]

                        line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                        for p1, p2 in line:
                            future_trajectory_temporal_images[p2, p1, i] = 1.0
                        future_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0

                        prev_point = (x_scaled_discretized, y_scaled_discretized)
                        future_points_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        future_points_temporal_images[:, :, i] = gaussian_filter(
                            future_points_temporal_images[:, :, i], sigma=self.BLUR_STD)

                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    yield (
                        past_trajectory_temporal_images,
                        future_trajectory_temporal_images,
                        future_points_temporal_images,
                        future_trajectory_points,
                        object_type, width, height,
                        resized_reference_image,
                        last_past_point
                    )

    def split_generator1(self):
        split = self.splits[1]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.STRIDE)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.STRIDE)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0

                    past_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS])
                    future_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])
                    future_points_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])

                    prev_point = None
                    for i, p in enumerate(past_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                            past_trajectory_temporal_images[:, :, i] = past_trajectory_temporal_images[:, :, i - 1]
                            for p1, p2 in line:
                                past_trajectory_temporal_images[p2, p1, i] = 1.0
                        if i == 0:
                            # past_trajectory_temporal_images[x_scaled_discretized, y_scaled_discretized, i] = 1.0
                            past_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        prev_point = (x_scaled_discretized, y_scaled_discretized)

                    last_past_point = prev_point
                    future_trajectory_temporal_images[:, :, 0] = past_trajectory_temporal_images[:, :, -1]
                    for i, p in enumerate(future_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            future_trajectory_temporal_images[:, :, i] = future_trajectory_temporal_images[:, :, i - 1]

                        line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                        for p1, p2 in line:
                            future_trajectory_temporal_images[p2, p1, i] = 1.0
                        future_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0

                        prev_point = (x_scaled_discretized, y_scaled_discretized)
                        future_points_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        future_points_temporal_images[:, :, i] = gaussian_filter(
                            future_points_temporal_images[:, :, i], sigma=self.BLUR_STD)

                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    yield (
                        past_trajectory_temporal_images,
                        future_trajectory_temporal_images,
                        future_points_temporal_images,
                        future_trajectory_points,
                        object_type, width, height,
                        resized_reference_image,
                        last_past_point
                    )

    def split_generator2(self):
        split = self.splits[2]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.STRIDE)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.STRIDE)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0

                    past_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS])
                    future_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])
                    future_points_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])

                    prev_point = None
                    for i, p in enumerate(past_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                            past_trajectory_temporal_images[:, :, i] = past_trajectory_temporal_images[:, :, i - 1]
                            for p1, p2 in line:
                                past_trajectory_temporal_images[p2, p1, i] = 1.0
                        if i == 0:
                            # past_trajectory_temporal_images[x_scaled_discretized, y_scaled_discretized, i] = 1.0
                            past_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        prev_point = (x_scaled_discretized, y_scaled_discretized)

                    last_past_point = prev_point
                    future_trajectory_temporal_images[:, :, 0] = past_trajectory_temporal_images[:, :, -1]
                    for i, p in enumerate(future_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            future_trajectory_temporal_images[:, :, i] = future_trajectory_temporal_images[:, :, i - 1]

                        line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                        for p1, p2 in line:
                            future_trajectory_temporal_images[p2, p1, i] = 1.0
                        future_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0

                        prev_point = (x_scaled_discretized, y_scaled_discretized)
                        future_points_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        future_points_temporal_images[:, :, i] = gaussian_filter(
                            future_points_temporal_images[:, :, i], sigma=self.BLUR_STD)

                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    yield (
                        past_trajectory_temporal_images,
                        future_trajectory_temporal_images,
                        future_points_temporal_images,
                        future_trajectory_points,
                        object_type, width, height,
                        resized_reference_image,
                        last_past_point
                    )

    def split_generator3(self):
        split = self.splits[3]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.STRIDE)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.STRIDE)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0

                    past_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS])
                    future_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])
                    future_points_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])

                    prev_point = None
                    for i, p in enumerate(past_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                            past_trajectory_temporal_images[:, :, i] = past_trajectory_temporal_images[:, :, i - 1]
                            for p1, p2 in line:
                                past_trajectory_temporal_images[p2, p1, i] = 1.0
                        if i == 0:
                            # past_trajectory_temporal_images[x_scaled_discretized, y_scaled_discretized, i] = 1.0
                            past_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        prev_point = (x_scaled_discretized, y_scaled_discretized)

                    last_past_point = prev_point
                    future_trajectory_temporal_images[:, :, 0] = past_trajectory_temporal_images[:, :, -1]
                    for i, p in enumerate(future_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            future_trajectory_temporal_images[:, :, i] = future_trajectory_temporal_images[:, :, i - 1]

                        line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                        for p1, p2 in line:
                            future_trajectory_temporal_images[p2, p1, i] = 1.0
                        future_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0

                        prev_point = (x_scaled_discretized, y_scaled_discretized)
                        future_points_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        future_points_temporal_images[:, :, i] = gaussian_filter(
                            future_points_temporal_images[:, :, i], sigma=self.BLUR_STD)

                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    yield (
                        past_trajectory_temporal_images,
                        future_trajectory_temporal_images,
                        future_points_temporal_images,
                        future_trajectory_points,
                        object_type, width, height,
                        resized_reference_image,
                        last_past_point
                    )

    def split_generator4(self):
        split = self.splits[4]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:
                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.STRIDE)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.STRIDE)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0

                    past_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS])
                    future_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])
                    future_points_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])

                    prev_point = None
                    for i, p in enumerate(past_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                            past_trajectory_temporal_images[:, :, i] = past_trajectory_temporal_images[:, :, i - 1]
                            for p1, p2 in line:
                                past_trajectory_temporal_images[p2, p1, i] = 1.0
                        if i == 0:
                            # past_trajectory_temporal_images[x_scaled_discretized, y_scaled_discretized, i] = 1.0
                            past_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        prev_point = (x_scaled_discretized, y_scaled_discretized)

                    last_past_point = prev_point
                    future_trajectory_temporal_images[:, :, 0] = past_trajectory_temporal_images[:, :, -1]
                    for i, p in enumerate(future_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            future_trajectory_temporal_images[:, :, i] = future_trajectory_temporal_images[:, :, i - 1]

                        line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                        for p1, p2 in line:
                            future_trajectory_temporal_images[p2, p1, i] = 1.0
                        future_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0

                        prev_point = (x_scaled_discretized, y_scaled_discretized)
                        future_points_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        future_points_temporal_images[:, :, i] = gaussian_filter(
                            future_points_temporal_images[:, :, i], sigma=self.BLUR_STD)

                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    yield (
                        past_trajectory_temporal_images,
                        future_trajectory_temporal_images,
                        future_points_temporal_images,
                        future_trajectory_points,
                        object_type, width, height,
                        resized_reference_image,
                        last_past_point
                    )

    def create_dataset(self):
        generator_list = [self.split_generator0, self.split_generator1, self.split_generator2, self.split_generator3,
                          self.split_generator4]

        def create_dataset_from_generator(generator):
            dataset = tf.data.Dataset.from_generator(
                generator,
                (tf.float32, tf.float32, tf.float32, tf.float32, tf.int64, tf.int64, tf.int64, tf.float32, tf.int64),
                (tf.TensorShape(
                    [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS]),
                 tf.TensorShape(
                     [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS]),
                 tf.TensorShape(
                     [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS]),
                 tf.TensorShape(
                     [self.N_FUTURE_POINTS, 2]),
                 tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]),
                 tf.TensorShape([self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT, 3]),
                 tf.TensorShape([2])
                ))
            return dataset

        self.train_dataset_list = [create_dataset_from_generator(generator_list[index]) for index in
                                   range(N_SPLITS) if index != self.TEST_SPLIT_ID]
        # self.test_dataset = create_dataset_from_generator(self.generator_list[self.TEST_SPLIT_ID])
        self.test_dataset = create_dataset_from_generator(generator_list[self.TEST_SPLIT_ID])
        self.unstable_dataset = create_dataset_from_generator(self.unstable_generator)
        self.train_dataset = self.train_dataset_list[0]
        for train_dataset in self.train_dataset_list[1:]:
            self.train_dataset = self.train_dataset.concatenate(train_dataset)

        def shuffle_repeat_batch_get(dataset, epoch=-1):
            return dataset.shuffle(buffer_size=self.SHUFFLE_BUFFER_SIZE, seed=42).repeat(epoch).batch(
                self.BATCH_SIZE).make_one_shot_iterator().get_next()

        self.get_train_batch = shuffle_repeat_batch_get(self.train_dataset)
        self.get_test_batch = shuffle_repeat_batch_get(self.test_dataset, epoch=1)
        self.get_unstable_batch = shuffle_repeat_batch_get(self.unstable_dataset)

    def show_ref_image_trajectories(self, scene_name, video_id):
        ref_image = np.array(self.load_reference_image(scene_name, video_id))
        plt.imshow(ref_image)
        frame_dict = dict(self.load_frame_dict(scene_name, video_id))
        point_list = []
        point_type = []
        for frame in frame_dict.keys():
            print(frame)
            # if frame > 100:
            #     break
            for p in frame_dict[frame]:
                if p[-3] == 1 or p[-4] == 1:
                    continue
                x, y = (p[1] + p[3]) / 2, (p[2] + p[4]) / 2
                point_list.append((x, y))
                point_type.append(p[9])

        point_list = np.array(point_list)
        plt.scatter(point_list[:, 0], point_list[:, 1], alpha=0.01, s=1.0)
        plt.show()
        print(ref_image.shape)
        pass


class IMG_Multi_Object_Position_REF_IMG_Dataloader:
    SDD_DATASET_DIR = "Dataset/SDD/"
    FPS = 30
    PAST_LENGTH = 60
    FUTURE_LENGTH = 120
    N_SPLITS = 5

    STRIDE = 30

    SHUFFLE_BUFFER_SIZE = 32
    BATCH_SIZE = 128
    TEST_SPLIT_ID = 0
    NORMALIZE = True
    BLUR_STD = 1.0

    SIZE = 128
    REF_IMAGE_WIDTH_HEIGHT = SIZE
    TRAJECTORY_IMAGE_WIDTH_HEIGHT = SIZE

    def __init__(self, stride=STRIDE, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE, batch_size=BATCH_SIZE,
                 test_split_id=TEST_SPLIT_ID,
                 sdd_dataset_dir=SDD_DATASET_DIR, normalize=NORMALIZE, size=SIZE, blur_std=BLUR_STD):
        self.SDD_DATASET_DIR = sdd_dataset_dir
        sdd_split_file = self.SDD_DATASET_DIR + "sdd_data_split.npy"
        sdd_split = dict(np.load(sdd_split_file).item())
        self.videos = sdd_split['videos']
        self.splits = sdd_split['splits']
        assert 30 % stride == 0
        self.STRIDE = stride
        self.N_FUTURE_POINTS = FUTURE_LENGTH // self.STRIDE
        self.N_PAST_POINTS = PAST_LENGTH // self.STRIDE

        self.SHUFFLE_BUFFER_SIZE = shuffle_buffer_size
        self.BATCH_SIZE = batch_size
        self.TEST_SPLIT_ID = test_split_id

        self.NORMALIZE = normalize

        self.BLUR_STD = blur_std
        self.SIZE = size
        self.REF_IMAGE_WIDTH_HEIGHT = self.SIZE
        self.TRAJECTORY_IMAGE_WIDTH_HEIGHT = self.SIZE

        self.create_dataset()

    def load_object_dict(self, scene_name, video_id):
        object_dict = np.load(
            self.SDD_DATASET_DIR + "annotations/{}/video{}/object_dict.npy".format(scene_name, video_id)).item()
        return object_dict

    def load_frame_dict(self, scene_name, video_id):
        frame_dict = np.load(
            self.SDD_DATASET_DIR + "annotations/{}/video{}/frame_dict.npy".format(scene_name, video_id)).item()
        return frame_dict

    def load_reference_image(self, scene_name, video_id):
        reference_image = Image.open(
            self.SDD_DATASET_DIR + "annotations/" + scene_name + "/video{}/reference.jpg".format(video_id))
        reference_image = reference_image.convert('RGB')
        return reference_image

    def get_other_objects_images(self, frame_dict, ref_object_id, frames_list, width, height):
        objects_in_frames_dict = defaultdict(lambda: [])
        for frame in frames_list:
            for object_point in frame_dict[frame]:
                object_id = object_point[0]
                objects_in_frames_dict[object_id].append(object_point)

        n_frames = len(frames_list)

        other_objects_temporal_images = np.zeros(
            [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, n_frames])
        for object_id in objects_in_frames_dict.keys():
            if object_id == ref_object_id:
                continue

            prev_point = None
            for i, object_point in enumerate(objects_in_frames_dict[object_id]):
                # check to see if object is lost
                if object_point[6] == 1:
                    break
                x = (object_point[1] + object_point[3]) / 2
                y = (object_point[2] + object_point[4]) / 2
                x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                frame_number = object_point[5]
                frame_index = frames_list.index(frame_number)

                if i == 0 or prev_point == None:
                    other_objects_temporal_images[y_scaled_discretized, x_scaled_discretized, frame_index:] = 1.0
                elif i > 0:
                    line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                    for p1, p2 in line:
                        other_objects_temporal_images[p2, p1, frame_index:] = 1.0
                prev_point = (x_scaled_discretized, y_scaled_discretized)

        return other_objects_temporal_images

    def unstable_generator(self):
        for scene_name, video_id in unstable_videos:
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            frame_dict = dict(self.load_frame_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:

                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.STRIDE)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.STRIDE)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0

                    past_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS])
                    future_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])
                    future_points_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])

                    prev_point = None
                    for i, p in enumerate(past_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                            past_trajectory_temporal_images[:, :, i] = past_trajectory_temporal_images[:, :, i - 1]
                            for p1, p2 in line:
                                past_trajectory_temporal_images[p2, p1, i] = 1.0
                        if i == 0:
                            # past_trajectory_temporal_images[x_scaled_discretized, y_scaled_discretized, i] = 1.0
                            past_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        prev_point = (x_scaled_discretized, y_scaled_discretized)

                    last_past_point = prev_point
                    future_trajectory_temporal_images[:, :, 0] = past_trajectory_temporal_images[:, :, -1]
                    for i, p in enumerate(future_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            future_trajectory_temporal_images[:, :, i] = future_trajectory_temporal_images[:, :, i - 1]

                        line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                        for p1, p2 in line:
                            future_trajectory_temporal_images[p2, p1, i] = 1.0
                        future_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0

                        prev_point = (x_scaled_discretized, y_scaled_discretized)
                        future_points_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        future_points_temporal_images[:, :, i] = gaussian_filter(
                            future_points_temporal_images[:, :, i], sigma=self.BLUR_STD)

                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    past_frames_list = past_trajectory_selected[:, 4]
                    future_frames_list = future_trajectory_selected[:, 4]
                    past_future_frames_list = list(past_frames_list) + list(future_frames_list)
                    other_objects_temporal_images = self.get_other_objects_images(frame_dict, obj,
                                                                                  past_future_frames_list, width,
                                                                                  height)

                    other_objects_past_temporal_images = other_objects_temporal_images[:, :, :self.N_PAST_POINTS]
                    other_objects_future_temporal_images = other_objects_temporal_images[:, :, self.N_PAST_POINTS:]

                    yield (
                        past_trajectory_temporal_images,
                        future_trajectory_temporal_images,
                        future_points_temporal_images,
                        future_trajectory_points,
                        object_type, width, height,
                        resized_reference_image,
                        last_past_point,
                        other_objects_past_temporal_images,
                        other_objects_future_temporal_images
                    )

    def split_generator0(self):
        split = self.splits[0]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            frame_dict = dict(self.load_frame_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:

                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.STRIDE)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.STRIDE)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0

                    past_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS])
                    future_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])
                    future_points_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])

                    prev_point = None
                    for i, p in enumerate(past_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                            past_trajectory_temporal_images[:, :, i] = past_trajectory_temporal_images[:, :, i - 1]
                            for p1, p2 in line:
                                past_trajectory_temporal_images[p2, p1, i] = 1.0
                        if i == 0:
                            # past_trajectory_temporal_images[x_scaled_discretized, y_scaled_discretized, i] = 1.0
                            past_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        prev_point = (x_scaled_discretized, y_scaled_discretized)

                    last_past_point = prev_point
                    future_trajectory_temporal_images[:, :, 0] = past_trajectory_temporal_images[:, :, -1]
                    for i, p in enumerate(future_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            future_trajectory_temporal_images[:, :, i] = future_trajectory_temporal_images[:, :, i - 1]

                        line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                        for p1, p2 in line:
                            future_trajectory_temporal_images[p2, p1, i] = 1.0
                        future_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0

                        prev_point = (x_scaled_discretized, y_scaled_discretized)
                        future_points_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        future_points_temporal_images[:, :, i] = gaussian_filter(
                            future_points_temporal_images[:, :, i], sigma=self.BLUR_STD)

                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    past_frames_list = past_trajectory_selected[:, 4]
                    future_frames_list = future_trajectory_selected[:, 4]
                    past_future_frames_list = list(past_frames_list) + list(future_frames_list)
                    other_objects_temporal_images = self.get_other_objects_images(frame_dict, obj,
                                                                                  past_future_frames_list, width,
                                                                                  height)

                    other_objects_past_temporal_images = other_objects_temporal_images[:, :, :self.N_PAST_POINTS]
                    other_objects_future_temporal_images = other_objects_temporal_images[:, :, self.N_PAST_POINTS:]

                    yield (
                        past_trajectory_temporal_images,
                        future_trajectory_temporal_images,
                        future_points_temporal_images,
                        future_trajectory_points,
                        object_type, width, height,
                        resized_reference_image,
                        last_past_point,
                        other_objects_past_temporal_images,
                        other_objects_future_temporal_images
                    )

    def split_generator1(self):
        split = self.splits[1]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            frame_dict = dict(self.load_frame_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:

                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.STRIDE)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.STRIDE)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0

                    past_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS])
                    future_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])
                    future_points_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])

                    prev_point = None
                    for i, p in enumerate(past_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                            past_trajectory_temporal_images[:, :, i] = past_trajectory_temporal_images[:, :, i - 1]
                            for p1, p2 in line:
                                past_trajectory_temporal_images[p2, p1, i] = 1.0
                        if i == 0:
                            # past_trajectory_temporal_images[x_scaled_discretized, y_scaled_discretized, i] = 1.0
                            past_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        prev_point = (x_scaled_discretized, y_scaled_discretized)

                    last_past_point = prev_point
                    future_trajectory_temporal_images[:, :, 0] = past_trajectory_temporal_images[:, :, -1]
                    for i, p in enumerate(future_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            future_trajectory_temporal_images[:, :, i] = future_trajectory_temporal_images[:, :, i - 1]

                        line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                        for p1, p2 in line:
                            future_trajectory_temporal_images[p2, p1, i] = 1.0
                        future_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0

                        prev_point = (x_scaled_discretized, y_scaled_discretized)
                        future_points_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        future_points_temporal_images[:, :, i] = gaussian_filter(
                            future_points_temporal_images[:, :, i], sigma=self.BLUR_STD)

                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    past_frames_list = past_trajectory_selected[:, 4]
                    future_frames_list = future_trajectory_selected[:, 4]
                    past_future_frames_list = list(past_frames_list) + list(future_frames_list)
                    other_objects_temporal_images = self.get_other_objects_images(frame_dict, obj,
                                                                                  past_future_frames_list, width,
                                                                                  height)

                    other_objects_past_temporal_images = other_objects_temporal_images[:, :, :self.N_PAST_POINTS]
                    other_objects_future_temporal_images = other_objects_temporal_images[:, :, self.N_PAST_POINTS:]

                    yield (
                        past_trajectory_temporal_images,
                        future_trajectory_temporal_images,
                        future_points_temporal_images,
                        future_trajectory_points,
                        object_type, width, height,
                        resized_reference_image,
                        last_past_point,
                        other_objects_past_temporal_images,
                        other_objects_future_temporal_images
                    )

    def split_generator2(self):
        split = self.splits[2]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            frame_dict = dict(self.load_frame_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:

                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.STRIDE)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.STRIDE)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0

                    past_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS])
                    future_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])
                    future_points_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])

                    prev_point = None
                    for i, p in enumerate(past_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                            past_trajectory_temporal_images[:, :, i] = past_trajectory_temporal_images[:, :, i - 1]
                            for p1, p2 in line:
                                past_trajectory_temporal_images[p2, p1, i] = 1.0
                        if i == 0:
                            # past_trajectory_temporal_images[x_scaled_discretized, y_scaled_discretized, i] = 1.0
                            past_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        prev_point = (x_scaled_discretized, y_scaled_discretized)

                    last_past_point = prev_point
                    future_trajectory_temporal_images[:, :, 0] = past_trajectory_temporal_images[:, :, -1]
                    for i, p in enumerate(future_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            future_trajectory_temporal_images[:, :, i] = future_trajectory_temporal_images[:, :, i - 1]

                        line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                        for p1, p2 in line:
                            future_trajectory_temporal_images[p2, p1, i] = 1.0
                        future_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0

                        prev_point = (x_scaled_discretized, y_scaled_discretized)
                        future_points_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        future_points_temporal_images[:, :, i] = gaussian_filter(
                            future_points_temporal_images[:, :, i], sigma=self.BLUR_STD)

                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    past_frames_list = past_trajectory_selected[:, 4]
                    future_frames_list = future_trajectory_selected[:, 4]
                    past_future_frames_list = list(past_frames_list) + list(future_frames_list)
                    other_objects_temporal_images = self.get_other_objects_images(frame_dict, obj,
                                                                                  past_future_frames_list, width,
                                                                                  height)

                    other_objects_past_temporal_images = other_objects_temporal_images[:, :, :self.N_PAST_POINTS]
                    other_objects_future_temporal_images = other_objects_temporal_images[:, :, self.N_PAST_POINTS:]

                    yield (
                        past_trajectory_temporal_images,
                        future_trajectory_temporal_images,
                        future_points_temporal_images,
                        future_trajectory_points,
                        object_type, width, height,
                        resized_reference_image,
                        last_past_point,
                        other_objects_past_temporal_images,
                        other_objects_future_temporal_images
                    )

    def split_generator3(self):
        split = self.splits[3]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            frame_dict = dict(self.load_frame_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:

                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.STRIDE)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.STRIDE)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0

                    past_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS])
                    future_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])
                    future_points_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])

                    prev_point = None
                    for i, p in enumerate(past_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                            past_trajectory_temporal_images[:, :, i] = past_trajectory_temporal_images[:, :, i - 1]
                            for p1, p2 in line:
                                past_trajectory_temporal_images[p2, p1, i] = 1.0
                        if i == 0:
                            # past_trajectory_temporal_images[x_scaled_discretized, y_scaled_discretized, i] = 1.0
                            past_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        prev_point = (x_scaled_discretized, y_scaled_discretized)

                    last_past_point = prev_point
                    future_trajectory_temporal_images[:, :, 0] = past_trajectory_temporal_images[:, :, -1]
                    for i, p in enumerate(future_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            future_trajectory_temporal_images[:, :, i] = future_trajectory_temporal_images[:, :, i - 1]

                        line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                        for p1, p2 in line:
                            future_trajectory_temporal_images[p2, p1, i] = 1.0
                        future_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0

                        prev_point = (x_scaled_discretized, y_scaled_discretized)
                        future_points_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        future_points_temporal_images[:, :, i] = gaussian_filter(
                            future_points_temporal_images[:, :, i], sigma=self.BLUR_STD)

                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    past_frames_list = past_trajectory_selected[:, 4]
                    future_frames_list = future_trajectory_selected[:, 4]
                    past_future_frames_list = list(past_frames_list) + list(future_frames_list)
                    other_objects_temporal_images = self.get_other_objects_images(frame_dict, obj,
                                                                                  past_future_frames_list, width,
                                                                                  height)

                    other_objects_past_temporal_images = other_objects_temporal_images[:, :, :self.N_PAST_POINTS]
                    other_objects_future_temporal_images = other_objects_temporal_images[:, :, self.N_PAST_POINTS:]

                    yield (
                        past_trajectory_temporal_images,
                        future_trajectory_temporal_images,
                        future_points_temporal_images,
                        future_trajectory_points,
                        object_type, width, height,
                        resized_reference_image,
                        last_past_point,
                        other_objects_past_temporal_images,
                        other_objects_future_temporal_images
                    )

    def split_generator4(self):
        split = self.splits[4]
        for vi in split:
            scene_name = self.videos[vi][0]
            video_id = self.videos[vi][1]
            object_dict = dict(self.load_object_dict(scene_name, video_id))
            frame_dict = dict(self.load_frame_dict(scene_name, video_id))
            objects = object_dict.keys()
            reference_img = self.load_reference_image(scene_name, video_id)
            width, height = reference_img.size
            resized_reference_image = np.array(
                reference_img.resize((self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT))) / 255
            for obj in objects:
                object_trajectories = object_dict[obj][1]
                if len(object_trajectories) == 0:
                    continue
                object_type = object_trajectories[0][0][-1]
                for trajectory in object_trajectories:

                    trajectory = np.array(trajectory)
                    past_trajectory = trajectory[:PAST_LENGTH]
                    future_trajectory = trajectory[PAST_LENGTH:]
                    past_trajectory_selected = past_trajectory[
                        list(reversed(range(PAST_LENGTH - 1, -1, -self.STRIDE)))]
                    future_trajectory_selected = future_trajectory[
                        list(reversed(range(FUTURE_LENGTH - 1, - 1, -self.STRIDE)))]
                    past_trajectory_points = (past_trajectory_selected[:, 0:2] + past_trajectory_selected[:,
                                                                                 2:4]) / 2.0  # center of the box
                    future_trajectory_points = (future_trajectory_selected[:, 0:2] + future_trajectory_selected[:,
                                                                                     2:4]) / 2.0

                    past_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS])
                    future_trajectory_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])
                    future_points_temporal_images = np.zeros(
                        [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS])

                    prev_point = None
                    for i, p in enumerate(past_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                            past_trajectory_temporal_images[:, :, i] = past_trajectory_temporal_images[:, :, i - 1]
                            for p1, p2 in line:
                                past_trajectory_temporal_images[p2, p1, i] = 1.0
                        if i == 0:
                            # past_trajectory_temporal_images[x_scaled_discretized, y_scaled_discretized, i] = 1.0
                            past_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        prev_point = (x_scaled_discretized, y_scaled_discretized)

                    last_past_point = prev_point
                    future_trajectory_temporal_images[:, :, 0] = past_trajectory_temporal_images[:, :, -1]
                    for i, p in enumerate(future_trajectory_points):
                        x, y = p[0], p[1]

                        x_scaled = x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / width
                        x_scaled_discretized = int(x * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // width)
                        y_scaled = y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT / height
                        y_scaled_discretized = int(y * self.TRAJECTORY_IMAGE_WIDTH_HEIGHT // height)
                        if i > 0:
                            future_trajectory_temporal_images[:, :, i] = future_trajectory_temporal_images[:, :, i - 1]

                        line = get_discrete_line(prev_point, (x_scaled_discretized, y_scaled_discretized))
                        for p1, p2 in line:
                            future_trajectory_temporal_images[p2, p1, i] = 1.0
                        future_trajectory_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0

                        prev_point = (x_scaled_discretized, y_scaled_discretized)
                        future_points_temporal_images[y_scaled_discretized, x_scaled_discretized, i] = 1.0
                        future_points_temporal_images[:, :, i] = gaussian_filter(
                            future_points_temporal_images[:, :, i], sigma=self.BLUR_STD)

                    if self.NORMALIZE:
                        past_trajectory_points[:, 0] = past_trajectory_points[:, 0] * 2 / width - 1
                        future_trajectory_points[:, 0] = future_trajectory_points[:, 0] * 2 / width - 1

                        past_trajectory_points[:, 1] = past_trajectory_points[:, 1] * 2 / height - 1
                        future_trajectory_points[:, 1] = future_trajectory_points[:, 1] * 2 / height - 1

                    past_frames_list = past_trajectory_selected[:, 4]
                    future_frames_list = future_trajectory_selected[:, 4]
                    past_future_frames_list = list(past_frames_list) + list(future_frames_list)
                    other_objects_temporal_images = self.get_other_objects_images(frame_dict, obj,
                                                                                  past_future_frames_list, width,
                                                                                  height)

                    other_objects_past_temporal_images = other_objects_temporal_images[:, :, :self.N_PAST_POINTS]
                    other_objects_future_temporal_images = other_objects_temporal_images[:, :, self.N_PAST_POINTS:]

                    yield (
                        past_trajectory_temporal_images,
                        future_trajectory_temporal_images,
                        future_points_temporal_images,
                        future_trajectory_points,
                        object_type, width, height,
                        resized_reference_image,
                        last_past_point,
                        other_objects_past_temporal_images,
                        other_objects_future_temporal_images
                    )

    def create_dataset(self):
        generator_list = [self.split_generator0, self.split_generator1, self.split_generator2, self.split_generator3,
                          self.split_generator4]

        def create_dataset_from_generator(generator):
            dataset = tf.data.Dataset.from_generator(
                generator,
                (tf.float32, tf.float32, tf.float32, tf.float32, tf.int64, tf.int64, tf.int64, tf.float32, tf.int64,
                 tf.float32, tf.float32),
                (tf.TensorShape(
                    [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS]),
                 tf.TensorShape(
                     [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS]),
                 tf.TensorShape(
                     [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS]),
                 tf.TensorShape(
                     [self.N_FUTURE_POINTS, 2]),
                 tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]),
                 tf.TensorShape([self.REF_IMAGE_WIDTH_HEIGHT, self.REF_IMAGE_WIDTH_HEIGHT, 3]),
                 tf.TensorShape([2]),
                 tf.TensorShape(
                     [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_PAST_POINTS]),
                 tf.TensorShape(
                     [self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.TRAJECTORY_IMAGE_WIDTH_HEIGHT, self.N_FUTURE_POINTS]),
                ))
            return dataset

        self.train_dataset_list = [create_dataset_from_generator(generator_list[index]) for index in
                                   range(N_SPLITS) if index != self.TEST_SPLIT_ID]
        # self.test_dataset = create_dataset_from_generator(self.generator_list[self.TEST_SPLIT_ID])
        self.test_dataset = create_dataset_from_generator(generator_list[self.TEST_SPLIT_ID])
        self.unstable_dataset = create_dataset_from_generator(self.unstable_generator)
        self.train_dataset = self.train_dataset_list[0]
        for train_dataset in self.train_dataset_list[1:]:
            self.train_dataset = self.train_dataset.concatenate(train_dataset)

        def shuffle_repeat_batch_get(dataset, epoch=-1):
            return dataset.shuffle(buffer_size=self.SHUFFLE_BUFFER_SIZE, seed=42).repeat(epoch).batch(
                self.BATCH_SIZE).make_one_shot_iterator().get_next()

        self.get_train_batch = shuffle_repeat_batch_get(self.train_dataset)
        self.get_test_batch = shuffle_repeat_batch_get(self.test_dataset, epoch=1)
        self.get_unstable_batch = shuffle_repeat_batch_get(self.unstable_dataset)

    def show_ref_image_trajectories(self, scene_name, video_id):
        ref_image = np.array(self.load_reference_image(scene_name, video_id))
        plt.imshow(ref_image)
        frame_dict = dict(self.load_frame_dict(scene_name, video_id))
        point_list = []
        point_type = []
        for frame in frame_dict.keys():
            print(frame)
            # if frame > 100:
            #     break
            for p in frame_dict[frame]:
                if p[-3] == 1 or p[-4] == 1:
                    continue
                x, y = (p[1] + p[3]) / 2, (p[2] + p[4]) / 2
                point_list.append((x, y))
                point_type.append(p[9])

        point_list = np.array(point_list)
        plt.scatter(point_list[:, 0], point_list[:, 1], alpha=0.01, s=1.0)
        plt.show()
        print(ref_image.shape)
        pass


# if __name__ == "__main__":
#     abosulute_Single_Object_Position_Dataloader = Abosulute_Single_Object_Position_Dataloader(future_resolution=1,
#                                                                                               past_resolution=1,
#                                                                                               shuffle_buffer_size=1024,
#                                                                                               batch_size=256,
#                                                                                               test_split_id=0)
#     ses = tf.InteractiveSession()
#     while True:
#         trbatch = ses.run(abosulute_Single_Object_Position_Dataloader.get_test_batch)
#         a = np.max(trbatch[0])
#         b = np.max(trbatch[1])
#         if a > 1 or b > 1 or a < -1 or b < -1:
#             print(a, b)
#     i = 0
#     past_sequence = trbatch[0][i]
#     future_sequence = trbatch[1][i]
#     print(past_sequence.shape)
#     plt.plot(past_sequence[:, 0], past_sequence[:, 1], 'r')
#     plt.plot(future_sequence[:, 0], future_sequence[:, 1], 'b')
#     plt.show()
#
#     tebatch = ses.run(abosulute_Single_Object_Position_Dataloader.get_test_batch)
#     print(trbatch[0] == tebatch[0])


# if __name__ == "__main__":
#     abosulute_Single_Object_Position_Dataloader = Abosulute_Single_Object_Position_REF_IMG_Dataloader(
#         future_resolution=1,
#         past_resolution=1,
#         shuffle_buffer_size=1024,
#         batch_size=256,
#         test_split_id=0)
#     ses = tf.InteractiveSession()
#     counter = 0
#     while True:
#         trbatch = ses.run(abosulute_Single_Object_Position_Dataloader.get_unstable_batch)
#         counter += len(trbatch[0])
#         print(counter)
#
#     i = 0
#     past_sequence = trbatch[0][i]
#     future_sequence = trbatch[1][i]
#     print(past_sequence.shape)
#     plt.plot(past_sequence[:, 0], past_sequence[:, 1], 'r')
#     plt.plot(future_sequence[:, 0], future_sequence[:, 1], 'b')
#     plt.show()
#
#     tebatch = ses.run(abosulute_Single_Object_Position_Dataloader.get_test_batch)
#     print(trbatch[0] == tebatch[0])


if __name__ == "__main__":
    stride = 15
    A = IMG_Multi_Object_Position_REF_IMG_Dataloader(stride=stride, batch_size=32, test_split_id=0, normalize=True,
                                                     blur_std=0.0, shuffle_buffer_size=128)

    # A.show_ref_image_trajectories("deathCircle", 0)
    counter = 0
    ses = tf.InteractiveSession()
    # while True:
    #     counter += 1
    #     print(counter)
    #     ses.run(A.get_train_batch)

    batch = ses.run(A.get_test_batch)
    import matplotlib.pyplot as plt

    i = 2
    p = batch[0][i]
    f = batch[1][i]
    fp = batch[2][i]
    fp = batch[-1][i]
    pp = batch[-2][i]
    ref_img = batch[-4][i]

    # for i in range(60 // stride):
    #     plt.figure()
    #     plt.imshow(p[:, :, i])
    # for i in range(120 // stride):
    #     plt.figure()
    #     plt.imshow(f[:, :, i])
    for i in range(60 // stride):
        plt.figure()
        plt.imshow(pp[:, :, i])
    for i in range(120 // stride):
        plt.figure()
        plt.imshow(fp[:, :, i])

    plt.figure()
    plt.imshow(ref_img)
    plt.show()
    print(p.shape)
