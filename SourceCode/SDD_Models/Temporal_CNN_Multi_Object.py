import datetime
import os

import tensorflow as tf

from BaseModel import BaseModel
from SDD_Metrics import *
from SDD_Dataloader import IMG_Multi_Object_Position_REF_IMG_Dataloader
from Utils import get_discrete_line


class Temporal_CNN_Multi_Object(BaseModel):
    NAME = "SDD"
    SAVE_DIR = "../SDD_Saved_Models/Temporal_CNN_Multi_Object/"
    RS = 42

    SIZE = 256
    BATCH_SIZE = 64
    EPOCH = 10
    SHUFFLE_BUFFER_SIZE = 4
    # SHUFFLE_BUFFER_SIZE = 2
    INITIAL_LR = 0.001
    TOTAL_STEPS = 5000

    N_H_ENCODER_DECODER_LSTM = 256
    N_H_X_EMBEDDING = 16
    N_H_OUTPUT_EMBEDDING = 200

    DECAY_STEP = 5000

    STRIDE = 1

    PRE_TRAIN_STEPS = 0
    USE_BATCH_NORM_REF_IMG = False
    USE_BATCH_NORM_PAST_IMG_ENCODER = False

    BLUR_STD = 1.0
    SPLIT_ID = 4
    GPU_ID = 0
    DESCRIPTION = ""
    USE_OTHER_OBJECTS = False
    USE_REF_IMG = False

    def __init__(self, name=NAME, save_dir=SAVE_DIR, summary=True):
        super(Temporal_CNN_Multi_Object, self).__init__(name, save_dir)
        self.MODEL_PATH = os.path.join(self.SAVE_DIR, self.NAME)
        try:
            os.mkdir(self.MODEL_PATH)
        except:
            pass
        init_time = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.LOG_PATH = os.path.join(self.SAVE_DIR, "log/" + self.NAME + "-run-" + init_time)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.GPU_ID)
        self.init_graph(summary)
        self.ses = tf.InteractiveSession()

    def cv_output(self):
        error_list = []
        for sp_id in [0, 1, 2, 3, 4]:
            self.SPLIT_ID = sp_id
            self.ses.close()
            tf.reset_default_graph()
            self.ses = tf.InteractiveSession()
            self.init_graph(False)
            self.train(summary=False)
            y_hat_dataset_list, y_dataset, W_dataset, H_dataset = A.generate_oracle_dataset(100, K=10, type='test')
            result = L2_error_1234_second(y_hat_dataset_list, y_dataset, W_dataset, H_dataset, A.STRIDE,
                                          "Min",
                                          down_scale_rate=5, top_percentage=0.1)
            error_list.append(np.array(result[2]))
            print(error_list[-1].shape)
            print("Split id = {}".format(self.SPLIT_ID))
            for t in range(4):
                print("\tMean L2 error at {} sec = {} +- {}".format(t + 1, result[0][t], result[1][t]))

        error_list = np.concatenate(error_list, axis=1)
        error_list = np.transpose(error_list, [1, 0])
        print(error_list.shape)
        print("Total Error")
        for t in range(4):
            print("\t Mean L2 error at {} sec = {} +- {}".format(t + 1, np.mean(error_list[:, t]),
                                                                 np.std(error_list[:, t]) / np.sqrt(len(error_list))))

    def ref_image_encoder(self, ref_img, reuse=False):
        with tf.variable_scope('Ref_IMG_Encoder'):
            conv1 = tf.layers.conv2d(
                inputs=ref_img,
                filters=32,
                kernel_size=[3, 3],
                strides=[2, 2],
                # padding="valid",
                padding="same",
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='conv1',
                reuse=reuse
            )
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=64,
                kernel_size=[3, 3],
                strides=[2, 2],
                # padding="valid",
                padding="same",
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='conv2',
                reuse=reuse
            )
            conv3 = tf.layers.conv2d(
                inputs=conv2,
                filters=128,
                kernel_size=[3, 3],
                strides=[2, 2],
                # padding="valid",
                padding="same",
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='conv3',
                reuse=reuse
            )
            conv4 = tf.layers.conv2d(
                inputs=conv3,
                filters=256,
                kernel_size=[3, 3],
                strides=[2, 2],
                # padding="valid",
                padding="same",
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='conv4',
                reuse=reuse
            )
            return conv1, conv2, conv3, conv4

    def past_trajectory_to_embedding(self, past_trajectory_images, reuse=False):
        # inputs = tf.transpose(past_trajectory_images, [0, 3, 1, 2, 4])
        with tf.variable_scope('Past_Trajectory_Embedding'):
            # past_trajectory_images_padded = tf.pad(past_trajectory_images, paddings=[[0, 0], [8, 8], [8, 8], [0, 0], [0, 0]])
            conv1 = tf.layers.conv3d(
                inputs=past_trajectory_images,
                # inputs=past_trajectory_images_padded,
                filters=32,
                kernel_size=[3, 3, 1],
                strides=[2, 2, 1],
                # padding="valid",
                padding="same",
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='conv1',
                reuse=reuse
            )
            conv2 = tf.layers.conv3d(
                inputs=conv1,
                filters=64,
                # kernel_size=[3, 3, 3],
                kernel_size=[3, 3, self.PAST_LEN],
                strides=[2, 2, 1],
                # padding="valid",
                padding="same",
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='conv2',
                reuse=reuse
            )

            conv3 = tf.layers.conv3d(
                inputs=conv2,
                filters=128,
                # kernel_size=[3, 3, self.PAST_LEN - 2],
                kernel_size=[3, 3, 1],
                strides=[2, 2, 1],
                # padding="valid",
                padding="same",
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='conv3',
                reuse=reuse
            )

            conv4 = tf.layers.conv3d(
                inputs=conv3,
                filters=196,
                kernel_size=[3, 3, 1],
                strides=[2, 2, 1],
                # padding="valid",
                padding="same",
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='conv4',
                reuse=reuse
            )

            if self.PAST_LEN == 1:
                last_conv = conv4[:, :, :, (self.PAST_LEN - 1) // 2:, :]
            else:
                last_conv = conv4[:, :, :, (self.PAST_LEN - 1) // 2: -(self.PAST_LEN // 2), :]

            # print(conv1.shape, conv2.shape, conv3.shape, conv4.shape)
            return last_conv

    def past_other_trajectories_to_embedding(self, past_trajectory_images, reuse=False):
        # inputs = tf.transpose(past_trajectory_images, [0, 3, 1, 2, 4])
        with tf.variable_scope('Past_Other_Trajectory_Embedding'):
            # past_trajectory_images_padded = tf.pad(past_trajectory_images, paddings=[[0, 0], [8, 8], [8, 8], [0, 0], [0, 0]])
            conv1 = tf.layers.conv3d(
                inputs=past_trajectory_images,
                # inputs=past_trajectory_images_padded,
                filters=32,
                kernel_size=[3, 3, 1],
                strides=[2, 2, 1],
                # padding="valid",
                padding="same",
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='conv1',
                reuse=reuse
            )
            conv2 = tf.layers.conv3d(
                inputs=conv1,
                filters=64,
                # kernel_size=[3, 3, 3],
                kernel_size=[3, 3, self.PAST_LEN],
                strides=[2, 2, 1],
                # padding="valid",
                padding="same",
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='conv2',
                reuse=reuse
            )

            conv3 = tf.layers.conv3d(
                inputs=conv2,
                filters=128,
                # kernel_size=[3, 3, self.PAST_LEN - 2],
                kernel_size=[3, 3, 1],
                strides=[2, 2, 1],
                # padding="valid",
                padding="same",
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='conv3',
                reuse=reuse
            )

            conv4 = tf.layers.conv3d(
                inputs=conv3,
                filters=196,
                kernel_size=[3, 3, 1],
                strides=[2, 2, 1],
                # padding="valid",
                padding="same",
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='conv4',
                reuse=reuse
            )

            if self.PAST_LEN == 1:
                last_conv = conv4[:, :, :, (self.PAST_LEN - 1) // 2:, :]
            else:
                last_conv = conv4[:, :, :, (self.PAST_LEN - 1) // 2: -(self.PAST_LEN // 2), :]

            return last_conv

    def past_trajectory_ref_img_embedding_to_prediction(self, past_trajectory_embedding, ref_img_embedding,
                                                        other_trajectories_embedding,
                                                        reuse=False):
        with tf.variable_scope("Embedding_To_Prediction"):
            # x = tf.concat([past_trajectory_embedding, ref_img_embedding], axis=-1)
            # x = ref_img_embedding
            # x = tf.multiply(past_trajectory_embedding, ref_img_embedding)
            # x = past_trajectory_embedding
            if self.USE_OTHER_OBJECTS:
                x = tf.concat([past_trajectory_embedding, other_trajectories_embedding], axis=-1)
            else:
                x = past_trajectory_embedding
            if self.USE_REF_IMG:
                x = tf.concat([x, ref_img_embedding[-1]], axis=-1)
            # deconv1_input = tf.concat([ref_img_embedding[-1], x], axis=-1)
            deconv1_input = x

            deconv1 = tf.layers.conv2d_transpose(
                inputs=deconv1_input,
                filters=128,
                kernel_size=[3, 3],
                strides=[2, 2],
                # padding='valid',
                padding='same',
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='upconv1',
                reuse=reuse
            )
            # deconv2_input = tf.concat([ref_img_embedding[-2], deconv1], axis=-1)
            deconv2_input = deconv1

            deconv2 = tf.layers.conv2d_transpose(
                inputs=deconv2_input,
                filters=64,
                kernel_size=[3, 3],
                strides=[2, 2],
                # padding='valid',
                padding='same',
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='upconv2',
                reuse=reuse
            )
            # deconv3_input = tf.concat([ref_img_embedding[-3], deconv2], axis=-1)
            deconv3_input = deconv2
            deconv3 = tf.layers.conv2d_transpose(
                inputs=deconv3_input,
                filters=32,
                kernel_size=[3, 3],
                strides=[2, 2],
                # padding='valid',
                padding='same',
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.01),
                name='upconv3',
                reuse=reuse
            )
            # deconv4_input = tf.concat([ref_img_embedding[-4], deconv3], axis=-1)
            deconv4_input = deconv3
            deconv4 = tf.layers.conv2d_transpose(
                inputs=deconv4_input,
                filters=1,
                kernel_size=[3, 3],
                strides=[2, 2],
                # padding='valid',
                padding='same',
                # activation=tf.nn.relu,
                # bias_initializer=tf.constant_initializer(0.01),
                name='upconv4',
                reuse=reuse
            )

        last_deconv = deconv4
        # output = tf.image.resize_nearest_neighbor(last_deconv, [self.SIZE, self.SIZE]) * ref_img_embedding
        # output = tf.image.resize_nearest_neighbor(last_deconv, [self.SIZE, self.SIZE])
        output = last_deconv
        return output[:, :, :, 0]

    def init_graph(self, summary=True):
        np.random.seed(self.RS)
        tf.set_random_seed(self.RS)
        with tf.variable_scope(self.NAME):
            with tf.name_scope("Dataset"):
                with tf.name_scope("Dataloader"):
                    self.Dataloader = IMG_Multi_Object_Position_REF_IMG_Dataloader(
                        stride=self.STRIDE,
                        shuffle_buffer_size=self.SHUFFLE_BUFFER_SIZE,
                        batch_size=self.BATCH_SIZE,
                        test_split_id=self.SPLIT_ID,
                        sdd_dataset_dir="../Dataset/SDD/", size=self.SIZE, blur_std=self.BLUR_STD)
                    self.PAST_LEN = self.Dataloader.N_PAST_POINTS
                    self.FUTURE_LEN = self.Dataloader.N_FUTURE_POINTS
                    self.WIDTH_HEIGHT = self.Dataloader.TRAJECTORY_IMAGE_WIDTH_HEIGHT
                    (
                        self.PAST_TRAJECTORY_IMAGES_tr,
                        self.FUTURE_TRAJECTORY_IMAGES_tr,
                        self.FUTURE_TRAJECTORY_POINT_IMAGES_tr,
                        self.FUTURE_TRAJECTORY_POINTS_tr,
                        self.TYPE_tr,
                        self.W_tr,
                        self.H_tr,
                        self.REF_IMG_tr,
                        self.LAST_PAST_POINT_tr,
                        self.OTHER_PAST_TRAJECTORY_IMAGES_tr,
                        self.OTHER_FUTURE_TRAJECTORY_IMAGES_tr
                    ) = self.Dataloader.get_train_batch

                    (
                        self.PAST_TRAJECTORY_IMAGES_te,
                        self.FUTURE_TRAJECTORY_IMAGES_te,
                        self.FUTURE_TRAJECTORY_POINT_IMAGES_te,
                        self.FUTURE_TRAJECTORY_POINTS_te,
                        self.TYPE_te,
                        self.W_te,
                        self.H_te,
                        self.REF_IMG_te,
                        self.LAST_PAST_POINT_te,
                        self.OTHER_PAST_TRAJECTORY_IMAGES_te,
                        self.OTHER_FUTURE_TRAJECTORY_IMAGES_te
                    ) = self.Dataloader.get_test_batch

                    (
                        self.PAST_TRAJECTORY_IMAGES_u,
                        self.FUTURE_TRAJECTORY_IMAGES_u,
                        self.FUTURE_TRAJECTORY_POINT_IMAGES_u,
                        self.FUTURE_TRAJECTORY_POINTS_u,
                        self.TYPE_u,
                        self.W_u,
                        self.H_u,
                        self.REF_IMG_u,
                        self.LAST_PAST_POINT_u,
                        self.OTHER_PAST_TRAJECTORY_IMAGES_u,
                        self.OTHER_FUTURE_TRAJECTORY_IMAGES_u
                    ) = self.Dataloader.get_unstable_batch

            with tf.name_scope("Placeholders"):
                self.FUTURE_TRAJECTORY_POINTS_placeholder = tf.placeholder_with_default(
                    self.FUTURE_TRAJECTORY_POINTS_tr,
                    shape=[None, self.FUTURE_LEN, 2],
                    name='FUTURE_TRAJECTORY_POINTS_placeholder'
                )

                self.PAST_TRAJECTORY_IMAGES_placeholder = tf.placeholder_with_default(
                    self.PAST_TRAJECTORY_IMAGES_tr,
                    shape=[None,
                           self.Dataloader.TRAJECTORY_IMAGE_WIDTH_HEIGHT,
                           self.Dataloader.TRAJECTORY_IMAGE_WIDTH_HEIGHT,
                           self.PAST_LEN],
                    name='PAST_TRAJECTORY_IMAGES_placeholder'
                )

                self.FUTURE_TRAJECTORY_IMAGES_placeholder = tf.placeholder_with_default(
                    self.FUTURE_TRAJECTORY_IMAGES_tr,
                    shape=[None,
                           self.Dataloader.TRAJECTORY_IMAGE_WIDTH_HEIGHT,
                           self.Dataloader.TRAJECTORY_IMAGE_WIDTH_HEIGHT,
                           self.FUTURE_LEN],
                    name='FUTURE_TRAJECTORY_IMAGES_placeholder'
                )

                self.OTHER_PAST_TRAJECTORY_IMAGES_placeholder = tf.placeholder_with_default(
                    self.OTHER_PAST_TRAJECTORY_IMAGES_tr,
                    shape=[None,
                           self.Dataloader.TRAJECTORY_IMAGE_WIDTH_HEIGHT,
                           self.Dataloader.TRAJECTORY_IMAGE_WIDTH_HEIGHT,
                           self.PAST_LEN],
                    name='PAST_TRAJECTORY_IMAGES_placeholder'
                )

                self.OTHER_FUTURE_TRAJECTORY_IMAGES_placeholder = tf.placeholder_with_default(
                    self.OTHER_FUTURE_TRAJECTORY_IMAGES_tr,
                    shape=[None,
                           self.Dataloader.TRAJECTORY_IMAGE_WIDTH_HEIGHT,
                           self.Dataloader.TRAJECTORY_IMAGE_WIDTH_HEIGHT,
                           self.FUTURE_LEN],
                    name='FUTURE_TRAJECTORY_IMAGES_placeholder'
                )

                self.FUTURE_TRAJECTORY_POINT_IMAGES_placeholder = tf.placeholder_with_default(
                    self.FUTURE_TRAJECTORY_POINT_IMAGES_tr,
                    shape=[None,
                           self.Dataloader.TRAJECTORY_IMAGE_WIDTH_HEIGHT,
                           self.Dataloader.TRAJECTORY_IMAGE_WIDTH_HEIGHT,
                           self.FUTURE_LEN],
                    name='FUTURE_TRAJECTORY_POINT_IMAGES_placeholder')

                self.REF_IMAGE_placeholder = tf.placeholder_with_default(
                    self.REF_IMG_tr,
                    shape=[None,
                           self.Dataloader.REF_IMAGE_WIDTH_HEIGHT,
                           self.Dataloader.REF_IMAGE_WIDTH_HEIGHT,
                           3],
                    name='REF_IMAGE_placeholder'
                )

                self.OBJECT_TYPE_placeholder = tf.placeholder_with_default(
                    self.TYPE_tr,
                    shape=[None],
                    name='OBJECT_TYPE_placeholder'
                )

                self.LAST_PAST_POINT_placeholder = tf.placeholder_with_default(
                    self.LAST_PAST_POINT_tr,
                    shape=[None, 2],
                    name='LAST_PAST_POINT_placeholder'
                )

                self.W_placeholder = tf.placeholder_with_default(self.W_tr, shape=[None], name='W_placeholder')
                self.H_placeholder = tf.placeholder_with_default(self.H_tr, shape=[None], name='H_placeholder')

                self.error_in_1_sec_placeholder = tf.placeholder_with_default(0.0, shape=[],
                                                                              name='error_in_1_sec_placeholder')
                self.error_in_2_sec_placeholder = tf.placeholder_with_default(0.0, shape=[],
                                                                              name='error_in_2_sec_placeholder')
                self.error_in_3_sec_placeholder = tf.placeholder_with_default(0.0, shape=[],
                                                                              name='error_in_3_sec_placeholder')
                self.error_in_4_sec_placeholder = tf.placeholder_with_default(0.0, shape=[],
                                                                              name='error_in_4_sec_placeholder')
                self.training_phase = tf.placeholder_with_default(True, shape=[])

            with tf.variable_scope("Inference"):
                (self.PAST_TRAJECTORY_IMAGES, self.FUTURE_TRAJECTORY_IMAGES, self.FUTURE_TRAJECTORY_POINT_IMAGES,
                 self.REF_IMAGE, self.OTHER_PAST_TRAJECTORY_IMAGES, self.OTHER_FUTURE_TRAJECTORY_IMAGES) = [
                    self.PAST_TRAJECTORY_IMAGES_placeholder,
                    self.FUTURE_TRAJECTORY_IMAGES_placeholder,
                    self.FUTURE_TRAJECTORY_POINT_IMAGES_placeholder,
                    self.REF_IMAGE_placeholder,
                    self.OTHER_PAST_TRAJECTORY_IMAGES_placeholder,
                    self.OTHER_FUTURE_TRAJECTORY_IMAGES_placeholder
                ]
                # print("Here", self.REF_IMAGE.shape)

                self.encoded_ref_img = self.ref_image_encoder(self.REF_IMAGE, reuse=False)
                past_future_trajectory_images = tf.concat(
                    [self.PAST_TRAJECTORY_IMAGES, self.FUTURE_TRAJECTORY_IMAGES[:, :, :, :-1]],
                    axis=-1, name='past_future_trajectory_images')
                past_future_other_trajectory_images = tf.concat(
                    [self.OTHER_PAST_TRAJECTORY_IMAGES, self.OTHER_FUTURE_TRAJECTORY_IMAGES[:, :, :, :-1]],
                    axis=-1, name='past_future_other_trajectory_images')
                self.ref_image_tiled = tf.transpose(tf.tile(self.REF_IMAGE[:, :, :, :, tf.newaxis],
                                                            [1, 1, 1, 1, self.FUTURE_LEN + self.PAST_LEN - 1]),
                                                    [0, 1, 2, 4, 3])

                past_future_trajectory_images_channeled = past_future_trajectory_images[:, :, :, :, tf.newaxis]
                past_future_other_trajectory_images_channeled = past_future_other_trajectory_images[:, :, :, :,
                                                                tf.newaxis]
                # past_future_information = tf.concat(
                #     [past_future_trajectory_images_channeled, past_future_other_trajectory_images_channeled], axis=-1)
                self.encoded_past_future_trajectory = self.past_trajectory_to_embedding(
                    past_future_trajectory_images_channeled, reuse=False)
                self.encoded_other_past_future_trajectories = self.past_other_trajectories_to_embedding(
                    past_future_other_trajectory_images_channeled, reuse=False)

                with tf.name_scope("sampling_first_point"):
                    encoded_past_trajectory = self.past_trajectory_to_embedding(
                        self.PAST_TRAJECTORY_IMAGES[:, :, :, :, tf.newaxis],
                        reuse=True)
                    encoded_other_past_trajectories = self.past_other_trajectories_to_embedding(
                        self.OTHER_PAST_TRAJECTORY_IMAGES[:, :, :, :, tf.newaxis],
                        reuse=True)

                    self.first_generated_logit = self.past_trajectory_ref_img_embedding_to_prediction(
                        encoded_past_trajectory[:, :, :, 0, :],
                        self.encoded_ref_img,
                        encoded_other_past_trajectories[:, :, :, 0, :],
                        reuse=False)
                    logits_reshaped = tf.reshape(self.first_generated_logit,
                                                 [-1, self.WIDTH_HEIGHT * self.WIDTH_HEIGHT])
                    self.first_log_probs = tf.nn.log_softmax(logits_reshaped)
                    sampled_pixel = tf.multinomial(logits_reshaped, 1, seed=self.RS + 1000)
                    sampled_y = tf.cast(tf.floor(sampled_pixel / self.WIDTH_HEIGHT, "argmax_generated_x"),
                                        dtype=tf.float32)
                    sampled_x = tf.cast(tf.mod(sampled_pixel, self.WIDTH_HEIGHT, "argmax_generated_y"),
                                        dtype=tf.float32)
                    self.first_sampled_position = tf.concat(
                        [sampled_x, sampled_y],
                        axis=1)

                    self.first_sampled_position_scaled = tf.concat(
                        [(sampled_x * 2 / self.SIZE - 1),
                         (sampled_y * 2 / self.SIZE - 1)],
                        axis=1)

                    self.guideless_generated_image = tf.reshape(
                        tf.nn.softmax(
                            tf.reshape(self.first_generated_logit, [-1, self.WIDTH_HEIGHT * self.WIDTH_HEIGHT])),
                        [-1, self.WIDTH_HEIGHT, self.WIDTH_HEIGHT, 1])

                self.guided_generated_logits_list = []
                self.guided_generated_log_probs_list = []
                self.guided_generated_argmax_positions_list = []
                for i in range(self.FUTURE_LEN):
                    guided_generated_logit = self.past_trajectory_ref_img_embedding_to_prediction(
                        self.encoded_past_future_trajectory[:, :, :, i, :],
                        self.encoded_ref_img,
                        self.encoded_other_past_future_trajectories[:, :, :, i, :],
                        reuse=True)
                    self.guided_generated_logits_list.append(guided_generated_logit)
                    logits_reshaped = tf.reshape(guided_generated_logit, [-1, self.WIDTH_HEIGHT * self.WIDTH_HEIGHT])

                    guided_argmax_pixel = tf.argmax(logits_reshaped, axis=1)
                    guided_argmax_generated_y = tf.cast(
                        tf.floor(guided_argmax_pixel / self.WIDTH_HEIGHT, "argmax_generated_y"),
                        dtype=tf.float32)
                    guided_argmax_generated_x = tf.cast(
                        tf.mod(guided_argmax_pixel, self.WIDTH_HEIGHT, "argmax_generated_x"),
                        dtype=tf.float32)
                    guided_generated_argmax_position = tf.concat(
                        [guided_argmax_generated_x[:, tf.newaxis],
                         guided_argmax_generated_y[:, tf.newaxis]],
                        axis=1)
                    log_probs = tf.reshape(tf.nn.log_softmax(logits_reshaped),
                                           [-1, self.WIDTH_HEIGHT, self.WIDTH_HEIGHT])
                    self.guided_generated_log_probs_list.append(log_probs)
                    self.guided_generated_argmax_positions_list.append(guided_generated_argmax_position)

                self.guided_generated_positions_tensor = tf.transpose(
                    tf.stack(self.guided_generated_argmax_positions_list),
                    [1, 0, 2]) * 2 / self.WIDTH_HEIGHT - 1
                self.guided_generated_logits_tensor = tf.transpose(
                    tf.stack(self.guided_generated_logits_list),
                    [1, 2, 3, 0]
                )
                self.guided_generated_log_probs_tensor = tf.transpose(
                    tf.stack(self.guided_generated_log_probs_list),
                    [1, 2, 3, 0]
                )

            with tf.name_scope("Optimization"):
                step = (120 // 4) // self.STRIDE
                starting_index = step - 1
                desired_points = [starting_index + _ * step for _ in range(4)]
                self.square_differences = tf.reduce_sum(
                    tf.square(self.guided_generated_positions_tensor - self.FUTURE_TRAJECTORY_POINTS_placeholder),
                    axis=2,
                    name='square_differences')
                self.mse_loss = tf.reduce_mean(tf.reduce_mean(self.square_differences, axis=1), name='mse_loss')

                y_hat = tf.reshape(self.guided_generated_log_probs_tensor,
                                   [-1, self.WIDTH_HEIGHT * self.WIDTH_HEIGHT, self.FUTURE_LEN])

                self.TEST1 = y_hat[:, :, 0] - self.first_log_probs
                self.TEST2 = self.guided_generated_logits_tensor[:, :, :, 0] - self.first_generated_logit

                y = tf.reshape(self.FUTURE_TRAJECTORY_POINT_IMAGES,
                               [-1, self.WIDTH_HEIGHT * self.WIDTH_HEIGHT, self.FUTURE_LEN])
                self.long_horizon_loss = -tf.reduce_mean(
                    tf.reduce_sum(tf.multiply(self.first_log_probs, y[:, :, self.FUTURE_LEN // 2]), axis=1))
                # print(self.first_log_probs.shape)
                # self.ce_loss = -tf.reduce_mean(
                #     tf.reduce_mean(
                #         tf.reduce_sum(
                #             tf.reduce_sum(
                #                 tf.multiply(
                #                     self.guided_generated_log_probs_tensor,
                #                     self.FUTURE_TRAJECTORY_POINT_IMAGES),
                #                 axis=1),
                #             axis=1),
                #         axis=1),
                # )
                self.ce_loss = -tf.reduce_mean(
                    tf.reduce_sum(
                        tf.reduce_sum(
                            tf.multiply(
                                y_hat,
                                # self.first_log_probs,
                                y,
                                # y[:, :, 0],
                            ),
                            axis=1),
                        axis=1)
                )
                self.ce_loss_batch = -tf.reduce_sum(
                    tf.reduce_sum(
                        tf.multiply(
                            y_hat,
                            # self.first_log_probs,
                            y,
                            # y[:, :, 0],
                        ),
                        axis=1),
                    axis=1)

                # self.segmentation_loss = tf.reduce_sum(
                #     tf.reduce_mean(self.encoded_ref_img * 2 / (self.WIDTH_HEIGHT * self.WIDTH_HEIGHT), axis=0))

                self.objective = self.ce_loss
                # self.objective = self.long_horizon_loss
                self.global_step = tf.train.get_or_create_global_step()
                self.learning_rate = tf.train.exponential_decay(self.INITIAL_LR, self.global_step, self.DECAY_STEP, 0.5,
                                                                name='learning_rate')
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                # self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
                # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(update_ops):
                # gvs = self.optimizer.compute_gradients(self.mse_loss)
                # clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                # self.train_operation = self.optimizer.apply_gradients(clipped_gvs, global_step=self.global_step)
                self.train_operation = self.optimizer.minimize(self.objective, global_step=self.global_step)

            self.init_node = tf.global_variables_initializer()
            self.save_node = tf.train.Saver()

        if summary:
            with tf.name_scope("Summary"):
                self.nll_summary = tf.summary.scalar(name='negative log likelihood', tensor=self.ce_loss)
                self.long_horizion_nll_summary = tf.summary.scalar(name='negative log likelihood long horizon',
                                                                   tensor=self.long_horizon_loss)
                self.mse_summary = tf.summary.scalar(name='guided mse loss', tensor=self.mse_loss)
                self.error_in_1_sec_summary = tf.summary.scalar(name='Error in 1.0s',
                                                                tensor=self.error_in_1_sec_placeholder)
                self.error_in_2_sec_summary = tf.summary.scalar(name='Error in 2.0s',
                                                                tensor=self.error_in_2_sec_placeholder)
                self.error_in_3_sec_summary = tf.summary.scalar(name='Error in 3.0s',
                                                                tensor=self.error_in_3_sec_placeholder)
                self.error_in_4_sec_summary = tf.summary.scalar(name='Error in 4.0s',
                                                                tensor=self.error_in_4_sec_placeholder)
                self.scalar_summaries = tf.summary.merge(
                    [
                        # self.long_horizion_nll_summary
                        self.nll_summary,
                        self.mse_summary,
                    ])
                self.additional_summaries = tf.summary.merge(
                    [
                        self.error_in_1_sec_summary,
                        self.error_in_2_sec_summary,
                        self.error_in_3_sec_summary,
                        self.error_in_4_sec_summary
                    ])

                ref_image_transparent = tf.concat(
                    [self.REF_IMAGE, (1 + self.FUTURE_TRAJECTORY_IMAGES[:, :, :, -1:]) / 2],
                    axis=-1)
                self.ref_image_summary = tf.summary.image("ref_image", ref_image_transparent, max_outputs=3)
                # self.embeded_image_summary = tf.summary.image("embeded_image", self.encoded_ref_img, max_outputs=3)
                self.heatmap_image_summary = tf.summary.image("heatmap_next_position",
                                                              self.guideless_generated_image,
                                                              max_outputs=3)

                trajectory_image = tf.concat([self.FUTURE_TRAJECTORY_IMAGES[:, :, :, -1:],
                                              self.PAST_TRAJECTORY_IMAGES[:, :, :, -1:],
                                              self.guideless_generated_image,
                                              ], axis=-1)
                self.last_given_trajectory_image_summary = tf.summary.image("last_given_trajectory_image_summary",
                                                                            trajectory_image,
                                                                            max_outputs=3)
                self.image_summary = tf.summary.merge(
                    [
                        self.ref_image_summary,
                        # self.embeded_image_summary,
                        self.heatmap_image_summary,
                        self.last_given_trajectory_image_summary,
                    ]
                )

                ref_image_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                        scope="{}/Inference/Ref_IMG_Encoder".format(self.NAME))
                # print(ref_image_variables)
                # print(len(ref_image_variables))
                # self.conv1_weights_historgram_summary = tf.summary.histogram("conv1", ref_image_variables[0])
                # self.conv2_weights_historgram_summary = tf.summary.histogram("conv2", ref_image_variables[2])
                # self.conv3_weights_historgram_summary = tf.summary.histogram("conv3", ref_image_variables[4])
                # self.conv4_weights_historgram_summary = tf.summary.histogram("conv4", ref_image_variables[6])
                #
                # self.histogram_summaries = tf.summary.merge(
                #     [
                #         self.conv1_weights_historgram_summary,
                #         self.conv2_weights_historgram_summary,
                #         self.conv3_weights_historgram_summary,
                #         self.conv4_weights_historgram_summary
                #     ]
                # )

                self.merged_summary = tf.summary.merge_all()

                self.summary_writer = tf.summary.FileWriter(self.LOG_PATH, tf.get_default_graph())
                self.validation_summary_writer = tf.summary.FileWriter(self.LOG_PATH + "validation")
                self.additional_summary_writer = tf.summary.FileWriter(self.LOG_PATH + "additional")

    def add_code_summary(self):
        code_string = "\n".join(open('Temporal_CNN_Multi_Object.py', 'r').readlines())
        text_tensor = tf.make_tensor_proto(self.DESCRIPTION + "\n\n" + code_string, dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        summary.value.add(tag="Hyper parameters", metadata=meta, tensor=text_tensor)
        self.summary_writer.add_summary(summary)

    def init_variables(self):
        self.ses.run(self.init_node)

    def train(self, summary=True):
        self.init_variables()
        print("Training on all splits excliding -{}".format(self.SPLIT_ID))
        if summary:
            self.add_code_summary()
        for step in range(self.TOTAL_STEPS):
            if step % 50 == 0:
                print("Step = {}".format(step))

            if summary:
                _, scalar_summaries = self.ses.run(
                    [self.train_operation,
                     self.scalar_summaries,
                     # self.histogram_summaries
                     ])
                self.summary_writer.add_summary(scalar_summaries, step)
                # self.summary_writer.add_summary(histogram_summaries, step)
            else:
                _ = self.ses.run(self.train_operation)

            if summary:
                if step % 10 == 0:
                    image_summary = self.ses.run(self.image_summary)
                    self.summary_writer.add_summary(image_summary, step)

                if step % 250 == 24999:

                    print("Step {}".format(step))
                    y_hat_dataset_list, y_dataset, W_dataset, H_dataset = A.generate_oracle_dataset(500, K=10,
                                                                                                    type='train')
                    result = L2_error_1234_second(y_hat_dataset_list, y_dataset, W_dataset, H_dataset,
                                                  A.STRIDE,
                                                  "Min",
                                                  down_scale_rate=5, top_percentage=0.1)
                    feed_dict = {self.error_in_1_sec_placeholder: result[0][0],
                                 self.error_in_2_sec_placeholder: result[0][1],
                                 self.error_in_3_sec_placeholder: result[0][2],
                                 self.error_in_4_sec_placeholder: result[0][3]}
                    for t in range(4):
                        print("\tMean L2 error at {} sec = {} +- {}".format(t + 1, result[0][t], result[1][t]))
                    additional_scalar_summaries = self.ses.run(
                        self.additional_summaries, feed_dict=feed_dict)
                    self.additional_summary_writer.add_summary(additional_scalar_summaries, step)

                print("Step = {}".format(step))
                if step % 50 == 0:
                    (PAST_TRAJECTORY_IMAGES, FUTURE_TRAJECTORY_IMAGES, FUTURE_TRAJECTORY_POINT_IMAGES,
                     FUTURE_TRAJECTORY_POINTS, TYPE, W, H, REF_IMAGE, LAST_PAST_POINT, OTHER_PAST_TRAJECTORY_IMAGES,
                     OTHER_FUTURE_TRAJECTORY_IMAGES) = self.ses.run(
                        [
                            self.PAST_TRAJECTORY_IMAGES_u,
                            self.FUTURE_TRAJECTORY_IMAGES_u,
                            self.FUTURE_TRAJECTORY_POINT_IMAGES_u,
                            self.FUTURE_TRAJECTORY_POINTS_u,
                            self.TYPE_u,
                            self.W_u,
                            self.H_u,
                            self.REF_IMG_u,
                            self.LAST_PAST_POINT_u,
                            self.OTHER_PAST_TRAJECTORY_IMAGES_u,
                            self.OTHER_FUTURE_TRAJECTORY_IMAGES_u
                        ])
                    feed_dict = {
                        self.PAST_TRAJECTORY_IMAGES_placeholder: PAST_TRAJECTORY_IMAGES,
                        self.FUTURE_TRAJECTORY_IMAGES_placeholder: FUTURE_TRAJECTORY_IMAGES,
                        self.FUTURE_TRAJECTORY_POINT_IMAGES_placeholder: FUTURE_TRAJECTORY_POINT_IMAGES,
                        self.FUTURE_TRAJECTORY_POINTS_placeholder: FUTURE_TRAJECTORY_POINTS,
                        self.OBJECT_TYPE_placeholder: TYPE,
                        self.W_placeholder: W,
                        self.H_placeholder: H,
                        self.REF_IMAGE_placeholder: REF_IMAGE,
                        self.LAST_PAST_POINT_placeholder: LAST_PAST_POINT,
                        self.training_phase: False,
                        self.OTHER_PAST_TRAJECTORY_IMAGES_placeholder: OTHER_PAST_TRAJECTORY_IMAGES,
                        self.OTHER_FUTURE_TRAJECTORY_IMAGES_placeholder: OTHER_FUTURE_TRAJECTORY_IMAGES
                    }
                    validation_scalar_summaries = self.ses.run(self.scalar_summaries,
                                                               feed_dict=feed_dict)
                    self.validation_summary_writer.add_summary(validation_scalar_summaries, step)

    def save(self):
        self.save_node.save(self.ses, save_path=self.MODEL_PATH + "/" + self.NAME + '.ckpt')

    def load(self):
        self.save_node.restore(self.ses, save_path=self.MODEL_PATH + "/" + self.NAME + '.ckpt')

    def generate_oracle_dataset(self, n=100, K=10, type='train', verbose=False):
        if type == 'train':
            input_fetch = [
                self.PAST_TRAJECTORY_IMAGES_tr,
                self.FUTURE_TRAJECTORY_IMAGES_tr,
                self.FUTURE_TRAJECTORY_POINT_IMAGES_tr,
                self.FUTURE_TRAJECTORY_POINTS_tr,
                self.TYPE_tr,
                self.W_tr,
                self.H_tr,
                self.REF_IMG_tr,
                self.LAST_PAST_POINT_tr,
                self.OTHER_PAST_TRAJECTORY_IMAGES_tr,
                self.OTHER_FUTURE_TRAJECTORY_IMAGES_tr
            ]
        elif type == 'test':
            input_fetch = [
                self.PAST_TRAJECTORY_IMAGES_te,
                self.FUTURE_TRAJECTORY_IMAGES_te,
                self.FUTURE_TRAJECTORY_POINT_IMAGES_te,
                self.FUTURE_TRAJECTORY_POINTS_te,
                self.TYPE_te,
                self.W_te,
                self.H_te,
                self.REF_IMG_te,
                self.LAST_PAST_POINT_te,
                self.OTHER_PAST_TRAJECTORY_IMAGES_te,
                self.OTHER_FUTURE_TRAJECTORY_IMAGES_te,
            ]
            n = 1000000
        else:
            input_fetch = [
                self.PAST_TRAJECTORY_IMAGES_u,
                self.FUTURE_TRAJECTORY_IMAGES_u,
                self.FUTURE_TRAJECTORY_POINT_IMAGES_u,
                self.FUTURE_TRAJECTORY_POINTS_u,
                self.TYPE_u,
                self.W_u,
                self.H_u,
                self.REF_IMG_u,
                self.LAST_PAST_POINT_u,
                self.OTHER_PAST_TRAJECTORY_IMAGES_u,
                self.OTHER_FUTURE_TRAJECTORY_IMAGES_u
            ]

        y_hat_dataset_list = [[] for _ in range(K)]
        y_dataset = []
        W_dataset = []
        H_dataset = []
        counter = 0
        while (counter < n):
            try:
                (
                    PAST_TRAJECTORY_IMAGES,
                    FUTURE_TRAJECTORY_IMAGES,
                    FUTURE_TRAJECTORY_POINT_IMAGES,
                    FUTURE_TRAJECTORY_POINTS,
                    TYPE,
                    W,
                    H,
                    REF_IMAGE,
                    LAST_PAST_POINT,
                    A,
                    B
                ) = self.ses.run(input_fetch)
            except:
                break
            l = len(PAST_TRAJECTORY_IMAGES)
            counter += l
            PAST_TRAJECTORY_IMAGES_BACKUP = np.copy(PAST_TRAJECTORY_IMAGES)
            LAST_PAST_POINT_BACKUP = np.copy(LAST_PAST_POINT)
            encoded_ref_img = self.ses.run(self.encoded_ref_img, feed_dict={self.REF_IMAGE_placeholder: REF_IMAGE})
            for k in range(K):
                PAST_TRAJECTORY_IMAGES = np.copy(PAST_TRAJECTORY_IMAGES_BACKUP)
                LAST_PAST_POINT = np.copy(LAST_PAST_POINT_BACKUP)
                if verbose:
                    print("Counter = {} / K = {}".format(counter, k))
                y_hat = np.zeros([l, self.FUTURE_LEN, 2])
                for t in range(self.FUTURE_LEN):
                    feed_dict = {
                        self.PAST_TRAJECTORY_IMAGES_placeholder: PAST_TRAJECTORY_IMAGES,
                        # self.FUTURE_TRAJECTORY_IMAGES_placeholder: FUTURE_TRAJECTORY_IMAGES,
                        # self.FUTURE_TRAJECTORY_POINT_IMAGES_placeholder: FUTURE_TRAJECTORY_POINT_IMAGES,
                        # self.FUTURE_TRAJECTORY_POINTS_placeholder: FUTURE_TRAJECTORY_POINTS,
                        # self.OBJECT_TYPE_placeholder: TYPE,
                        self.W_placeholder: W,
                        self.H_placeholder: H,
                        # self.REF_IMAGE_placeholder: REF_IMAGE,
                        self.encoded_ref_img: encoded_ref_img,
                        # self.LAST_PAST_POINT_placeholder: LAST_PAST_POINT,
                        self.training_phase: False
                    }
                    sampled_position, scaled_sampled_position = self.ses.run(
                        [self.first_sampled_position, self.first_sampled_position_scaled], feed_dict=feed_dict)

                    y_hat[:, t, :] = scaled_sampled_position
                    sampled_position = sampled_position.astype(np.int64)

                    PAST_TRAJECTORY_IMAGES[:, :, :, :-1] = PAST_TRAJECTORY_IMAGES[:, :, :, 1:]
                    for b in range(len(sampled_position)):
                        line = get_discrete_line(LAST_PAST_POINT[b], sampled_position[b])
                        for p1, p2 in line:
                            PAST_TRAJECTORY_IMAGES[b, p2, p1, -1] = 1.0

                    LAST_PAST_POINT = sampled_position

                y_hat_dataset_list[k].append(y_hat)
            y_dataset.append(FUTURE_TRAJECTORY_POINTS)
            W_dataset.append(W)
            H_dataset.append(H)

        y_hat_dataset_list = [np.concatenate(y_hat_dataset, axis=0) for y_hat_dataset in y_hat_dataset_list]
        y_dataset = np.concatenate(y_dataset, axis=0)
        W_dataset = np.concatenate(W_dataset, axis=0)
        H_dataset = np.concatenate(H_dataset, axis=0)
        return y_hat_dataset_list, y_dataset, W_dataset, H_dataset

    def generate_visual_samples(self, n=100, K=10, type='train', verbose=False):
        if type == 'train':
            input_fetch = [
                self.PAST_TRAJECTORY_IMAGES_tr,
                self.FUTURE_TRAJECTORY_IMAGES_tr,
                self.FUTURE_TRAJECTORY_POINT_IMAGES_tr,
                self.FUTURE_TRAJECTORY_POINTS_tr,
                self.TYPE_tr,
                self.W_tr,
                self.H_tr,
                self.REF_IMG_tr,
                self.LAST_PAST_POINT_tr,
                # self.OTHER_PAST_TRAJECTORY_IMAGES_tr,
                # self.OTHER_FUTURE_TRAJECTORY_IMAGES_tr
            ]
        elif type == 'test':
            input_fetch = [
                self.PAST_TRAJECTORY_IMAGES_te,
                self.FUTURE_TRAJECTORY_IMAGES_te,
                self.FUTURE_TRAJECTORY_POINT_IMAGES_te,
                self.FUTURE_TRAJECTORY_POINTS_te,
                self.TYPE_te,
                self.W_te,
                self.H_te,
                self.REF_IMG_te,
                self.LAST_PAST_POINT_te,
                # self.OTHER_PAST_TRAJECTORY_IMAGES_te,
                # self.OTHER_FUTURE_TRAJECTORY_IMAGES_te,
            ]
        else:
            input_fetch = [
                self.PAST_TRAJECTORY_IMAGES_u,
                self.FUTURE_TRAJECTORY_IMAGES_u,
                self.FUTURE_TRAJECTORY_POINT_IMAGES_u,
                self.FUTURE_TRAJECTORY_POINTS_u,
                self.TYPE_u,
                self.W_u,
                self.H_u,
                self.REF_IMG_u,
                self.LAST_PAST_POINT_u,
                # self.OTHER_PAST_TRAJECTORY_IMAGES_u,
                # self.OTHER_FUTURE_TRAJECTORY_IMAGES_u
            ]

        y_hat_dataset_list = [[] for _ in range(K)]
        y_hat_discrete_dataset_list = [[] for _ in range(K)]
        generated_images_dataset_list = [[] for _ in range(K)]
        y_dataset = []
        W_dataset = []
        H_dataset = []
        past_images_dataset = []
        past_last_point_dataset = []
        ref_images_dataset = []

        counter = 0
        while (counter < n):
            try:
                (
                    PAST_TRAJECTORY_IMAGES,
                    FUTURE_TRAJECTORY_IMAGES,
                    FUTURE_TRAJECTORY_POINT_IMAGES,
                    FUTURE_TRAJECTORY_POINTS,
                    TYPE,
                    W,
                    H,
                    REF_IMAGE,
                    LAST_PAST_POINT
                ) = self.ses.run(input_fetch)
            except:
                break
            l = len(PAST_TRAJECTORY_IMAGES)
            counter += l
            PAST_TRAJECTORY_IMAGES_BACKUP = np.copy(PAST_TRAJECTORY_IMAGES)
            LAST_PAST_POINT_BACKUP = np.copy(LAST_PAST_POINT)
            for k in range(K):
                PAST_TRAJECTORY_IMAGES = np.copy(PAST_TRAJECTORY_IMAGES_BACKUP)
                LAST_PAST_POINT = np.copy(LAST_PAST_POINT_BACKUP)
                if verbose:
                    print("Counter = {} / K = {}".format(counter, k))
                y_hat = np.zeros([l, self.FUTURE_LEN, 2])
                y_hat_discrete = np.zeros([l, self.FUTURE_LEN, 2])
                for t in range(self.FUTURE_LEN):
                    feed_dict = {
                        self.PAST_TRAJECTORY_IMAGES_placeholder: PAST_TRAJECTORY_IMAGES,
                        # self.FUTURE_TRAJECTORY_IMAGES_placeholder: FUTURE_TRAJECTORY_IMAGES,
                        # self.FUTURE_TRAJECTORY_POINT_IMAGES_placeholder: FUTURE_TRAJECTORY_POINT_IMAGES,
                        # self.FUTURE_TRAJECTORY_POINTS_placeholder: FUTURE_TRAJECTORY_POINTS,
                        # self.OBJECT_TYPE_placeholder: TYPE,
                        self.W_placeholder: W,
                        self.H_placeholder: H,
                        self.REF_IMAGE_placeholder: REF_IMAGE,
                        # self.LAST_PAST_POINT_placeholder: LAST_PAST_POINT,
                        self.training_phase: False
                    }
                    sampled_position, scaled_sampled_position = self.ses.run(
                        [self.first_sampled_position, self.first_sampled_position_scaled], feed_dict=feed_dict)

                    y_hat[:, t, :] = scaled_sampled_position
                    y_hat_discrete[:, t, :] = sampled_position
                    sampled_position = sampled_position.astype(np.int64)

                    PAST_TRAJECTORY_IMAGES[:, :, :, :-1] = PAST_TRAJECTORY_IMAGES[:, :, :, 1:]
                    for b in range(len(sampled_position)):
                        line = get_discrete_line(LAST_PAST_POINT[b], sampled_position[b])
                        for p1, p2 in line:
                            PAST_TRAJECTORY_IMAGES[b, p2, p1, -1] = 1.0

                    LAST_PAST_POINT = sampled_position

                y_hat_dataset_list[k].append(y_hat)
                y_hat_discrete_dataset_list[k].append(y_hat_discrete)

            ref_images_dataset.append(REF_IMAGE)
            past_images_dataset.append(PAST_TRAJECTORY_IMAGES_BACKUP[:, :, :, -1])
            past_last_point_dataset.append(LAST_PAST_POINT_BACKUP)
            y_dataset.append(FUTURE_TRAJECTORY_POINTS)
            W_dataset.append(W)
            H_dataset.append(H)

        y_hat_dataset_list = [np.concatenate(y_hat_dataset, axis=0) for y_hat_dataset in y_hat_dataset_list]
        y_hat_discrete_dataset_list = [np.concatenate(y_hat_discrete, axis=0).astype(np.int32) for y_hat_discrete in
                                       y_hat_discrete_dataset_list]

        y_dataset = np.concatenate(y_dataset, axis=0)
        W_dataset = np.concatenate(W_dataset, axis=0)
        H_dataset = np.concatenate(H_dataset, axis=0)
        past_images_dataset = np.concatenate(past_images_dataset, axis=0)
        past_last_point_dataset = np.concatenate(past_last_point_dataset, axis=0)
        ref_images_dataset = np.concatenate(ref_images_dataset, axis=0)

        for b in range(counter):
            for k in range(K):
                generated_trajectory_image = np.copy(past_images_dataset[b])
                last_past_point = np.copy(past_last_point_dataset[b])
                for l in range(self.FUTURE_LEN):

                    line = get_discrete_line(last_past_point, y_hat_discrete_dataset_list[k][b, l])
                    for p1, p2 in line:
                        generated_trajectory_image[p2, p1] = 1.0
                    last_past_point = y_hat_discrete_dataset_list[k][b, l]
                generated_images_dataset_list[k].append(generated_trajectory_image)


        generated_images_dataset_list = [np.stack(generated_images_dataset, axis=0).astype(np.int32) for generated_images_dataset in
                                         generated_images_dataset_list]

        return generated_images_dataset_list, ref_images_dataset, past_images_dataset

    def evaluate_likelihood(self, n=1000, type='train'):
        if type == 'train':
            input_fetch = [
                self.PAST_TRAJECTORY_IMAGES_tr,
                self.FUTURE_TRAJECTORY_IMAGES_tr,
                self.FUTURE_TRAJECTORY_POINT_IMAGES_tr,
                self.FUTURE_TRAJECTORY_POINTS_tr,
                self.TYPE_tr,
                self.W_tr,
                self.H_tr,
                self.REF_IMG_tr,
                self.LAST_PAST_POINT_tr,
                self.OTHER_PAST_TRAJECTORY_IMAGES_tr,
                self.OTHER_FUTURE_TRAJECTORY_IMAGES_tr
            ]
        elif type == 'test':
            input_fetch = [
                self.PAST_TRAJECTORY_IMAGES_te,
                self.FUTURE_TRAJECTORY_IMAGES_te,
                self.FUTURE_TRAJECTORY_POINT_IMAGES_te,
                self.FUTURE_TRAJECTORY_POINTS_te,
                self.TYPE_te,
                self.W_te,
                self.H_te,
                self.REF_IMG_te,
                self.LAST_PAST_POINT_te,
                self.OTHER_PAST_TRAJECTORY_IMAGES_te,
                self.OTHER_FUTURE_TRAJECTORY_IMAGES_te,
            ]
            n = 1000000
        else:
            input_fetch = [
                self.PAST_TRAJECTORY_IMAGES_u,
                self.FUTURE_TRAJECTORY_IMAGES_u,
                self.FUTURE_TRAJECTORY_POINT_IMAGES_u,
                self.FUTURE_TRAJECTORY_POINTS_u,
                self.TYPE_u,
                self.W_u,
                self.H_u,
                self.REF_IMG_u,
                self.LAST_PAST_POINT_u,
                self.OTHER_PAST_TRAJECTORY_IMAGES_u,
                self.OTHER_FUTURE_TRAJECTORY_IMAGES_u
            ]
        counter = 0
        L = 0
        convergence_list = []
        y_hat_dataset = []
        y_dataset = []
        while (counter < n):
            print("{} / {}".format(counter, n))
            try:
                (
                    PAST_TRAJECTORY_IMAGES,
                    FUTURE_TRAJECTORY_IMAGES,
                    FUTURE_TRAJECTORY_POINT_IMAGES,
                    FUTURE_TRAJECTORY_POINTS,
                    TYPE,
                    W,
                    H,
                    REF_IMAGE,
                    LAST_PAST_POINT,
                    OTHER_PAST_TRAJECTORY_IMAGES,
                    OTHER_FUTURE_TRAJECTORY_IMAGES,
                ) = self.ses.run(input_fetch)
            except:
                break
            length = len(PAST_TRAJECTORY_IMAGES)
            feed_dict = {
                self.PAST_TRAJECTORY_IMAGES_placeholder: PAST_TRAJECTORY_IMAGES,
                self.FUTURE_TRAJECTORY_IMAGES_placeholder: FUTURE_TRAJECTORY_IMAGES,
                self.FUTURE_TRAJECTORY_POINT_IMAGES_placeholder: FUTURE_TRAJECTORY_POINT_IMAGES,
                self.FUTURE_TRAJECTORY_POINTS_placeholder: FUTURE_TRAJECTORY_POINTS,
                self.OBJECT_TYPE_placeholder: TYPE,
                self.W_placeholder: W,
                self.H_placeholder: H,
                self.REF_IMAGE_placeholder: REF_IMAGE,
                self.LAST_PAST_POINT_placeholder: LAST_PAST_POINT,
                self.training_phase: False,
                self.OTHER_PAST_TRAJECTORY_IMAGES_placeholder: OTHER_PAST_TRAJECTORY_IMAGES,
                self.OTHER_FUTURE_TRAJECTORY_IMAGES_placeholder: OTHER_FUTURE_TRAJECTORY_IMAGES
            }
            l, y_hat = self.ses.run([self.ce_loss_batch, self.guided_generated_positions_tensor],
                                    feed_dict=feed_dict)
            y_hat_dataset.append(y_hat)
            y_dataset.append(FUTURE_TRAJECTORY_POINTS)
            convergence_list.append(l)
            counter += length
        convergence_list = np.concatenate(convergence_list)
        y_hat_dataset = np.concatenate(y_hat_dataset)
        y_dataset = np.concatenate(y_dataset)
        L = np.mean(convergence_list)
        error = np.std(convergence_list) / np.sqrt(counter)
        print("N = {}\nLikelihood on {} Split = {} +- {}".format(counter, self.SPLIT_ID, L, error))
        return y_hat_dataset, y_dataset

    def unit_test_cnn(self):
        A = self.ses.run([self.TEST1, self.TEST2])
        print(A[1])
        print(A[0])

    def get_n_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Number of Parameters in model = {}".format(total_parameters))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-n", "--name", dest="NAME", default="SDD",
                        help="BATCH SIZE")
    parser.add_argument("-b", "--batch_size", dest="BATCH_SIZE", default=1,
                        help="BATCH SIZE")
    parser.add_argument("-ilr", "--initial_lr", dest="INITIAL_LR", default=0.005,
                        help="Initial learning rate")
    parser.add_argument("-lrd", "--lr_decay", dest="DECAY_STEP", default=300,
                        help="Number of steps to halve the learning rate")
    parser.add_argument("-blur_std", "--blur_std", dest="BLUR_STD", default=0.0,
                        help="Standard Deviation of generated samples")
    parser.add_argument("-ts", "--total_steps", dest="TOTAL_STEPS", default=100,
                        help="total training steps")
    parser.add_argument("-spid", "--split_id", dest="SPLIT_ID", default=0,
                        help="which split to use for testing")

    parser.add_argument("-STRIDE", "--stride", dest="STRIDE", default=10,
                        help="STRIDE")
    parser.add_argument("-SIZE", "--size", dest="SIZE", default=512,
                        help="Image Size")

    parser.add_argument("-BNR", "--batch_norm_reference", dest="USE_BATCH_NORM_REF_IMG", default="False",
                        help="Whether to use batch normalization for ref image encoder or not")
    parser.add_argument("-OTHER", "--other_objects", dest="USE_OTHER_OBJECTS", default="False",
                        help="Whether to use other objects trajectories or not")
    parser.add_argument("-REF_IMG", "--use_ref_img", dest="USE_REF_IMG", default="False",
                        help="Whether to use reference image or not")
    parser.add_argument("-BNT", "--batch_norm_trajectory", dest="USE_BATCH_NORM_PAST_IMG_ENCODER", default="False",
                        help="Whether to use batch normalization for past encoder or not")

    parser.add_argument("-PTS", "--pre_train_steps", dest="PRE_TRAIN_STEPS", default=0,
                        help="Number of steps to pre-train with maximum likelihood")

    parser.add_argument("-gpu", "--gpu_id", dest="GPU_ID", default=0,
                        help="GPU_ID to use")
    parser.add_argument("-train", "--train", dest="TRAIN", default="True",
                        help="Train or load and evaluate")

    args = parser.parse_args()
    USE_BATCH_NORM_REF_IMG = True if args.USE_BATCH_NORM_REF_IMG == "True" else False
    USE_OTHER_OBJECTS = True if args.USE_OTHER_OBJECTS == "True" else False
    USE_REF_IMG = True if args.USE_REF_IMG == "True" else False
    USE_BATCH_NORM_PAST_IMG_ENCODER = True if args.USE_BATCH_NORM_PAST_IMG_ENCODER == "True" else False

    Temporal_CNN_Multi_Object.NAME = args.NAME
    Temporal_CNN_Multi_Object.TOTAL_STEPS = int(args.TOTAL_STEPS)
    Temporal_CNN_Multi_Object.DECAY_STEP = int(args.DECAY_STEP)
    Temporal_CNN_Multi_Object.INITIAL_LR = float(args.INITIAL_LR)
    Temporal_CNN_Multi_Object.BATCH_SIZE = int(args.BATCH_SIZE)
    Temporal_CNN_Multi_Object.SPLIT_ID = int(args.SPLIT_ID)

    Temporal_CNN_Multi_Object.STRIDE = int(args.STRIDE)
    Temporal_CNN_Multi_Object.SIZE = int(args.SIZE)
    Temporal_CNN_Multi_Object.PRE_TRAIN_STEPS = int(args.PRE_TRAIN_STEPS)
    Temporal_CNN_Multi_Object.BLUR_STD = float(args.BLUR_STD)
    Temporal_CNN_Multi_Object.USE_BATCH_NORM_REF_IMG = USE_BATCH_NORM_REF_IMG
    Temporal_CNN_Multi_Object.USE_OTHER_OBJECTS = USE_OTHER_OBJECTS
    Temporal_CNN_Multi_Object.USE_REF_IMG = USE_REF_IMG
    Temporal_CNN_Multi_Object.USE_BATCH_NORM_PAST_IMG_ENCODER = USE_BATCH_NORM_PAST_IMG_ENCODER

    Temporal_CNN_Multi_Object.GPU_ID = int(args.GPU_ID)
    Temporal_CNN_Multi_Object.DESCRIPTION = str(args)

    TRAIN = True if args.TRAIN == "True" else False
    # TRAIN = False

    if TRAIN:
        A = Temporal_CNN_Multi_Object(args.NAME)
        A.get_n_params()
        A.train()
        A.save()
    else:
        A = Temporal_CNN_Multi_Object(name=args.NAME, summary=False)
        # A.get_n_params()


        # A.init_variables()
        A.load()
        # A.unit_test_cnn()
        # exit()
        # generated_images, ref_images, past_images = A.generate_visual_samples(n=512, K=10, type='test', verbose=True)
        # print(generated_images[0].shape)
        # print(ref_images.shape)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(generated_images[1][1])
        # plt.figure()
        # plt.imshow(generated_images[0][1])
        # plt.figure()
        # plt.imshow(past_images[1])
        # plt.figure()
        # plt.imshow(ref_images[1])
        # plt.show()
        # dataset_dict = {"1": generated_images, "2": ref_images, "3": past_images}
        # np.save("visual_samples_{}.npy".format(A.SPLIT_ID), dataset_dict)
        # exit(0)

        # A.load()
        # y_hat_dataset, y_dataset = A.evaluate_likelihood(30000, 'test')
        y_hat_dataset, y_dataset = A.evaluate_likelihood(30, 'train')
        # print(y_hat_dataset.shape)
        # mse = np.mean(np.sum(np.sum(np.square(128 * (y_hat_dataset - y_dataset)), axis=2), axis=1))
        # mse_error = np.sqrt(np.std(np.sum(np.sum(np.square(128 * (y_hat_dataset - y_dataset)), axis=2), axis=1))) / len(
        #     y_dataset)
        # print("MSSE = ", mse, "+-", mse_error)
        #
        # acu, acu_std = ACU_(128 * y_hat_dataset, 128 * y_dataset, 12)
        # print("Accuracy = ", acu, "+-", acu_std)

        # A.unit_test_cnn()
        # A.load()
        # A.init_variables()
        y_hat_dataset_list, y_dataset, W_dataset, H_dataset = A.generate_oracle_dataset(n=2, K=10,
                                                                                        type='train',
                                                                                        verbose=True)
        result = L2_error_1234_second(y_hat_dataset_list, y_dataset, W_dataset, H_dataset, A.STRIDE, "Min",
                                      down_scale_rate=5, top_percentage=0.1)
        for t in range(4):
            print("Mean L2 error at {} sec = {} +- {}".format(t + 1, result[0][t], result[1][t]))
