import tensorflow as tf
import os
import datetime

from BaseModel import BaseModel
from Mnist_Dataloader import Mnist_Temporal_Image2Image_Dataloader
from Mnist_Metrics import *


class TemporalDCNN(BaseModel):
    NAME = "TemporalDCNN_Mnist"
    SAVE_DIR = "../Mnist_Saved_Models/TemporalDCNN/"

    INITIAL_LENGTH = 10
    BATCH_SIZE = 32
    EPOCH = 10
    BUFFER_SIZE = 4096
    L = 28
    MAX_LEN = 120

    INITIAL_LR = 0.005
    TOTAL_STEPS = 1000
    DECAY_STEP = 200
    GPU_ID = 0
    DESCRIPTION = ""
    H = 32

    SKIP_TYPE = "ADD"  # "ADD" , "NONE"
    USE_BATCH_NORM = False

    def __init__(self, name=NAME, save_dir=SAVE_DIR, summary=True, dataset_type='train'):
        super(TemporalDCNN, self).__init__(name, save_dir)
        self.MODEL_PATH = os.path.join(self.SAVE_DIR, self.NAME)
        self.dataset_type = dataset_type

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

    def down_up_convolution(self, image_sequence, reuse=False):
        with tf.variable_scope("down_convolution"):
            conv1 = tf.layers.conv3d(
                inputs=image_sequence,
                # filters=self.H,
                filters=32,
                # kernel_size=[3, 3, self.INITIAL_LENGTH],
                kernel_size=[3, 3, 1],
                strides=[2, 2, 1],
                padding="same",
                activation=tf.nn.relu,
                name='conv1',
                reuse=reuse
            )
            # print(conv1.shape)
            conv1_batch_norm = tf.layers.batch_normalization(conv1, axis=-1,
                                                             training=self.training_phase,
                                                             name="conv1_batch_norm", reuse=reuse)
            conv2_input = conv1_batch_norm if self.USE_BATCH_NORM else conv1
            conv2 = tf.layers.conv3d(
                inputs=conv2_input,
                # filters=self.H * 2,
                filters=64,
                # kernel_size=[3, 3, 1],
                kernel_size=[3, 3, self.INITIAL_LENGTH],
                strides=[2, 2, 1],
                padding="same",
                activation=tf.nn.relu,
                name='conv2',
                reuse=reuse
            )
            # print(conv2.shape)
            conv2_batch_norm = tf.layers.batch_normalization(conv2, axis=-1,
                                                             training=self.training_phase,
                                                             name="conv2_batch_norm", reuse=reuse)

            conv3_input = conv2_batch_norm if self.USE_BATCH_NORM else conv2
            conv3 = tf.layers.conv3d(
                inputs=conv3_input,
                # filters=self.H * 4,
                filters=128,
                kernel_size=[3, 3, 1],
                # kernel_size=[3, 3, self.INITIAL_LENGTH],
                strides=[2, 2, 1],
                padding="same",
                activation=tf.nn.relu,
                name='conv3',
                reuse=reuse
            )
            # print(conv3.shape)
            conv3_batch_norm = tf.layers.batch_normalization(conv3, axis=-1,
                                                             training=self.training_phase,
                                                             name="conv3_batch_norm", reuse=reuse)
            conv4_input = conv3_batch_norm if self.USE_BATCH_NORM else conv3
            conv4 = tf.layers.conv3d(
                inputs=conv4_input,
                # filters=self.H * 8,
                filters=196,
                kernel_size=[3, 3, 1],
                # kernel_size=[3, 3, self.INITIAL_LENGTH],
                strides=[2, 2, 1],
                padding="same",
                activation=tf.nn.relu,
                name='conv4',
                reuse=reuse
            )
            # print(conv4.shape)
            conv4_batch_norm = tf.layers.batch_normalization(conv4, axis=-1,
                                                             training=self.training_phase,
                                                             name="conv4_batch_norm", reuse=reuse)
            # print(conv4_batch_norm.shape)

        with tf.variable_scope("up_convolution"):
            deconv1 = tf.layers.conv3d_transpose(
                inputs=conv4_batch_norm,
                # filters=self.H * 4,
                filters=128,
                kernel_size=[3, 3, 1],
                strides=[2, 2, 1],
                padding='same',
                activation=tf.nn.relu,
                name='deconv1',
                reuse=reuse
            )
            # print(deconv1.shape)

            deconv1_batch_norm = tf.layers.batch_normalization(deconv1, axis=-1,
                                                               training=self.training_phase,
                                                               name="deconv1_batch_norm", reuse=reuse)

            deconv1_output = deconv1_batch_norm if self.USE_BATCH_NORM else deconv1
            if self.SKIP_TYPE == 'NONE':
                deconv2_input = deconv1_output
            elif self.SKIP_TYPE == "ADD":
                deconv2_input = deconv1_output + conv4_input
            else:
                deconv2_input = tf.concat([deconv1_output, conv4_input], axis=4)

            deconv2 = tf.layers.conv3d_transpose(
                inputs=deconv2_input,
                # filters=self.H * 2,
                filters=64,
                kernel_size=[3, 3, 1],
                strides=[2, 2, 1],
                padding='same',
                name='deconv2',
                reuse=reuse
            )
            # print(deconv2.shape)
            deconv2_batch_norm = tf.layers.batch_normalization(deconv2, axis=-1,
                                                               training=self.training_phase,
                                                               name="deconv2_batch_norm", reuse=reuse)
            deconv2_output = deconv2_batch_norm if self.USE_BATCH_NORM else deconv2
            if self.SKIP_TYPE == 'NONE':
                deconv3_input = deconv2_output
            elif self.SKIP_TYPE == "ADD":
                deconv3_input = deconv2_output + conv3_input
            else:
                deconv3_input = tf.concat([deconv2_output, conv3_input], axis=4)

            deconv3 = tf.layers.conv3d_transpose(
                inputs=deconv3_input,
                # filters=self.H,
                filters=32,
                kernel_size=[3, 3, 1],
                strides=[2, 2, 1],
                padding='same',
                name='deconv3',
                reuse=reuse
            )
            # print(deconv3.shape)
            deconv3_batch_norm = tf.layers.batch_normalization(deconv3, axis=-1,
                                                               training=self.training_phase,
                                                               name="deconv3_batch_norm", reuse=reuse)
            deconv3_output = deconv3_batch_norm if self.USE_BATCH_NORM else deconv3
            if self.SKIP_TYPE == 'NONE':
                deconv4_input = deconv3_output
            elif self.SKIP_TYPE == "ADD":
                deconv4_input = deconv3_output + conv2_input
            else:
                deconv4_input = tf.concat([deconv3_output, conv2_input], axis=4)

            deconv4 = tf.layers.conv3d_transpose(
                inputs=deconv4_input,
                filters=1,
                kernel_size=[3, 3, 1],
                strides=[2, 2, 1],
                padding='same',
                name='deconv4',
                reuse=reuse
            )
            # print(deconv4.shape)
            return deconv4

    def init_graph(self, summary=True):
        with tf.name_scope(self.NAME):
            with tf.name_scope("Dataset"):
                with tf.name_scope("Dataloader"):
                    self.Dataloader = Mnist_Temporal_Image2Image_Dataloader(self.dataset_type, dataset_dir="../Dataset/Mnist/",
                    # self.Dataloader = Mnist_Temporal_Image2Image_Dataloader('test', dataset_dir="../Dataset/Mnist/",
                                                                            batch_size=self.BATCH_SIZE,
                                                                            buffer_size=self.BUFFER_SIZE,
                                                                            initial_length=self.INITIAL_LENGTH,
                                                                            max_len=self.MAX_LEN)
                    self.X, self.Y, self.LEN = self.Dataloader.get_batch
                    self.X_v, self.Y_v, self.LEN_v = self.Dataloader.get_validation_batch

                with tf.name_scope("Placeholders"):
                    self.X_placeholder = tf.placeholder_with_default(self.X,
                                                                     shape=[self.BATCH_SIZE, 28, 28, self.MAX_LEN],
                                                                     name='X_placeholder')
                    self.Y_placeholder = tf.placeholder_with_default(self.Y,
                                                                     shape=[self.BATCH_SIZE, 28, 28, self.MAX_LEN],
                                                                     name='Y_placeholder')
                    self.LEN_placeholder = tf.placeholder_with_default(self.LEN, shape=[self.BATCH_SIZE],
                                                                       name='LEN_placeholder')
                    self.LEN_placeholder_float = tf.cast(self.LEN_placeholder, tf.float32, name='LEN_placeholder_float')

                    self.training_phase = tf.placeholder_with_default(True, shape=[])

            with tf.variable_scope("Inference"):
                self.mask = tf.sequence_mask(self.LEN_placeholder, self.MAX_LEN, dtype=tf.float32, name='mask')
                self.padded_image = tf.pad(self.X_placeholder, [[0, 0], [2, 2], [2, 2], [0, 0]], 'CONSTANT',
                                           name='padded_image')

                self.padded_image_reshaped = self.padded_image[:, :, :, :, tf.newaxis]

                # print(self.padded_image_reshaped.shape)

                deconv4 = self.down_up_convolution(self.padded_image_reshaped, reuse=False)
                first_deconv4 = self.down_up_convolution(self.padded_image_reshaped[:, :, :, :self.INITIAL_LENGTH, :],
                                                         reuse=True)

                if self.INITIAL_LENGTH == 1:
                    self.first_logits = first_deconv4[:, 2:-2, 2:-2,
                                        (self.INITIAL_LENGTH - 1) // 2:,
                                        0]
                    self.logits = deconv4[:, 2:-2, 2:-2, (self.INITIAL_LENGTH - 1) // 2:,
                                  0]
                else:
                    self.first_logits = first_deconv4[:, 2:-2, 2:-2,
                                        (self.INITIAL_LENGTH - 1) // 2: -(self.INITIAL_LENGTH // 2),
                                        0]
                    self.logits = deconv4[:, 2:-2, 2:-2, (self.INITIAL_LENGTH - 1) // 2: -(self.INITIAL_LENGTH // 2),
                                  0]
                # print("Here", self.first_logits.shape)

                self.logits_padded = tf.pad(self.logits, [[0, 0], [0, 0], [0, 0], [0, self.INITIAL_LENGTH - 1]],
                                            'CONSTANT',
                                            name='logits_padded')

                self.logits_reshaped = tf.reshape(self.logits_padded, [-1, self.L * self.L, self.MAX_LEN],
                                                  name='logits_reshaped')
                self.first_logits_reshaped = tf.reshape(self.first_logits, [-1, self.L * self.L])

                # print("Here", self.logits_reshaped.shape)

                self.log_prob_sequence = tf.nn.log_softmax(self.logits_reshaped, dim=1, name='log_prob_sequence')
                self.first_log_prob = tf.nn.log_softmax(self.first_logits_reshaped, name='first_log_prob')

                self.argmax_pixel = tf.argmax(self.logits_reshaped, axis=1)
                self.sampled_pixel = tf.reshape(tf.multinomial(
                    tf.reshape(tf.transpose(self.logits_reshaped, [0, 2, 1]), [-1, self.L * self.L]), 1),
                    [-1, self.MAX_LEN])
                self.first_sampled_pixel = tf.multinomial(self.first_logits_reshaped, 1)

                # y_p : y_predicted
                self.y_p = tf.floor(self.argmax_pixel / self.L, "predicted_y")
                self.x_p = tf.mod(self.argmax_pixel, self.L, "predicted_x")

                # y_s : y_sampled
                self.y_s = tf.floor(self.sampled_pixel / self.L, "sampled_y")
                self.x_s = tf.mod(self.sampled_pixel, self.L, "sampled_x")

                self.y_s_first = tf.floor(self.first_sampled_pixel / self.L, "sampled_y_first")
                self.x_s_first = tf.mod(self.first_sampled_pixel, self.L, "sampled_x_first")

                self.Y_placeholder_reshaped = tf.reshape(self.Y_placeholder, [-1, self.L * self.L, self.MAX_LEN],
                                                         name='Y_placeholder_reshaped')
                true_pixel = tf.argmax(self.Y_placeholder_reshaped, axis=1)

                self.y_t = tf.floor(true_pixel / self.L, "true_y")
                self.x_t = tf.mod(true_pixel, self.L, "true_x")

                self.distance = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(
                            tf.sqrt(
                                tf.square(tf.cast(self.x_t - self.x_p, dtype=tf.float32)) + tf.square(
                                    tf.cast(self.y_t - self.y_p, dtype=tf.float32))),
                            self.mask),
                        axis=1) / self.LEN_placeholder_float,
                    axis=0, name='distance')

                self.accuracy = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(
                            tf.cast(
                                tf.equal(self.argmax_pixel, true_pixel)
                                , tf.float32),
                            self.mask),
                        axis=1) / self.LEN_placeholder_float
                    , axis=0, name='accuracy')

            with tf.name_scope("Optimization"):
                self.negative_log_probs = -tf.nn.log_softmax(self.logits_reshaped, dim=1, name='log_probs')
                self.sequence_negative_log_likelihood = tf.reduce_sum(
                    tf.multiply(self.Y_placeholder_reshaped, self.negative_log_probs),
                    axis=1,
                    name='sequecne_likelihood')
                # print(self.sequence_negative_log_likelihood.shape)
                self.masked_sequence_negative_log_likelihood = tf.multiply(self.sequence_negative_log_likelihood,
                                                                           self.mask,
                                                                           name='masked_sequence_likelihood')
                self.negative_log_likelihood = tf.reduce_mean(
                    tf.reduce_sum(self.masked_sequence_negative_log_likelihood, axis=1), axis=0,
                    name='negative_log_likelihood')

                self.consistent_negative_log_likelihood = tf.reduce_mean(
                    tf.reduce_sum(self.masked_sequence_negative_log_likelihood[:, 10 - self.INITIAL_LENGTH:], axis=1),
                    axis=0,
                    name='negative_log_likelihood')

                self.global_step = tf.train.get_or_create_global_step()
                self.learning_rate = tf.train.exponential_decay(self.INITIAL_LR, self.global_step, self.DECAY_STEP, 0.5,
                                                                name='learning_rate')
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_operation = self.optimizer.minimize(self.negative_log_likelihood,
                                                                   global_step=self.global_step)

            self.init_node = tf.global_variables_initializer()
            self.save_node = tf.train.Saver()

        if summary:
            self.nll_summary = tf.summary.scalar(name='loss', tensor=self.negative_log_likelihood)
            self.consistent_nll_summary = tf.summary.scalar(name='consistent nll',
                                                            tensor=self.consistent_negative_log_likelihood)
            self.accuracy_summary = tf.summary.scalar(name="accuracy", tensor=self.accuracy)
            self.distance_summary = tf.summary.scalar(name="distance", tensor=self.distance)
            self.scalar_summaries = tf.summary.merge(
                [self.nll_summary, self.accuracy_summary, self.distance_summary, self.consistent_nll_summary])

            # self.input_image_summary = tf.summary.image(name="input_image", tensor=self.input_image, max_outputs=3)
            # self.output_image_summary = tf.summary.image(name="output_image", tensor=self.output_image,
            #                                              max_outputs=3)
            # self.image_summary = tf.summary.merge([self.input_image_summary, self.output_image_summary])

            self.merged_summary = tf.summary.merge_all()

            self.summary_writer = tf.summary.FileWriter(self.LOG_PATH, tf.get_default_graph())
            self.validation_summary_writer = tf.summary.FileWriter(self.LOG_PATH + "validation")

    def add_code_summary(self):
        code_string = "\n".join(open('TemporalDCNN.py', 'r').readlines())
        text_tensor = tf.make_tensor_proto(self.DESCRIPTION + "\n\n" + code_string, dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        summary.value.add(tag="Hyper parameters", metadata=meta, tensor=text_tensor)
        self.summary_writer.add_summary(summary)

    def init_variables(self):
        self.ses.run(self.init_node)

    def train(self):
        self.init_variables()
        self.add_code_summary()
        for step in range(self.TOTAL_STEPS):

            _, scalar_summaries = self.ses.run([self.train_operation, self.scalar_summaries])
            self.summary_writer.add_summary(scalar_summaries, step)
            # if step % 100 == 0:
            # image_summary = self.ses.run(self.image_summary)
            # self.summary_writer.add_summary(image_summary, step)
            if step % 100 == 0:
                print("Step {}".format(step))
            if step % 50 == 0:
                X, Y, LEN = self.ses.run([self.X_v, self.Y_v, self.LEN_v])
                feed_dict = {self.X_placeholder: X, self.Y_placeholder: Y, self.LEN_placeholder: LEN,
                             self.training_phase: False}
                # validation_scalar_summaries, image_summary = self.ses.run([self.scalar_summaries, self.image_summary],
                validation_scalar_summaries = self.ses.run(self.scalar_summaries,
                                                           feed_dict=feed_dict)
                self.validation_summary_writer.add_summary(validation_scalar_summaries, step)
                # self.validation_summary_writer.add_summary(image_summary, step)

    def save(self):
        self.save_node.save(self.ses, save_path=self.MODEL_PATH + "/" + self.NAME + '.ckpt')

    def load(self):
        self.save_node.restore(self.ses, save_path=self.MODEL_PATH + "/" + self.NAME + '.ckpt')

    def get_n_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Number of Parameters in model = {}".format(total_parameters))

    def generate_dataset(self, n=1000, type='validation'):
        y_hat_dataset = []
        y_dataset = []
        len_dataset = []
        counter = 0
        if type == 'validation':
            input_fetch = [self.X_v, self.Y_v, self.LEN_v]
        else:
            input_fetch = [self.X, self.Y, self.LEN]
        print("generating dataset")
        while counter < n:
            try:
                X, Y, LEN = self.ses.run(input_fetch)
            except:
                break
            feed_dict = {self.X_placeholder: X, self.Y_placeholder: Y, self.LEN_placeholder: LEN,
                         self.training_phase: False}
            x_t, y_t, x_p, y_p = self.ses.run([self.x_t, self.y_t, self.x_p, self.y_p], feed_dict=feed_dict)
            y_hat = np.concatenate([x_p[:, :, np.newaxis], y_p[:, :, np.newaxis]], axis=2)
            y = np.concatenate([x_t[:, :, np.newaxis], y_t[:, :, np.newaxis]], axis=2)
            y_hat_dataset.append(y_hat)
            y_dataset.append(y)
            len_dataset.append(LEN)
            counter += len(X)
            # print(counter)
        y_hat_dataset = np.concatenate(y_hat_dataset, axis=0)
        y_dataset = np.concatenate(y_dataset, axis=0)
        len_dataset = np.concatenate(len_dataset, axis=0)
        return y_hat_dataset, y_dataset, len_dataset

    def unit_test(self):
        X, Y, LEN = self.ses.run([self.X_v, self.Y_v, self.LEN_v])
        feed_dict = {self.X_placeholder: X, self.Y_placeholder: Y, self.LEN_placeholder: LEN,
                     self.training_phase: False}
        l1 = self.ses.run(self.logits[0, :, :, 0], feed_dict=feed_dict)
        X[:, :, :, self.INITIAL_LENGTH:] = 1000000
        # X[:, :, :, :-1] = X[:, :, :, 1:]
        feed_dict = {self.X_placeholder: X, self.Y_placeholder: Y, self.LEN_placeholder: LEN,
                     self.training_phase: False}
        l2 = self.ses.run(self.logits[0, :, :, 0], feed_dict=feed_dict)
        # print(np.max(l1 - l2))
        if np.max(l1 - l2) < 1e-6:
            print("Unit test passed")

    def evaluate_likelihood(self, n=10000, type='validation'):
        counter = 0
        print("Evaluating nll")
        convergence_list = []
        L = 0
        if type == "validation":
            input_fetch = [self.X_v, self.Y_v, self.LEN_v]
        else:
            input_fetch = [self.X, self.Y, self.LEN]
        while counter < n:
            try:
                X, Y, LEN = self.ses.run(input_fetch)
            except:
                break
            feed_dict = {self.X_placeholder: X, self.Y_placeholder: Y, self.LEN_placeholder: LEN,
                         self.training_phase: False}
            # l = self.ses.run(self.negative_log_likelihood, feed_dict=feed_dict)
            l = self.ses.run(self.consistent_negative_log_likelihood, feed_dict=feed_dict)

            L += len(X) * l
            counter += len(X)
            convergence_list.append(l)
        L /= counter
        print("Likelihood  = {} +- {}".format(L, np.std(convergence_list) / np.sqrt(counter)))

    def generate_batch_sequences(self, X, LEN):
        X_G = X.copy()
        heat_map_list = []
        X_G[:, :, :, self.INITIAL_LENGTH:] = 0

        log_prob = [[] for _ in range(len(X))]
        for i in range(self.INITIAL_LENGTH, self.MAX_LEN, 1):
            X_feed = X_G.copy()
            X_feed[:, :, :, :self.INITIAL_LENGTH] = X_G[:, :, :, i - self.INITIAL_LENGTH: i]
            feed_dict = {self.X_placeholder: X_feed, self.training_phase: False}
            x_s, y_s, lp = self.ses.run([self.x_s_first[:, 0], self.y_s_first[:, 0], self.first_log_prob],
                                        feed_dict=feed_dict)
            x_s = np.array(x_s, dtype=np.int32)
            y_s = np.array(y_s, dtype=np.int32)
            X_G[:, :, :, i] = X_G[:, :, :, i - 1]
            heat_map = np.reshape(np.exp(lp), [-1, self.L, self.L])
            heat_map_list.append(heat_map)
            # flag = False
            for j in range(len(x_s)):
                # if i - self.INITIAL_LENGTH < LEN[j]:
                # flag = True
                X_G[j, y_s[j], x_s[j], i] = 1
                # log_prob[j] += lp[j, y_s[j] * self.L + x_s[j]]
                log_prob[j].append(lp[j, y_s[j] * self.L + x_s[j]])
            # if not flag:
            #     break
        heat_map = np.transpose(np.stack(heat_map_list), [1, 2, 3, 0])
        log_prob = np.array(log_prob)
        return X_G, log_prob, heat_map

    def generate_sequences(self, n=1000):
        counter = 0
        X_dataset = []
        X_G_dataset = []
        LEN_dataset = []
        Likelihood_dataset = []
        Heat_map_dataset = []
        while counter < n:
            print(counter)
            X, LEN = A.ses.run([A.X_v, A.LEN_v])
            X_G, log_prob, heat_map = A.generate_batch_sequences(X, LEN)
            counter += len(X)
            X_dataset.append(X)
            LEN_dataset.append(LEN)
            X_G_dataset.append(X_G)
            Likelihood_dataset.append(log_prob)
            Heat_map_dataset.append(heat_map)

        X_dataset = np.concatenate(X_dataset)
        X_G_dataset = np.concatenate(X_G_dataset)
        LEN_dataset = np.concatenate(LEN_dataset)
        Likelihood_dataset = np.concatenate(Likelihood_dataset)
        Heat_map_dataset = np.concatenate(Heat_map_dataset)
        return X_dataset, X_G_dataset, LEN_dataset, Likelihood_dataset, Heat_map_dataset

    def generate_sequence_with_same_initial_segment(self, n=1024, k=16):
        assert self.BATCH_SIZE % k == 0
        counter = 0
        X_dataset = []
        X_G_dataset = []
        LEN_dataset = []
        Likelihood_dataset = []
        Heat_map_dataset = []
        X, LEN = A.ses.run([A.X_v, A.LEN_v])
        for i in range(0, self.BATCH_SIZE, k):
            X[i: i + k] = X[:k]
            LEN[i: i + k] = LEN[:k]

        while counter < n:
            print(counter)
            X_G, log_prob, heat_map = A.generate_batch_sequences(X, LEN)
            counter += len(X)
            X_dataset.append(X)
            LEN_dataset.append(LEN)
            X_G_dataset.append(X_G)
            Likelihood_dataset.append(log_prob)
            Heat_map_dataset.append(heat_map)

        X_dataset = np.concatenate(X_dataset)
        X_G_dataset = np.concatenate(X_G_dataset)
        LEN_dataset = np.concatenate(LEN_dataset)
        Likelihood_dataset = np.concatenate(Likelihood_dataset)
        Heat_map_dataset = np.concatenate(Heat_map_dataset)
        return X_dataset, X_G_dataset, LEN_dataset, Likelihood_dataset, Heat_map_dataset


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-n", "--name", dest="NAME", default="Mnist",
                        help="BATCH SIZE")
    parser.add_argument("-b", "--batch_size", dest="BATCH_SIZE", default=2,
                        help="BATCH SIZE")
    parser.add_argument("-hid", "--hidden", dest="H", default=32,
                        help="Hidden Layer size")
    parser.add_argument("-ilr", "--initial_lr", dest="INITIAL_LR", default=0.005,
                        help="Initial learning rate")
    parser.add_argument("-lrd", "--lr_decay", dest="DECAY_STEP", default=1000,
                        help="Number of steps to halve the learning rate")
    parser.add_argument("-il", "--initial_length", dest="INITIAL_LENGTH", default=10,
                        help="Initial Length")
    parser.add_argument("-ts", "--total_steps", dest="TOTAL_STEPS", default=200,
                        help="total training steps")
    parser.add_argument("-gpu", "--gpu_id", dest="GPU_ID", default=0,
                        help="GPU_ID to use")
    parser.add_argument("-skip", "--skip", dest="SKIP_TYPE", default="CONCAT",
                        help="skip type")

    parser.add_argument("-BNR", "--BNR", dest="BNR", default="False",
                        help="GPU_ID to use")
    parser.add_argument("-train", "--train", dest="TRAIN", default="True",
                        help="Train or load and evaluate")

    args = parser.parse_args()
    TemporalDCNN.NAME = args.NAME
    TemporalDCNN.TOTAL_STEPS = int(args.TOTAL_STEPS)
    TemporalDCNN.DECAY_STEP = int(args.DECAY_STEP)
    TemporalDCNN.INITIAL_LR = float(args.INITIAL_LR)
    TemporalDCNN.BATCH_SIZE = int(args.BATCH_SIZE)
    TemporalDCNN.H = int(args.H)
    TemporalDCNN.INITIAL_LENGTH = int(args.INITIAL_LENGTH)
    TemporalDCNN.GPU_ID = int(args.GPU_ID)
    TemporalDCNN.DESCRIPTION = str(args)
    TemporalDCNN.SKIP_TYPE = str(args.SKIP_TYPE)
    TRAIN = True if args.TRAIN == "True" else False
    TemporalDCNN.USE_BATCH_NORM = True if args.BNR == "True" else False

    TRAIN = False
    if TRAIN:
        # A = TemporalDCNN(args.NAME, summary=False)
        A = TemporalDCNN(args.NAME)
        # A.get_n_params()
        A.init_variables()

        A.unit_test()
        A.train()
        A.save()
    else:
        A = TemporalDCNN(name=args.NAME, summary=False, dataset_type='test')
        A.load()
        # X, X_G, LEN, likelihood, heat_map = A.generate_sequences(1024)
        # X, X_G, LEN, likelihood, heat_map = A.generate_sequence_with_same_initial_segment(1024)
        # generated_sequences = {"X": X, "X_G": X_G, "LEN": LEN, "likelihood": likelihood, "heatmap": heat_map}
        # np.save(A.MODEL_PATH + "generated_sequences_same_initial_segment_1024.npy", generated_sequences)
        # np.save("generated_sequences_same_initial_segment_1024.npy", generated_sequences)
        # animate_X_X_G(X[:16], X_G[:16], LEN[:16], 10, 100, "../Animations/1.mp4")

        y_hat_dataset, y_dataset, len_dataset = A.generate_dataset(100000, type='test')
        msse, msse_std = MSSE_(y_hat_dataset, y_dataset, len_dataset, A.INITIAL_LENGTH)
        mase, mase_std = MASE_(y_hat_dataset, y_dataset, len_dataset, A.INITIAL_LENGTH)
        acu, acu_std = ACU_(y_hat_dataset, y_dataset, len_dataset, A.INITIAL_LENGTH)
        print("Mean Sum Squared Error = ", msse, "+-", msse_std)
        print("Mean Average Squared Error = ", mase, "+-", mase_std)
        print("Accuracy = ", acu, "+-", acu_std)
        # A.evaluate_likelihood(100000, type='test')

# Mean Sum Squared Error =  43.737548828125 +- 3.4140489226873476
# Mean Average Squared Error =  1.5525705313833915 +- 0.12409194764058218
# Accuracy =  0.635978629307323 +- 0.0028545166880132545
# Likelihood on validation dataset = 29.214286267757416
