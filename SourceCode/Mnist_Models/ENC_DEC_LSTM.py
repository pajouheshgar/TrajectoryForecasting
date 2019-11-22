import tensorflow as tf
import os
import datetime
from BaseModel import BaseModel
from Mnist_Dataloader import Mnist_Absolute_Sequence_Dataloader
from Mnist_Metrics import *


class ENC_DEC_LSTM(BaseModel):
    # NAME = "LSTM_Oracle"
    NAME = "Mnist"
    SAVE_DIR = "../Mnist_Saved_Models/ENC_DEC_LSTM/"
    RS = 42

    BATCH_SIZE = 64
    EPOCH = 10
    BUFFER_SIZE = 8192
    L = 28
    INITIAL_LR = 0.001
    TOTAL_STEPS = 5000

    MAX_LEN = 120
    N_H_ENCODER_LSTM = 256
    N_H_DECODER_LSTM = 256
    N_H_X_EMBEDDING = 200
    N_H_OUTPUT_EMBEDDING = 200

    STD = 1.0
    INITIAL_LENGTH = 10
    GPU_ID = 0
    DESCRIPTION = ""

    def __init__(self, name=NAME, save_dir=SAVE_DIR, summary=True):
        super(ENC_DEC_LSTM, self).__init__(name, save_dir)
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

    def init_graph(self, summary=True):
        np.random.seed(self.RS)
        tf.set_random_seed(self.RS)
        with tf.variable_scope(self.NAME):
            with tf.name_scope("Dataset"):
                with tf.name_scope("Dataloader"):
                    self.Dataloader = Mnist_Absolute_Sequence_Dataloader('train', dataset_dir="../Dataset/Mnist/",
                    # self.Dataloader = Mnist_Absolute_Sequence_Dataloader('test', dataset_dir="../Dataset/Mnist/",
                                                                         batch_size=self.BATCH_SIZE,
                                                                         buffer_size=self.BUFFER_SIZE,
                                                                         max_len=self.MAX_LEN,
                                                                         initial_length=self.INITIAL_LENGTH, sep=True)
                    # self.Dataloader = Oracle_Dataloader('train',
                    #                                     dataset_dir="../Mnist_Saved_Models/Oracle/Oracle1/dataset.npy",
                    #                                     batch_size=self.BATCH_SIZE,
                    #                                     buffer_size=self.BUFFER_SIZE)
                    self.MAX_LEN = self.Dataloader.MAX_LEN
                    self.X, self.Y, self.LEN = self.Dataloader.get_batch
                    self.X_v, self.Y_v, self.LEN_v = self.Dataloader.get_validation_batch

            with tf.name_scope("Placeholders"):
                self.X_placeholder = tf.placeholder_with_default(self.X, shape=[None, self.INITIAL_LENGTH, 2], name='X')
                self.Y_placeholder = tf.placeholder_with_default(self.Y, shape=[None, self.MAX_LEN, 2], name='Y')
                self.LEN_placeholder = tf.placeholder_with_default(self.LEN, shape=[None], name='LEN')
                self.INITIAL_LEN_placeholder = tf.placeholder_with_default(self.INITIAL_LENGTH, shape=[],
                                                                           name='INITIAL_LENGTH')

                # self.training_phase = tf.placeholder_with_default(True, shape=[])

            with tf.variable_scope("Inference"):
                self.mask = tf.sequence_mask(self.LEN_placeholder, self.MAX_LEN, dtype=tf.float32, name='mask')
                self.encoder_input = tf.layers.dense(self.X_placeholder, self.N_H_X_EMBEDDING, tf.nn.relu,
                                                 name='embeded_X')
                self.decoder_input = tf.concat([self.X_placeholder[:, -1:, :], self.Y_placeholder[:, :-1, :]],
                                               axis=1, name='decoder_input')
                self.embeded_X = tf.layers.dense(self.decoder_input, self.N_H_X_EMBEDDING, tf.nn.relu,
                                                 name='embeded_X', reuse=True)
                with tf.variable_scope("Encoder"):
                    self.encoder_cell = tf.nn.rnn_cell.LSTMCell(self.N_H_ENCODER_LSTM)

                    # encoder_output, encoder_last_state = tf.nn.dynamic_rnn(self.encoder_cell, self.X_placeholder,
                    encoder_output, encoder_last_state = tf.nn.dynamic_rnn(self.encoder_cell, self.encoder_input,
                                                                           self.INITIAL_LEN_placeholder * tf.ones_like(
                                                                               self.LEN_placeholder), dtype=tf.float32)

                with tf.variable_scope("Decoder"):
                    self.decoder_cell = tf.nn.rnn_cell.LSTMCell(self.N_H_DECODER_LSTM)

                    self.decoder_output, decoder_last_state = tf.nn.dynamic_rnn(self.decoder_cell, self.decoder_input,
                                                                                self.LEN_placeholder,
                                                                                encoder_last_state)

                    self.embeded_rnn_output = tf.layers.dense(self.decoder_output, self.N_H_OUTPUT_EMBEDDING,
                                                              activation=tf.nn.relu, name='embeded_rnn_output')

                    self.guided_generated_positions = tf.layers.dense(self.embeded_rnn_output, 2,
                                                                      name='next_position') + self.decoder_input
                self.discrete_guided_generated_positions = tf.round(self.guided_generated_positions,
                                                                    name='discrete_guided_generated_positions')

                self.generated_index = self.discrete_guided_generated_positions[:, :,
                                       1] * self.L + self.discrete_guided_generated_positions[:, :, 0]
                self.true_index = self.Y_placeholder[:, :,
                                  1] * self.L + self.Y_placeholder[:, :, 0]

                self.accuracy = tf.reduce_mean(
                    tf.reduce_sum(tf.multiply(tf.cast(tf.equal(self.generated_index, self.true_index), tf.float32),
                                              self.mask), axis=1) / tf.cast(self.LEN_placeholder,
                                                                            tf.float32))

                # with tf.name_scope("Sequence_Generation"):
                #     outputs, last_state = tf.nn.dynamic_rnn(self.cell,
                #                                             self.embeded_XV[:, :self.INITIAL_LEN_placeholder, :],
                #                                             self.INITIAL_LEN_placeholder * tf.ones_like(
                #                                                 self.LEN_placeholder),
                #                                             dtype=tf.float32)
                #     embeded_outputs = tf.layers.dense(outputs, self.N_H_OUTPUT_EMBEDDING, activation=tf.nn.relu,
                #                                       name='embeded_rnn_output', reuse=True)
                #     self.generated_initial_positions = tf.layers.dense(embeded_outputs, 2, name='next_position',
                #                                                        reuse=True) + self.X_placeholder[:,
                #                                                                      :self.INITIAL_LEN_placeholder, :]
                #     last_position = self.generated_initial_positions[:, -1, :]
                #     # last_velocity = tf.zeros_like(last_position, dtype=tf.float32)
                #     self.generated_positions_list = []
                #     self.generated_positions_list.append(last_position)
                #     for i in range(self.MAX_LEN):
                #         # last_xv = tf.concat([last_position, last_velocity], axis=1)
                #         last_xv = last_position
                #         last_xv_embeded = tf.layers.dense(last_xv, self.N_H_X_EMBEDDING, tf.nn.relu,
                #                                           name='embeded_X', reuse=True)
                #         last_output, last_state = self.cell(last_xv_embeded, last_state)
                #         embeded_last_output = tf.layers.dense(last_output, self.N_H_OUTPUT_EMBEDDING,
                #                                               activation=tf.nn.relu,
                #                                               name='embeded_rnn_output', reuse=True)
                #         delta_x = tf.layers.dense(embeded_last_output, 2, name='next_position', reuse=True)
                #         # last_velocity = last_x - last_position
                #         last_position = delta_x + last_position
                #         self.generated_positions_list.append(last_position)
                #
                #     self.generated_positions = tf.transpose(
                #         tf.stack(self.generated_positions_list)[:self.MAX_LEN - self.INITIAL_LEN_placeholder + 1],
                #         [1, 0, 2],
                #         name='generated_positions')
                #     self.guideless_generated_positions = tf.concat(
                #         [self.X_placeholder[:, 1:self.INITIAL_LEN_placeholder, :], self.generated_positions],
                #         axis=1, name='guideless_generated_positions')
                #     self.discrete_guideless_generated_positions = tf.round(self.guideless_generated_positions,
                #                                                            name='discrete_guideless_generated_positions')

            with tf.name_scope("Optimization"):
                self.square_differences = tf.reduce_sum(tf.square(self.guided_generated_positions - self.Y_placeholder),
                                                        axis=2,
                                                        name='square_differences')
                self.masked_square_differences = tf.multiply(self.square_differences, self.mask,
                                                             name='masked_square_differences')
                self.mse_loss = tf.reduce_mean(tf.reduce_sum(self.masked_square_differences, axis=1), name='mse_loss')
                self.likelihood = tf.reduce_mean(
                    tf.reduce_sum(self.masked_square_differences / (
                            2 * self.STD * self.STD), axis=1) + np.log(
                        2 * math.pi * self.STD * self.STD) * tf.cast(
                        self.LEN_placeholder, dtype=tf.float32), name='likelihood')

                # with tf.name_scope("Generation_Loss"):
                #     self.g_square_differences = tf.reduce_sum(
                #         # tf.square(self.guideless_generated_positions[:, self.INITIAL_LEN_placeholder:, :]
                #         tf.square(self.guideless_generated_positions[:, self.INITIAL_LEN_placeholder - 1:, :]
                #                   - self.Y_placeholder[:, self.INITIAL_LEN_placeholder - 1:, :]),
                #         axis=2,
                #         name='g_square_differences')
                #     self.g_mask = tf.sequence_mask(self.LEN_placeholder - self.INITIAL_LEN_placeholder + 1,
                #                                    self.MAX_LEN - self.INITIAL_LEN_placeholder + 1, dtype=tf.float32,
                #                                    name='g_mask')
                #     self.g_masked_square_differences = tf.multiply(self.g_square_differences, self.g_mask,
                #                                                    name='g_masked_square_differences')
                #
                #     self.g_likelihood = tf.reduce_mean(
                #         tf.reduce_sum(self.g_masked_square_differences / (2 * self.STD * self.STD), axis=1)
                #         + np.log(2 * math.pi * self.STD * self.STD)
                #         * tf.cast(self.LEN_placeholder - self.INITIAL_LEN_placeholder, dtype=tf.float32),
                #         name='g_likelihood')
                #
                #     self.average_generation_loss = tf.reduce_mean(
                #         tf.reduce_sum(self.g_masked_square_differences, axis=1) / tf.cast(
                #             self.LEN_placeholder - self.INITIAL_LEN_placeholder,
                #             tf.float32),
                #         name='average_generation_loss')
                #     self.sum_generation_loss = tf.reduce_mean(tf.reduce_sum(self.g_masked_square_differences, axis=1),
                #                                               name='sum_generation_loss')

                self.global_step = tf.train.get_or_create_global_step()
                self.learning_rate = tf.train.exponential_decay(self.INITIAL_LR, self.global_step, 5000, 0.5,
                                                                name='learning_rate')
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_operation = self.optimizer.minimize(self.mse_loss, global_step=self.global_step)

            self.init_node = tf.global_variables_initializer()
            self.save_node = tf.train.Saver()

        if summary:
            self.loss_summary = tf.summary.scalar(name='mse_loss', tensor=self.mse_loss)
            # self.sum_generation_loss_summary = tf.summary.scalar(name='sum_generation_loss',
            #                                                      tensor=self.sum_generation_loss)
            # self.average_generation_loss_summary = tf.summary.scalar(name='average_generation_loss',
            #                                                          tensor=self.average_generation_loss)
            self.likelihood_summary = tf.summary.scalar(name='likelihood', tensor=self.likelihood)
            # self.g_likelihood_summary = tf.summary.scalar(name='generation_likelihood', tensor=self.g_likelihood)
            self.accuracy_summary = tf.summary.scalar(name='accuracy', tensor=self.accuracy)
            # self.accuracy_summary = tf.summary.scalar(name="accuracy", tensor=self.accuracy)
            # self.distance_summary = tf.summary.scalar(name="distance", tensor=self.distance)
            self.scalar_summaries = tf.summary.merge(
                # [self.loss_summary, self.accuracy_summary, self.distance_summary])
                [
                    self.loss_summary,
                    # self.loss_summary,
                    # self.sum_generation_loss_summary,
                    # self.average_generation_loss_summary,
                    self.likelihood_summary,
                    # self.g_likelihood_summary,
                    self.accuracy_summary
                ])

            # self.input_image_summary = tf.summary.image(name="input_image", tensor=self.input_image, max_outputs=3)
            # self.output_image_summary = tf.summary.image(name="output_image", tensor=self.output_image,
            #                                              max_outputs=3)
            # self.image_summary = tf.summary.merge([self.input_image_summary, self.output_image_summary])
            #
            self.merged_summary = tf.summary.merge_all()

            self.summary_writer = tf.summary.FileWriter(self.LOG_PATH, tf.get_default_graph())
            self.validation_summary_writer = tf.summary.FileWriter(self.LOG_PATH + "validation")

    def add_code_summary(self):
        code_string = "\n".join(open('LSTM.py', 'r').readlines())
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
            if step % 100 == 0:
                print("Step {}".format(step))
            #     image_summary = self.ses.run(self.image_summary)
            #     self.summary_writer.add_summary(image_summary, step)
            if step % 50 == 0:
                X, Y, LEN_v = self.ses.run([self.X_v, self.Y_v, self.LEN_v])
                feed_dict = {self.X_placeholder: X, self.Y_placeholder: Y, self.LEN_placeholder: LEN_v}
                validation_scalar_summaries = self.ses.run(self.scalar_summaries,
                                                           feed_dict=feed_dict)
                self.validation_summary_writer.add_summary(validation_scalar_summaries, step)

    def save(self):
        self.save_node.save(self.ses, save_path=self.MODEL_PATH + "/" + self.NAME + '.ckpt')

    def load(self):
        self.save_node.restore(self.ses, save_path=self.MODEL_PATH + "/" + self.NAME + '.ckpt')

    def generate_image_sequecne(self, complete_sequence, guidance_length, with_gauidance=True, discrete=False):
        l = len(complete_sequence)
        if len(complete_sequence) == self.MAX_LEN:
            padded_complete_sequence = [complete_sequence]
        else:
            padded_complete_sequence = [complete_sequence[:self.MAX_LEN] + [(0, 0)] * max(0, self.MAX_LEN - len(
                complete_sequence))]
        if with_gauidance:
            if discrete:
                fetch = self.discrete_guided_generated_positions
            else:
                fetch = self.guided_generated_positions
            feed_dict = {self.X_placeholder: padded_complete_sequence, self.LEN_placeholder: [l]}
            generated_sequence = self.ses.run(fetch, feed_dict=feed_dict)
            # generated_sequence = np.array(generated_sequence[0][:l], dtype=np.int32)
            generated_sequence = generated_sequence[0][:l]
        else:
            if discrete:
                fetch = self.discrete_guideless_generated_positions
            else:
                fetch = self.guideless_generated_positions
            feed_dict = {self.X_placeholder: padded_complete_sequence, self.LEN_placeholder: [l],
                         self.INITIAL_LEN_placeholder: guidance_length}

            generated_sequence = self.ses.run(fetch, feed_dict=feed_dict)
            # generated_sequence = np.array(generated_sequence[0][:l], dtype=np.int32)
            generated_sequence = generated_sequence[0][:l]

        if discrete:
            generated_sequence = np.array(generated_sequence, np.int32)
        return generated_sequence

    def evaluate_validation_loss(self, n=10000, initial_length=10):
        counter = 0
        AGL, SGL, L, LH, GLH, AC = 0, 0, 0, 0, 0, 0
        while (counter < n):
            X_v, Y_v, LEN_v = self.ses.run([self.X_v, self.Y_v, self.LEN_v])
            batch_size = len(X_v)
            feed_dict = {self.X_placeholder: X_v, self.Y_placeholder: Y_v, self.LEN_placeholder: LEN_v,
                         self.INITIAL_LEN_placeholder: initial_length}
            agl, sgl, l, lh, glh, ac = self.ses.run(
                [self.average_generation_loss, self.sum_generation_loss, self.mse_loss, self.likelihood,
                 self.g_likelihood, self.accuracy],
                feed_dict=feed_dict)
            counter += batch_size
            AGL += agl * batch_size
            SGL += sgl * batch_size
            L += l * batch_size
            LH += lh * batch_size
            GLH += glh * batch_size
            AC += ac * batch_size

        AGL /= counter
        SGL /= counter
        L /= counter
        LH /= counter
        GLH /= counter
        AC /= counter
        print("Sum loss on a given sequence = {}".format(L))
        print("Likelihood on a given sequence = {}".format(LH))
        print("Average loss on generated sequences = {}".format(AGL))
        print("Sum loss on generated sequences = {}".format(SGL))
        print("Likelihood on generated sequences = {}".format(GLH))
        print("Accuracy on validation set = {}".format(AC))

    def generate_dataset(self, n=10000, guidance_length=10, with_guidance=True, discrete=False):
        initial_length = guidance_length
        counter = 0
        y_hat_dataset = []
        y_dataset = []
        len_dataset = []
        while (counter < n):
            X_v, Y_v, LEN_v = self.ses.run([self.X_v, self.Y_v, self.LEN_v])
            feed_dict = {self.INITIAL_LEN_placeholder: initial_length, self.LEN_placeholder: LEN_v,
                         self.X_placeholder: X_v, self.Y_placeholder: Y_v}
            # generated_positions = self.ses.run(self.guideless_generated_positions, feed_dict=feed_dict)
            if with_guidance:
                if discrete:
                    fetch = self.discrete_guided_generated_positions
                else:
                    fetch = self.guided_generated_positions
            else:
                if discrete:
                    fetch = self.discrete_guideless_generated_positions
                else:
                    fetch = self.guideless_generated_positions

            generated_positions = self.ses.run(fetch, feed_dict=feed_dict)
            len_dataset.append(LEN_v)
            y_dataset.append(Y_v)
            y_hat_dataset.append(generated_positions)
            counter += len(X_v)

        y_hat_dataset = np.concatenate(y_hat_dataset, axis=0)
        y_dataset = np.concatenate(y_dataset, axis=0)
        len_dataset = np.concatenate(len_dataset, axis=0)
        return y_hat_dataset, y_dataset, len_dataset

    def evaluate_likelihood_on_validation(self, n=10000):
        counter = 0
        L = 0
        while (counter < n):
            X_v, Y_v, LEN_v = self.ses.run([self.X_v, self.Y_v, self.LEN_v])
            b = len(X_v)
            counter += b
            feed_dict = {self.LEN_placeholder: LEN_v, self.X_placeholder: X_v, self.Y_placeholder: Y_v}
            fetch = self.likelihood
            l = self.ses.run(fetch, feed_dict=feed_dict)
            L += l * b
        L /= counter
        print("Likelihood on Validation Dataset = {}".format(L))

    def get_model_entropy(self, dataset, with_guidance=True):
        y_dataset = dataset
        x_dataset = np.concatenate([np.zeros_like(y_dataset)[:, :1, :], y_dataset[:, :-1, :]], axis=1)
        n = len(dataset)
        entropy = 0
        entropy_list = []
        counter = 0
        print(self.MAX_LEN)
        for b in range(0, n, self.BATCH_SIZE):
            x = x_dataset[b: b + self.BATCH_SIZE]
            y = y_dataset[b: b + self.BATCH_SIZE]
            batch_size = len(x)
            counter += batch_size
            l = [self.MAX_LEN] * batch_size
            feed_dict = {self.X_placeholder: x, self.Y_placeholder: y, self.LEN_placeholder: l}
            fetch = self.likelihood if with_guidance else self.g_likelihood
            glh = self.ses.run(fetch, feed_dict=feed_dict)
            entropy += glh * batch_size
            entropy_list.append(entropy / counter)

        entropy /= counter

        print("Entropy = {}".format(entropy))
        import matplotlib.pyplot as plt
        plt.plot(entropy_list)
        plt.show()

    def get_n_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Number of Parameters in model = {}".format(total_parameters))


# if __name__ == "__main__":
#     # Model = LSTM()
#     # Model.train()
#     # Model.save()
#     #
#     Model = LSTM(summary=False)
#     Model.load()
#     # Model.evaluate_validation_loss()
#     #
#     # Model.evaluate_validation_loss(10000, 10)
#     # dataset, _ = Model.generate_dataset(10000)
#     # np.save(Model.MODEL_PATH + "/dataset.npy", dataset)
#     dataset = np.load(Model.MODEL_PATH + "/dataset.npy")
#     Model.get_model_entropy(dataset)
#
#     # print(dataset.shape)
#     # #
#     # i = 1
#     # g = 10
#     # s = Model.Dataloader.validation_sequence_list[i]
#     # x = s.copy()
#     # s[1:] = s[:-1]
#     # s[0] = 0
#     # sg = Model.generate_image_sequecne(s, g, False, False)
#     # print(len(s))
#     # show_point_sequence(s)
#     # show_point_sequence(sg)
#     # show_trajectory(sg)
#
#     # show_positions_mus(sg, x)

#
# if __name__ == "__main__":
#     # Mnist_Model = LSTM("Mnist", "../Mnist_Saved_Models/LSTM/", True)
#     # Mnist_Model.train()
#     # Mnist_Model.save()
#     Mnist_Model = LSTM("Mnist", "../Mnist_Saved_Models/LSTM/", False)
#     Mnist_Model.load()
#     Mnist_Model.get_n_params()
#
#     # y_hat_dataset, y_dataset, len_dataset = Mnist_Model.generate_dataset(n=10000, guidance_length=10)
#     # dataset = [y_hat_dataset, y_dataset, len_dataset]
#     # dataset = {"y_hat": y_hat_dataset, "y": y_dataset, "len": len_dataset}
#     # np.save(Mnist_Model.MODEL_PATH + "/dataset.npy", dataset)
#     dataset = np.load(Mnist_Model.MODEL_PATH + "/dataset.npy").item()
#     y_hat_dataset, y_dataset, len_dataset = dataset["y_hat"], dataset["y"], dataset["len"]
#
#     msse_cl = MSSE(y_hat_dataset, y_dataset, len_dataset)
#     mase_cl = MASE(y_hat_dataset, y_dataset, len_dataset)
#     import matplotlib.pyplot as plt
#
#     print(msse_cl[-1])
#     print(mase_cl[-1])
#     plt.figure()
#     plt.plot(msse_cl)
#     plt.figure()
#     plt.plot(mase_cl)
#     plt.show()
#
#     # show_trajectory(sg)
#     # show_positions_mus(y_hat, y)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-n", "--name", dest="NAME", default="Mnist",
                        help="BATCH SIZE")
    parser.add_argument("-b", "--batch_size", dest="BATCH_SIZE", default=64,
                        help="BATCH SIZE")
    parser.add_argument("-ilr", "--initial_lr", dest="INITIAL_LR", default=0.005,
                        help="Initial learning rate")
    parser.add_argument("-lrd", "--lr_decay", dest="DECAY_STEP", default=1000,
                        help="Number of steps to halve the learning rate")
    parser.add_argument("-il", "--initial_length", dest="INITIAL_LENGTH", default=10,
                        help="Initial Length")
    parser.add_argument("-ts", "--total_steps", dest="TOTAL_STEPS", default=300,
                        help="total training steps")
    parser.add_argument("-gpu", "--gpu_id", dest="GPU_ID", default=0,
                        help="GPU_ID to use")
    parser.add_argument("-train", "--train", dest="TRAIN", default="True",
                        help="Train or load and evaluate")

    args = parser.parse_args()
    ENC_DEC_LSTM.NAME = args.NAME
    ENC_DEC_LSTM.TOTAL_STEPS = int(args.TOTAL_STEPS)
    ENC_DEC_LSTM.DECAY_STEP = int(args.DECAY_STEP)
    ENC_DEC_LSTM.INITIAL_LR = float(args.INITIAL_LR)
    ENC_DEC_LSTM.BATCH_SIZE = int(args.BATCH_SIZE)
    ENC_DEC_LSTM.INITIAL_LENGTH = int(args.INITIAL_LENGTH)
    ENC_DEC_LSTM.GPU_ID = int(args.GPU_ID)
    ENC_DEC_LSTM.DESCRIPTION = str(args)
    TRAIN = True if args.TRAIN == "True" else False
    if TRAIN:
        A = ENC_DEC_LSTM(args.NAME)
        A.train()
        A.save()
    else:
        A = ENC_DEC_LSTM(name=args.NAME, summary=False)
        A.load()
        y_hat_dataset, y_dataset, len_dataset = A.generate_dataset(512, with_guidance=True, discrete=True)
        msse, msse_std = MSSE_(y_hat_dataset, y_dataset, len_dataset, 10)
        mase, mase_std = MASE_(y_hat_dataset, y_dataset, len_dataset, 10)
        acu, acu_std = ACU_(y_hat_dataset, y_dataset, len_dataset, 10)
        dgl, dgl_std = discretized_gaussian_likelihood_(y_hat_dataset, y_dataset, len_dataset, sigma=1.0)
        print("Mean Sum Squared Error = ", msse, "+-", msse_std)
        print("Mean Average Squared Error = ", mase, "+-", mase_std)
        print("Accuracy = ", acu, "+-", acu_std)
        print("Likelihood = ", dgl, "+-", dgl_std)
        # A.evaluate_likelihood_on_validation(2000)

# Mean Sum Squared Error =  36.49241866424015 +- 3.0763565372561543
# Mean Average Squared Error =  1.3014993127216992 +- 0.18552703608200982
# Accuracy =  0.5623635178433669 +- 0.0034884161895027866
# Likelihood on Validation Dataset = 70.97678029537201
