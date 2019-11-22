import datetime
import os
import math

import tensorflow as tf

from BaseModel import BaseModel
from SDD_Metrics import *
from SDD_Dataloader import Abosulute_Single_Object_Position_Dataloader


class SimpleTenModel(BaseModel):
    NAME = "SDD"
    SAVE_DIR = "../SDD_Saved_Models/SimpleTenModel/"
    RS = 42

    SHUFFLE_BUFFER_SIZE = 2048

    STRIDE = 10

    STD = 1.0
    SPLIT_ID = 4
    DESCRIPTION = ""

    N_MODELS = 3

    def __init__(self, name=NAME, save_dir=SAVE_DIR, summary=True):
        super(SimpleTenModel, self).__init__(name, save_dir)
        self.MODEL_PATH = os.path.join(self.SAVE_DIR, self.NAME)
        try:
            os.mkdir(self.MODEL_PATH)
        except:
            pass
        init_time = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.LOG_PATH = os.path.join(self.SAVE_DIR, "log/" + self.NAME + "-run-" + init_time)
        self.init_graph(summary)
        self.ses = tf.InteractiveSession()

    def init_graph(self, summary=True):
        np.random.seed(self.RS)
        tf.set_random_seed(self.RS)
        with tf.variable_scope(self.NAME):
            with tf.name_scope("Dataset"):
                with tf.name_scope("Dataloader"):
                    self.Dataloader = Abosulute_Single_Object_Position_Dataloader(
                        future_resolution=self.STRIDE,
                        past_resolution=self.STRIDE,
                        shuffle_buffer_size=self.SHUFFLE_BUFFER_SIZE,
                        batch_size=256,
                        test_split_id=self.SPLIT_ID,
                        sdd_dataset_dir="../Dataset/SDD/")
                    self.PAST_LEN = self.Dataloader.N_PAST_POINTS
                    self.FUTURE_LEN = self.Dataloader.N_FUTURE_POINTS

                    self.PAST_TRAJECTORY_tr, self.FUTURE_TRAJECTORY_tr, self.TYPE_tr, self.W_tr, self.H_tr = self.Dataloader.get_train_batch
                    self.PAST_TRAJECTORY_te, self.FUTURE_TRAJECTORY_te, self.TYPE_te, self.W_te, self.H_te = self.Dataloader.get_test_batch
                    self.PAST_TRAJECTORY_u, self.FUTURE_TRAJECTORY_u, self.TYPE_u, self.W_u, self.H_u = self.Dataloader.get_unstable_batch
                    self.PAST_TRAJECTORY_w, self.FUTURE_TRAJECTORY_w, self.TYPE_w, self.W_w, self.H_w = self.Dataloader.get_whole_batch

            self.init_node = tf.global_variables_initializer()

            # with tf.name_scope("Placeholders"):
            #     self.PAST_TRAJECTORY_placeholder = tf.placeholder_with_default(self.PAST_TRAJECTORY_tr,
            #                                                                    shape=[None, self.PAST_LEN, 2],
            #                                                                    name='PAST_TRAJECTORY_placeholder')
            #     self.FUTURE_TRAJECTORY_placeholder = tf.placeholder_with_default(self.FUTURE_TRAJECTORY_tr,
            #                                                                      shape=[None, self.FUTURE_LEN, 2],
            #                                                                      name='FUTURE_TRAJECTORY_placeholder')
            #     self.OBJECT_TYPE_placeholder = tf.placeholder_with_default(self.TYPE_tr, shape=[None],
            #                                                                name='OBJECT_TYPE_placeholder')
            #     self.W_placeholder = tf.placeholder_with_default(self.W_tr, shape=[None], name='W_placeholder')
            #     self.H_placeholder = tf.placeholder_with_default(self.H_tr, shape=[None], name='H_placeholder')

    def init_variables(self):
        self.ses.run(self.init_node)

    def train(self):
        self.init_variables()
        self.MODELS_LIST = [
            lambda x: self._1_Stay_Model(x),
            lambda x: self._2_Move_With_Constant_Speed(x, 0.0),
            lambda x: self._2_Move_With_Constant_Speed(x, 0.3),
            lambda x: self._2_Move_With_Constant_Speed(x, 0.7),
            lambda x: self._2_Move_With_Constant_Speed(x, 1.0),

            lambda x: self._3_Move_With_Average_Speed(x, 0.0),
            lambda x: self._3_Move_With_Average_Speed(x, 8.0),
            lambda x: self._3_Move_With_Average_Speed(x, -8.0),
            lambda x: self._3_Move_With_Average_Speed(x, 15.0),
            lambda x: self._3_Move_With_Average_Speed(x, -15.0),
        ]
        self.N_MODELS = len(self.MODELS_LIST)

    def _1_Stay_Model(self, past_trajectories_list):
        def Stay_Model(past_trajectory):
            predicted_future_trajectory = []
            for l in range(self.FUTURE_LEN):
                predicted_future_trajectory.append(past_trajectory[-1])
            return predicted_future_trajectory

        predicted_future_trajectories_list = []
        for past_trajectory in past_trajectories_list:
            predicted_future_trajectories_list.append(Stay_Model(past_trajectory))
        return predicted_future_trajectories_list

    def _2_Move_With_Constant_Speed(self, past_trajectories_list, alpha):

        def Move_With_Constant_Speed(past_trajectory):
            speed = past_trajectory[1] - past_trajectory[0]
            for i in range(1, len(past_trajectory) - 1):
                speed = speed * (1 - alpha) + alpha * (past_trajectory[i + 1] - past_trajectory[i])

            last_position = past_trajectory[-1]
            predicted_future_trajectory = []
            for l in range(self.FUTURE_LEN):
                last_position = last_position + speed
                predicted_future_trajectory.append(last_position)
            return predicted_future_trajectory

        predicted_future_trajectories_list = []
        for past_trajectory in past_trajectories_list:
            predicted_future_trajectories_list.append(Move_With_Constant_Speed(past_trajectory))
        return predicted_future_trajectories_list

    def _3_Move_With_Average_Speed(self, past_trajectories_list, rotation=0.0):
        radian_angle = rotation * math.pi / 180.0
        def Move_With_Constant_Speed(past_trajectory):
            speed = 0
            for i in range(len(past_trajectory) - 1):
                speed = speed + (past_trajectory[i + 1] - past_trajectory[i])
            speed /= (len(past_trajectory) - 1)
            x_speed = speed[1] * math.sin(radian_angle) + speed[0] * math.cos(radian_angle)
            y_speed = speed[1] * math.cos(radian_angle) - speed[0] * math.sin(radian_angle)
            speed = np.array([x_speed, y_speed])

            last_position = past_trajectory[-1]
            predicted_future_trajectory = []
            for l in range(self.FUTURE_LEN):
                last_position = last_position + speed
                predicted_future_trajectory.append(last_position)
            return predicted_future_trajectory

        predicted_future_trajectories_list = []
        for past_trajectory in past_trajectories_list:
            predicted_future_trajectories_list.append(Move_With_Constant_Speed(past_trajectory))
        return predicted_future_trajectories_list



    def generate_oracle_dataset(self, n=1500, type='train'):
        counter = 0
        y_hat_dataset_list = [[] for _ in range(self.N_MODELS)]
        y_dataset = []
        W_dataset = []
        H_dataset = []
        if type == 'train':
            input_fetch = [self.PAST_TRAJECTORY_tr, self.FUTURE_TRAJECTORY_tr, self.TYPE_tr, self.W_tr, self.H_tr]
        elif type == 'test':
            input_fetch = [self.PAST_TRAJECTORY_te, self.FUTURE_TRAJECTORY_te, self.TYPE_te, self.W_te, self.H_te]
            n = 1000000
        elif type == 'whole':
            input_fetch = [self.PAST_TRAJECTORY_w, self.FUTURE_TRAJECTORY_w, self.TYPE_w, self.W_w, self.H_w]
            n = 1000000
        else:
            input_fetch = [self.PAST_TRAJECTORY_u, self.FUTURE_TRAJECTORY_u, self.TYPE_u, self.W_u, self.H_u]

        print('Generating Oracle dataset')
        while (counter < n):
            print("{} / {}".format(counter, n))
            try:
                PAST_TRAJECTORY, FUTURE_TRAJECTORY, TYPE, W, H = self.ses.run(input_fetch)
            except:
                break

            l = len(PAST_TRAJECTORY)

            for i in range(self.N_MODELS):
                generated_positions = self.MODELS_LIST[i](PAST_TRAJECTORY)
                y_hat_dataset_list[i].append(generated_positions)
            y_dataset.append(FUTURE_TRAJECTORY)
            W_dataset.append(W)
            H_dataset.append(H)
            counter += l

        y_hat_dataset_list = [np.concatenate(y_hat_dataset, axis=0) for y_hat_dataset in y_hat_dataset_list]
        y_dataset = np.concatenate(y_dataset, axis=0)
        W_dataset = np.concatenate(W_dataset, axis=0)
        H_dataset = np.concatenate(H_dataset, axis=0)
        return y_hat_dataset_list, y_dataset, W_dataset, H_dataset

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



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-n", "--name", dest="NAME", default="SDD",
                        help="Model Name")

    parser.add_argument("-STRIDE", "--stride", dest="STRIDE", default=3,
                        help="STRIDE")

    parser.add_argument("-train", "--train", dest="TRAIN", default="True",
                        help="Train or load and evaluate")

    args = parser.parse_args()

    SimpleTenModel.STRIDE = int(args.STRIDE)

    TRAIN = True if args.TRAIN == "True" else False
    TRAIN = False

    if TRAIN:
        A = SimpleTenModel(args.NAME)
        A.train()

    else:
        A = SimpleTenModel(name=args.NAME, summary=False)
        A.train()
        y_hat_dataset_list, y_dataset, W_dataset, H_dataset = A.generate_oracle_dataset(0, type='whole')
        result = L2_error_1234_second(y_hat_dataset_list, y_dataset, W_dataset, H_dataset, A.STRIDE, "Min",
                                      down_scale_rate=5, top_percentage=0.1)
        for t in range(4):
            print("Mean L2 error at {} sec = {} +- {}".format(t + 1, result[0][t], result[1][t]))

# Mean L2 error at 1 sec = 0.7433820237726501 +- 0.009420267349287682
# Mean L2 error at 2 sec = 1.7425942571648563 +- 0.023926810615537787
# Mean L2 error at 3 sec = 3.0043488275825805 +- 0.04370343176592408
# Mean L2 error at 4 sec = 4.49521126526468 +- 0.06778902006557971
