import numpy as np
import math
import scipy.stats as statistics

FUTURE_LEN = 120


def L2_distance(r1, r2, w, h, scale=5):
    x1 = (r1[0] + 1) * w / 2 / scale
    x2 = (r2[0] + 1) * w / 2 / scale

    y1 = (r1[1] + 1) * h / 2 / scale
    y2 = (r2[1] + 1) * h / 2 / scale

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def L2_error_1234_second(y_hat_dataset_list, y_dataset, W_dataset, H_dataset, future_resolution=30, policy="Min",
                         down_scale_rate=5, top_percentage=0.1):
    assert 30 % future_resolution == 0
    step = (FUTURE_LEN // 4) // future_resolution
    starting_index = step - 1

    desired_points = [starting_index + _ * step for _ in range(4)]
    error_in_desired_points = [[] for _ in range(4)]
    n = len(y_dataset)
    for i in range(n):
        w = W_dataset[i]
        h = H_dataset[i]
        for t in range(4):
            error_list_in_t = []
            for k in range(len(y_hat_dataset_list)):
                L2_error = L2_distance(y_hat_dataset_list[k][i][desired_points[t]],
                                       y_dataset[i][desired_points[t]], w, h, down_scale_rate)
                error_list_in_t.append(L2_error)
            if policy == "Min":
                error_list_in_t = sorted(error_list_in_t)
                top = int(len(error_list_in_t) * top_percentage)
                top = top if top > 0 else len(y_hat_dataset_list)
                error_in_desired_points[t].append(np.mean(error_list_in_t[:top]))

    error_list_in_desired_points = error_in_desired_points
    std_in_desired_points = [np.std(error_in_desired_points[i]) / math.sqrt(n) for i in range(4)]
    error_in_desired_points = [np.mean(error_in_desired_points[i]) for i in range(4)]
    return error_in_desired_points, std_in_desired_points, error_list_in_desired_points


def discretized_gaussian_likelihood_(y_hat_dataset, y_dataset, sigma, scale=256.0, alpha=0.1):
    def approximate_log_prob(x, sigma=1):
        x_left = math.floor(x)
        x_right = x_left + 1
        x_min = x_right if abs(x_right) > abs(x_left) else x_left
        x_max = x_left if abs(x_right) > abs(x_left) else x_right
        min_log_prob = -(x_min ** 2) / (2 * (sigma ** 2)) - 0.5 * math.log(2 * math.pi * (sigma ** 2))
        max_log_prob = -(x_max ** 2) / (2 * (sigma ** 2)) - 0.5 * math.log(2 * math.pi * (sigma ** 2))
        return min_log_prob, max_log_prob

    # Mean Sum Squared Error
    min_L = 0
    max_L = 0
    min_convergence_list = []
    max_convergence_list = []
    counter = 0
    y_hat_dataset = y_hat_dataset * scale / 2.0
    y_dataset = y_dataset * scale / 2.0

    for y_hat, y in zip(y_hat_dataset, y_dataset):
        min_sequence_likelihood = 0
        max_sequence_likelihood = 0
        counter += 1
        for p_hat, p in zip(y_hat, y):
            for x_hat, x in zip(p_hat, p):
                left_point = math.floor(x_hat - x)
                right_point = left_point + 1
                p = statistics.norm.cdf(right_point, scale=sigma) - statistics.norm.cdf(
                    left_point, scale=sigma)
                if p == 0 and False:
                    min_log_prob, max_log_prob = approximate_log_prob(x_hat - x, sigma)
                    min_sequence_likelihood += -min_log_prob
                    max_sequence_likelihood += -max_log_prob
                else:
                    p = p * (1 - alpha) + alpha / (scale * scale)
                    min_sequence_likelihood += -math.log(p)
                    max_sequence_likelihood += -math.log(p)

        min_L += min_sequence_likelihood
        max_L += max_sequence_likelihood
        min_convergence_list.append(min_sequence_likelihood)
        max_convergence_list.append(max_sequence_likelihood)
    min_L /= counter
    max_L /= counter
    min_convergence_list = np.array(min_convergence_list)
    max_convergence_list = np.array(min_convergence_list)
    min_L_error = np.std(min_convergence_list[min_convergence_list < 200]) / len(min_convergence_list[min_convergence_list < 200])
    max_L_error = np.std(max_convergence_list[max_convergence_list < 200]) / len(max_convergence_list[max_convergence_list < 200])
    return min_L, min_L_error, max_L, max_L_error


def MASE_(y_hat_dataset, y_dataset, len_dataset, initial_length):
    # Mean Average Squared Error

    mase = 0
    convergence_list = []
    counter = 0
    for y_hat, y, l in zip(y_hat_dataset, y_dataset, len_dataset):
        counter += 1
        tmp = 0
        for p_hat, p in zip(y_hat[10 - initial_length:l], y[10 - initial_length:l]):
            for x_hat, x in zip(p_hat, p):
                tmp += (x_hat - x) ** 2
        tmp /= l - 10 + initial_length
        mase += tmp
        convergence_list.append(tmp)
    mase /= counter
    return mase, np.std(convergence_list) / np.sqrt(counter)


def MSSE_(y_hat_dataset, y_dataset, len_dataset, initial_length):
    assert initial_length <= 10
    # Mean Sum Squared Error
    msse = 0
    convergence_list = []
    counter = 0
    for y_hat, y, l in zip(y_hat_dataset, y_dataset, len_dataset):
        tmp = 0
        counter += 1
        for p_hat, p in zip(y_hat[10 - initial_length:l], y[10 - initial_length:l]):
            for x_hat, x in zip(p_hat, p):
                tmp += (x_hat - x) ** 2
        msse += tmp
        convergence_list.append(tmp)
    msse /= counter
    return msse, np.std(convergence_list) / np.sqrt(counter)


def ACU_(y_hat_dataset, y_dataset, future_len):
    # Mean Average Squared Error

    acu = 0
    convergence_list = []
    counter = 0
    for y_hat, y in zip(y_hat_dataset, y_dataset):
        counter += 1
        tmp = 0
        for p_hat, p in zip(y_hat, y):
            flag = True
            for x_hat, x in zip(p_hat, p):
                if round(x) != round(x_hat):
                    flag = False
            if flag:
                tmp += 1
        tmp /= future_len
        acu += tmp
        convergence_list.append(tmp)
    acu /= counter
    return acu, np.std(convergence_list) / np.sqrt(counter)


if __name__ == "__main__":
    pass
