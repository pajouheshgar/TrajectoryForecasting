import numpy as np
import scipy.stats as statistics
import math


def MASE(y_hat_dataset, y_dataset, len_dataset, initial_length):
    # Mean Average Squared Error

    mase = 0
    convergence_list = []
    counter = 0
    for y_hat, y, l in zip(y_hat_dataset, y_dataset, len_dataset):
        counter += 1
        tmp = 0
        for p_hat, p in zip(y_hat[initial_length - 1:l], y[initial_length - 1:l]):
            for x_hat, x in zip(p_hat, p):
                tmp += (x_hat - x) ** 2
        tmp /= (l - initial_length + 1)
        mase += tmp
        convergence_list.append(tmp)
    mase /= counter
    return mase, np.std(convergence_list) / np.sqrt(counter)


def MSSE(y_hat_dataset, y_dataset, len_dataset, initial_length):
    # Mean Sum Squared Error
    msse = 0
    convergence_list = []
    counter = 0
    for y_hat, y, l in zip(y_hat_dataset, y_dataset, len_dataset):
        tmp = 0
        counter += 1
        for p_hat, p in zip(y_hat[initial_length - 1:l], y[initial_length - 1:l]):
            for x_hat, x in zip(p_hat, p):
                tmp += (x_hat - x) ** 2
        msse += tmp
        convergence_list.append(tmp)
    msse /= counter
    return msse, np.std(convergence_list) / np.sqrt(counter)


def ACU(y_hat_dataset, y_dataset, len_dataset, initial_length):
    # Mean Average Squared Error

    acu = 0
    convergence_list = []
    counter = 0
    for y_hat, y, l in zip(y_hat_dataset, y_dataset, len_dataset):
        counter += 1
        tmp = 0
        for p_hat, p in zip(y_hat[initial_length - 1:l], y[initial_length - 1:l]):
            flag = True
            for x_hat, x in zip(p_hat, p):
                if round(x) != round(x_hat):
                    flag = False
            if flag:
                tmp += 1
        tmp /= (l - initial_length + 1)
        acu += tmp
        convergence_list.append(tmp)
    acu /= counter
    return acu, np.std(convergence_list) / np.sqrt(counter)


# def discretized_gaussian_likelihood(y_hat_dataset, y_dataset, len_dataset, initial_length, sigma):
#     def approximate_log_prob(x, sigma=1):
#         x_left = math.floor(x)
#         x_right = x_left + 1
#         x_min = x_right if abs(x_right) > abs(x_left) else x_left
#         log_prob = -(x_min ** 2) / (2 * (sigma ** 2)) - 0.5 * math.log(2 * math.pi * (sigma ** 2))
#         return log_prob
#
#     # Mean Sum Squared Error
#     L = 0
#     convergence_list = []
#     counter = 0
#     for y_hat, y, l in zip(y_hat_dataset, y_dataset, len_dataset):
#         sequence_likelihood = 0
#         counter += 1
#         for p_hat, p in zip(y_hat[initial_length - 1:l], y[initial_length - 1:l]):
#             for x_hat, x in zip(p_hat, p):
#                 left_point = math.floor(x_hat - x)
#                 right_point = left_point + 1
#                 p = statistics.norm.cdf(right_point, scale=sigma) - statistics.norm.cdf(
#                     left_point, scale=sigma)
#                 if p == 0:
#                     log_prob = approximate_log_prob(x_hat - x, sigma)
#                     sequence_likelihood += -log_prob
#                 else:
#                     sequence_likelihood += -math.log(p)
#
#
#         L += sequence_likelihood
#         convergence_list.append(sequence_likelihood)
#     L /= counter
#     return L, np.std(convergence_list) / np.sqrt(counter)

def discretized_gaussian_likelihood(y_hat_dataset, y_dataset, len_dataset, initial_length, sigma):
    def approximate_log_prob(x, sigma):
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
    for y_hat, y, l in zip(y_hat_dataset, y_dataset, len_dataset):
        min_sequence_likelihood = 0
        max_sequence_likelihood = 0
        counter += 1
        for p_hat, p in zip(y_hat[initial_length - 1:l], y[initial_length - 1:l]):
            # for x_hat, x in zip(p_hat, p):
            #     if math.ceil(x_hat - x) == math.floor(x_hat - x):
            #         t = math.ceil(x_hat - x)
            #         p_right = statistics.norm.cdf(t + 1, scale=sigma) - statistics.norm.cdf(t, scale=sigma)
            #         p_left = statistics.norm.cdf(t, scale=sigma) - statistics.norm.cdf(t - 1, scale=sigma)
            #         p = (p_right + p_left) / 2
            #         if p == 0:
            #             sequence_likelihood += -approximate_log_prob(x, sigma)
            #         else:
            #             sequence_likelihood += -math.log(p)
            #     else:
            #         p = statistics.norm.cdf(math.ceil(x_hat - x), scale=sigma) - statistics.norm.cdf(
            #             math.floor(x_hat - x), scale=sigma)
            #         if p == 0:
            #             sequence_likelihood += -approximate_log_prob(x_hat - x, sigma)
            #         else:
            #             sequence_likelihood += -math.log(p)
            for x_hat, x in zip(p_hat, p):
                left_point = math.floor(x_hat - x)
                right_point = left_point + 1
                p = statistics.norm.cdf(right_point, scale=sigma) - statistics.norm.cdf(
                    left_point, scale=sigma)
                if p == 0:
                    min_log_prob, max_log_prob = approximate_log_prob(x_hat - x, sigma)
                    min_sequence_likelihood += -min_log_prob
                    max_sequence_likelihood += -max_log_prob
                else:
                    min_sequence_likelihood += -math.log(p)
                    max_sequence_likelihood += -math.log(p)

        min_L += min_sequence_likelihood
        max_L += max_sequence_likelihood
        min_convergence_list.append(min_sequence_likelihood)
        max_convergence_list.append(max_sequence_likelihood)
    min_L /= counter
    max_L /= counter
    return min_L, np.std(min_convergence_list) / np.sqrt(counter), max_L, np.std(max_convergence_list) / np.sqrt(counter)



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


def ACU_(y_hat_dataset, y_dataset, len_dataset, initial_length):
    # Mean Average Squared Error

    acu = 0
    convergence_list = []
    counter = 0
    for y_hat, y, l in zip(y_hat_dataset, y_dataset, len_dataset):
        counter += 1
        tmp = 0
        for p_hat, p in zip(y_hat[10-initial_length:l], y[10-initial_length:l]):
            flag = True
            for x_hat, x in zip(p_hat, p):
                if round(x) != round(x_hat):
                    flag = False
            if flag:
                tmp += 1
        tmp /= (l - 10 + initial_length)
        acu += tmp
        convergence_list.append(tmp)
    acu /= counter
    return acu, np.std(convergence_list) / np.sqrt(counter)


# def discretized_gaussian_likelihood_(y_hat_dataset, y_dataset, len_dataset, sigma):
#     def approximate_log_prob(x, sigma=1):
#         x_right = math.ceil(x)
#         x_left = math.floor(x)
#         x_min = x_right if abs(x_right) > abs(x_left) else x_left
#         log_prob = -(x_min ** 2) / (2 * (sigma ** 2)) - 0.5 * math.log(2 * math.pi * (sigma ** 2))
#         return log_prob
#
#     # Mean Sum Squared Error
#     L = 0
#     convergence_list = []
#     counter = 0
#     for y_hat, y, l in zip(y_hat_dataset, y_dataset, len_dataset):
#         sequence_likelihood = 0
#         counter += 1
#         for p_hat, p in zip(y_hat[:l], y[:l]):
#             for x_hat, x in zip(p_hat, p):
#                 if math.ceil(x_hat - x) == math.floor(x_hat - x):
#                     t = math.ceil(x_hat - x)
#                     p_right = statistics.norm.cdf(t + 1, scale=sigma) - statistics.norm.cdf(t, scale=sigma)
#                     p_left = statistics.norm.cdf(t, scale=sigma) - statistics.norm.cdf(t - 1, scale=sigma)
#                     p = (p_right + p_left) / 2
#                     if p == 0:
#                         sequence_likelihood += -approximate_log_prob(x, sigma)
#                     else:
#                         sequence_likelihood += -math.log(p)
#                 else:
#                     p = statistics.norm.cdf(math.ceil(x_hat - x), scale=sigma) - statistics.norm.cdf(
#                         math.floor(x_hat - x), scale=sigma)
#                     if p == 0:
#                         sequence_likelihood += -approximate_log_prob(x, sigma)
#                     else:
#                         sequence_likelihood += -math.log(p)
#
#         L += sequence_likelihood
#         convergence_list.append(sequence_likelihood)
#     L /= counter
#     return L, np.std(convergence_list) / np.sqrt(counter)

def discretized_gaussian_likelihood_(y_hat_dataset, y_dataset, len_dataset, sigma):
    def approximate_log_prob(x, sigma):
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
    for y_hat, y, l in zip(y_hat_dataset, y_dataset, len_dataset):
        min_sequence_likelihood = 0
        max_sequence_likelihood = 0
        counter += 1
        for p_hat, p in zip(y_hat[:l], y[:l]):
            # for x_hat, x in zip(p_hat, p):
            #     if math.ceil(x_hat - x) == math.floor(x_hat - x):
            #         t = math.ceil(x_hat - x)
            #         p_right = statistics.norm.cdf(t + 1, scale=sigma) - statistics.norm.cdf(t, scale=sigma)
            #         p_left = statistics.norm.cdf(t, scale=sigma) - statistics.norm.cdf(t - 1, scale=sigma)
            #         p = (p_right + p_left) / 2
            #         if p == 0:
            #             sequence_likelihood += -approximate_log_prob(x, sigma)
            #         else:
            #             sequence_likelihood += -math.log(p)
            #     else:
            #         p = statistics.norm.cdf(math.ceil(x_hat - x), scale=sigma) - statistics.norm.cdf(
            #             math.floor(x_hat - x), scale=sigma)
            #         if p == 0:
            #             sequence_likelihood += -approximate_log_prob(x_hat - x, sigma)
            #         else:
            #             sequence_likelihood += -math.log(p)
            for x_hat, x in zip(p_hat, p):
                left_point = math.floor(x_hat - x)
                right_point = left_point + 1
                p = statistics.norm.cdf(right_point, scale=sigma) - statistics.norm.cdf(
                    left_point, scale=sigma)
                if p == 0:
                    min_log_prob, max_log_prob = approximate_log_prob(x_hat - x, sigma)
                    min_sequence_likelihood += -min_log_prob
                    max_sequence_likelihood += -max_log_prob
                else:
                    min_sequence_likelihood += -math.log(p)
                    max_sequence_likelihood += -math.log(p)

        min_L += min_sequence_likelihood
        max_L += max_sequence_likelihood
        min_convergence_list.append(min_sequence_likelihood)
        max_convergence_list.append(max_sequence_likelihood)
    min_L /= counter
    max_L /= counter
    return min_L, np.std(min_convergence_list) / np.sqrt(counter), max_L, np.std(max_convergence_list) / np.sqrt(counter)



def discretized_gaussian_likelihood__(y_hat_dataset, y_dataset, len_dataset, sigma, alpha=0.1):
    def approximate_log_prob(x, sigma=1):
        x_right = math.ceil(x)
        x_left = math.floor(x)
        x_min = x_right if abs(x_right) > abs(x_left) else x_left
        log_prob = -(x_min ** 2) / (2 * (sigma ** 2)) - 0.5 * math.log(2 * math.pi * (sigma ** 2))
        return log_prob

    # Mean Sum Squared Error
    L = 0
    convergence_list = []
    counter = 0
    for y_hat, y, l in zip(y_hat_dataset, y_dataset, len_dataset):
        sequence_likelihood = 0
        counter += 1
        for p_hat, p in zip(y_hat[:l], y[:l]):
            for x_hat, x in zip(p_hat, p):
                if math.ceil(x_hat - x) == math.floor(x_hat - x):
                    t = math.ceil(x_hat - x)
                    p_right = statistics.norm.cdf(t + 1, scale=sigma) - statistics.norm.cdf(t, scale=sigma)
                    p_left = statistics.norm.cdf(t, scale=sigma) - statistics.norm.cdf(t - 1, scale=sigma)
                    p = (p_right + p_left) / 2
                    if p == 0:
                        sequence_likelihood += -approximate_log_prob(x, sigma)
                    else:
                        sequence_likelihood += -math.log(p)
                else:
                    p = statistics.norm.cdf(math.ceil(x_hat - x), scale=sigma) - statistics.norm.cdf(
                        math.floor(x_hat - x), scale=sigma)
                    if p == 0:
                        sequence_likelihood += -approximate_log_prob(x, sigma)
                    else:
                        sequence_likelihood += -math.log(p)

        L += sequence_likelihood
        convergence_list.append(sequence_likelihood)
    L /= counter
    return L, np.std(convergence_list) / np.sqrt(counter)
