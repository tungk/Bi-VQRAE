import json
import os
import random
from math import pi
from pathlib import Path
import ast
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def rolling_window_2D(a, n):
    # a: 2D Input array
    # n: Group/sliding window length
    return a[np.arange(a.shape[0] - n + 1)[:, None] + np.arange(n)]


def cutting_window_2D(a, n):
    # a: 2D Input array
    # n: Group/sliding window length
    split_positions = list(range(n, a.shape[0], n))
    split_result = np.array_split(a, split_positions)
    np_result = []
    if split_result[-1].shape[0] == split_result[-2].shape[0]:
        for array in split_result[:-1]:
            np_result.append(array)
    else:
        for array in split_result[:-1]:
            np_result.append(array)
    return np.stack(np_result)


def unroll_window_2D(a):
    '''
    :param a: 2D data, matrix of probability scores of rolling windows
    :return: 1D data, final probability score for points
    '''
    return np.array([a.diagonal(i).mean() for i in range(-a.shape[0] + 1, a.shape[1])])

def unroll_window_3D(a):
    '''
    :param a: 3D data, matrix of probability scores of rolling windows (total_length, rolling_size, features)
    :return: 1D data, final probability score for points
    '''
    multi_ts = []
    for channel in range(a.shape[2]):
        uni_ts = np.array([a[:, :, channel].diagonal(i).mean() for i in range(-a[:, :, channel].shape[0] + 1, a[:, :, channel].shape[1])])
        multi_ts.append(uni_ts)

    multi_ts = np.stack(multi_ts, axis=1)
    return multi_ts


def generate_synthetic_dataset(case=0, length=2000, max=100, theta=1, anomalies=100, noise=False, verbose=True):
    random.seed(a=2412)
    if case == 0:
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        t = np.arange(0, length)
        # trend = t * np.sin(0.2 * (t - N)) + 2 * t
        # trend = 200 * np.sin(0.2 * (t - length)) + 2 * t
        trend = max * np.sin(0.05 * (t - length))
        if verbose:
            plt.figure(figsize=(9, 3))
            ax = plt.axes()

            ax.plot(t, scaler.fit_transform(np.expand_dims(trend, 1)), 'blue', lw=2)

            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim(-10, 260)
            ax.set_ylim(-0.10, 1.10)

            plt.xlabel('$t$')
            plt.ylabel('$s$')
            plt.grid(True)
            plt.tight_layout()

            plt.show()
            # plt.savefig('./figures/0/T_0.png')
            # plt.close()
        if noise:
            noise = 3 * (np.random.rand(length) - 0.5)
        output = trend + noise
        label = np.ones(output.shape)
        # output = np.expand_dims(output, 1)
        # output = scaler.fit_transform(output)
        # injection_1 = [18, 24, 27, 62, 80, 83, 143, 173, 181, 205]
        injection_1 = random.sample(range(len(output)), anomalies)
        # injection_2 = [40, 65, 108, 127, 135, 196, 234]
        # injection_3 = [17, 18, 23, 24, 26, 27, 39, 40, 61, 62, 64, 65, 79, 80, 82, 83, 107, 108, 126, 127, 134, 135, 142, 143, 172, 173,
        #                180, 181, 195, 196, 204, 205, 233, 234]
        for index in injection_1:
            # output[index] = random.randrange(0, 1)
            # if random.randint(0, 1) == 0:
            #     output[index] = random.randrange(0, 1)
            # else:
            #     output[index] = random.randrange(-1, 0)
            output[index] = output[index] + theta
            label[index] = -1
        output = np.expand_dims(output, 1)
        output = scaler.fit_transform(output)
        # for index in injection_2:
        #     # if random.randint(0, 1) == 0:
        #     #     output[index] = random.randint(5, 8)
        #     # else:
        #     #     output[index] = random.randint(-8, -5)
        #     output[index] = -5
        # output = np.expand_dims(output, 1)
        # output = scaler.fit_transform(output)
        # if verbose:
        #     plt.figure(figsize=(9, 3))
        #     plt.plot(t, output, 'k', lw=2)
        #     plt.xlabel('$t$')
        #     plt.ylabel('$s$')
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.show()
        #     plt.savefig('./figures/0/T_0.png')
        #     plt.close()
        # if verbose:
        #     linecolors = ['red' if i in injection_3 else 'blue' for i in range(249)]
        #     segments_y = np.r_[output[0], output[1:-1].repeat(2), output[-1]].reshape(-1, 2)
        #     segments_x = np.r_[t[0], t[1:-1].repeat(2), t[-1]].reshape(-1, 2)
        #     segments = [list(zip(x_, y_)) for x_, y_ in zip(segments_x, segments_y)]
        #
        #     plt.figure(figsize=(9, 3))
        #     # plt.margins(0.02)
        #     ax = plt.axes()
        #
        #     # Add a collection of lines
        #     ax.add_collection(LineCollection(segments, colors=linecolors, lw=2))
        #
        #     # Set x and y limits... sadly this is not done automatically for line
        #     # collections
        #     plt.yticks([0, 0.25, 0.5, 0.75, 1])
        #     ax.set_xlim(-10, 260)
        #     ax.set_ylim(-0.10, 1.10)
        #
        #     plt.xlabel('$t$')
        #     plt.ylabel('$s$')
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.margins(0.1)
        #
        #     plt.show()
        #     # plt.savefig('./figures/0/T_0.png')
        #     # plt.close()
    elif case == 1:
        t_1 = np.arange(0, length // 2)
        t_2 = np.arange(length // 2, length)
        trend_1 = 8 * np.sin(0.15 * t_1)
        trend_2 = 8 * np.sin(0.4 * t_2)
        t = np.concatenate([t_1, t_2])
        trend = np.concatenate([trend_1, trend_2])
        if noise:
            noise = 2 * (np.random.rand(length) - 0.2)
        output = trend + noise
        output = np.expand_dims(output, 1)
        scaler = MinMaxScaler()
        output = scaler.fit_transform(output)
        if verbose:
            plt.figure(figsize=(9, 3))
            plt.plot(t, output, lw=2)
            plt.xlabel('t')
            plt.ylabel('s')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            # plt.savefig('./figures/0/T_1.png')
            # plt.close()
    elif case == 2:
        t = np.arange(0, length)
        trend = 8 * np.sin(1.5 * t / 200 * np.sin(0.4 * t))
        if noise:
            noise = 2 * (np.random.rand(length) - 0.5)
        output = trend + noise
        output = np.expand_dims(output, 1)
        scaler = MinMaxScaler()
        output = scaler.fit_transform(output)
        if verbose:
            plt.figure(figsize=(9, 3))
            plt.plot(t, output, lw=2)
            plt.xlabel('t')
            plt.ylabel('s')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            # plt.savefig('./figures/0/T_2.png')
            # plt.close()
    elif case == 3:
        t_1 = np.arange(0, 120)
        t_2 = np.arange(120, 140)
        t_3 = np.arange(140, length)
        t = np.concatenate([t_1, t_2, t_3])
        trend_1 = 8 * np.sin(0.2 * t_1)
        trend_2 = 0 * t_2
        trend_3 = 8 * np.sin(0.2 * t_3)
        if noise:
            noise_1 = 3 * (np.random.rand(120-0) - 0.5)
            noise_3 = 3 * (np.random.rand(length - 140) - 0.5)
        output = np.concatenate([trend_1 + noise_1, trend_2, trend_3 + noise_3])
        output = np.expand_dims(output, 1)
        scaler = MinMaxScaler()
        output = scaler.fit_transform(output)
        if verbose:
            plt.figure(figsize=(9, 3))
            plt.plot(t, output, lw=2)
            plt.xlabel('t')
            plt.ylabel('s')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            # plt.savefig('./figures/0/T_3.png')
            # plt.close()
    elif case == 4:
        t = np.arange(0, length)
        trend = 0.001 * (t - length // 2) ** 2
        p1, p2 = 20, 30
        periodic1 = 2 * np.sin(2 * pi * t / p1)
        np.random.seed(123)  # So we generate the same noisy time series every time.
        if noise:
            noise = 2 * (np.random.rand(length) - 0.5)
        output = trend + periodic1 + noise
        output = np.expand_dims(output, 1)
        scaler = MinMaxScaler()
        output = scaler.fit_transform(output)
        if verbose:
            plt.figure(figsize=(9, 3))
            plt.plot(t, output, lw=2)
            plt.xlabel('t')
            plt.ylabel('s')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            # plt.savefig('./figures/0/T_4.png')
            # plt.close()
    return output.astype(np.float32), label


def get_loader(input, label=None, batch_size=128, shuffle=False, from_numpy=False, drop_last=False):
    """Convert input and label Tensors to a DataLoader

        If label is None, use a dummy Tensor
    """
    if label is None:
        label = input
    if from_numpy:
        input = torch.from_numpy(input)
        label = torch.from_numpy(label)
    loader = DataLoader(TensorDataset(input, label), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader


def cross_brain_get_loader(input_x, input_y, label=None, batch_size=128, shuffle=False, from_numpy=False, drop_last=False):
    """Convert input and label Tensors to a DataLoader
        If label is None, use a dummy Tensor
    """
    if label is None:
        label = input_x
    if from_numpy:
        input_x = torch.from_numpy(input_x)
        input_y = torch.from_numpy(input_y)
        label = torch.from_numpy(label)
    loader = DataLoader(TensorDataset(input_x, input_y, label), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader


def partition_data(ts, label, part_number=10):
    splitted_data = np.array_split(ts, part_number, axis=0)
    splitted_label = np.array_split(label, part_number, axis=0)
    return splitted_data, splitted_label


def create_batch_data(X, y=None, cutting_size=128, shuffle=False, from_numpy=False, drop_last=True):
    '''Convert X and y Tensors to a DataLoader

            If y is None, use a dummy Tensor
    '''
    if y is None:
        y = X
    if from_numpy:
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
    loader = DataLoader(TensorDataset(X, y), batch_size=cutting_size, shuffle=shuffle, drop_last=drop_last)
    b_X = []
    b_Y = []
    for i, (batch_X, batch_y) in enumerate(loader):
        b_X.append(batch_X)
        b_Y.append(batch_y)
    # return tensor
    b_X = torch.stack(b_X)
    b_Y = torch.stack(b_Y)
    return b_X, b_Y


def read_S5_dataset(file_name, normalize=True):
    abnormal = pd.read_csv(file_name, header=0, index_col=None)
    abnormal_data = abnormal['value'].values.astype(dtype='float32')
    abnormal_label = abnormal['is_anomaly'].values
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1

    abnormal_data = np.expand_dims(abnormal_data, axis=1)
    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if normalize == True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label[abnormal_label == 1] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label


def read_NAB_dataset(file_name, normalize=True):
    with open('../data/NAB/labels/combined_windows.json') as data_file:
        json_label = json.load(data_file)
    abnormal = pd.read_csv(file_name, header=0, index_col=0)
    abnormal['label'] = 1
    list_windows = json_label.get(os.path.basename(file_name))
    for window in list_windows:
        start = window[0]
        end = window[1]
        abnormal.loc[start:end, 'label'] = -1

    abnormal_data = abnormal['value'].values.astype(dtype='float32')
    # abnormal_preprocessing_data = np.reshape(abnormal_preprocessing_data, (abnormal_preprocessing_data.shape[0], 1))
    abnormal_label = abnormal['label'].values

    abnormal_data = np.expand_dims(abnormal_data, axis=1)
    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if normalize == True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    # Normal = 1, Abnormal = -1
    return abnormal_data, abnormal_label


def read_UAH_dataset(file_folder, normalize=True):
    def calculate_steering_angle(a):
        b = np.zeros(shape=(a.shape[0], 1))
        for i in range(a.size):
            if i == 0:
                b[i] = a[i]
            else:
                b[i] = (a[i] - a[i - 1])
                if b[i] >= 180:
                    b[i] = 360 - b[i]
                elif -180 < b[i] < 180:
                    b[i] = abs(b[i])
                elif b[i] <= -180:
                    b[i] = b[i] + 360
        return b

    def calculate_by_previous_element(a):
        b = np.zeros(shape=(a.shape[0], 1))
        for i in range(a.size):
            if i == 0:
                b[i] = 0
            else:
                b[i] = (a[i] - a[i - 1])
        return b

    def read_raw_GPS_dataset(folder_name):
        dataset = np.loadtxt(fname=folder_name + '/' + os.path.basename(folder_name) + '_RAW_GPS.txt', delimiter=' ',
                             usecols=(1, 7))
        return dataset

    def read_timestamp_and_label_of_semantic_dataset(folder_name):
        dataset = np.loadtxt(fname=folder_name + '/' + os.path.basename(folder_name) + '_SEMANTIC_ONLINE.txt',
                             delimiter=' ', usecols=(0, 23, 24, 25))
        return dataset

    def preprocess_raw_data(raw_data):
        speed_array = raw_data[:, 0]
        dir_array = raw_data[:, 1]

        # calculate acceleration (diff of speed)
        acceleration_array = calculate_by_previous_element(speed_array)

        # calculate jerk (diff of acceleration)
        jerk_array = calculate_by_previous_element(acceleration_array)

        # calculate steering (diff of direction)
        steering_array = calculate_steering_angle(dir_array)

        add_acceleration = np.c_[speed_array, acceleration_array]
        add_jerk = np.c_[add_acceleration, jerk_array]
        add_steering = np.c_[add_jerk, steering_array]

        return add_steering

    def compute_label_for_semantic(semantic_online_data):
        label = np.zeros(semantic_online_data.shape[0])
        for i in range(semantic_online_data.shape[0]):
            if semantic_online_data[i][0] <= semantic_online_data[i][1] or semantic_online_data[i][0] <= \
                    semantic_online_data[i][2] or semantic_online_data[i][0] <= semantic_online_data[i][1] + \
                    semantic_online_data[i][2]:
                label[i] = -1
            else:
                label[i] = 1
        return label

    abnormal = read_raw_GPS_dataset(file_folder)
    abnormal_data = preprocess_raw_data(abnormal)

    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label = read_timestamp_and_label_of_semantic_dataset(file_folder)
    abnormal_label_data = compute_label_for_semantic(abnormal_label[:, [1, 2, 3]])

    return abnormal_data, abnormal_label_data


def read_2D_dataset(file_name, normalize=True):
    file_name_wo_path = Path(file_name).name
    parent_path = Path(file_name).parent.parent
    train_frame = pd.read_csv(str(parent_path) + '/train/' + file_name_wo_path, header=None, index_col=None, sep=' ')
    test_frame = pd.read_csv(str(parent_path) + '/test/' + file_name_wo_path, skiprows=1, header=None, index_col=None, sep=' ')
    train_data = train_frame.iloc[:, [0, 1]].values.astype(dtype='float32')
    test_data = test_frame.iloc[:, [0, 1]].values.astype(dtype='float32')
    test_label = test_frame.iloc[:, 2].values
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1

    # abnormal_data = np.expand_dims(abnormal_data, axis=1)
    test_label = np.expand_dims(test_label, axis=1)

    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

    test_label[test_label == 2] = -1
    test_label[test_label == 0] = 1
    return train_data, test_data, test_label


def read_ECG_dataset(file_name, normalize=True):
    abnormal = pd.read_csv(file_name, header=None, index_col=None, skiprows=0, sep=',')
    abnormal_data = abnormal.iloc[:, [1, 2]].values.astype(dtype='float32')
    abnormal_label = abnormal.iloc[:, 3].values
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1

    # abnormal_data = np.expand_dims(abnormal_data, axis=1)
    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label[abnormal_label == 1] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label


def read_GD_dataset(file_name, normalize=True):
    abnormal = pd.read_csv(file_name, header=0, index_col=0)
    abnormal_data = abnormal[
        ['MotorData.ActCurrent', 'MotorData.ActPosition', 'MotorData.ActSpeed', 'MotorData.IsAcceleration',
         'MotorData.IsForce', 'MotorData.Motor_Pos1reached', 'MotorData.Motor_Pos2reached',
         'MotorData.Motor_Pos3reached',
         'MotorData.Motor_Pos4reached', 'NVL_Recv_Ind.GL_Metall', 'NVL_Recv_Ind.GL_NonMetall',
         'NVL_Recv_Storage.GL_I_ProcessStarted', 'NVL_Recv_Storage.GL_I_Slider_IN', 'NVL_Recv_Storage.GL_I_Slider_OUT',
         'NVL_Recv_Storage.GL_LightBarrier', 'NVL_Send_Storage.ActivateStorage', 'PLC_PRG.Gripper',
         'PLC_PRG.MaterialIsMetal']].values.astype(dtype='float32')
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label = abnormal['Label'].values
    # Normal = 0, Abnormal = 2 => # Normal = 1, Abnormal = -1

    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    abnormal_label[abnormal_label != 0] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label


def read_HSS_dataset(file_name, normalize=True):
    abnormal = pd.read_csv(file_name, header=0, index_col=0)
    abnormal_data = abnormal[
        ['I_w_BLO_Weg', 'O_w_BLO_power', 'O_w_BLO_voltage', 'I_w_BHL_Weg', 'O_w_BHL_power', 'O_w_BHL_voltage',
         'I_w_BHR_Weg', 'O_w_BHR_power', 'O_w_BHR_voltage', 'I_w_BRU_Weg', 'O_w_BRU_power', 'O_w_BRU_voltage',
         'I_w_HR_Weg', 'O_w_HR_power', 'O_w_HR_voltage', 'I_w_HL_Weg', 'O_w_HL_power', 'O_w_HL_voltage']].values.astype(dtype='float32')
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label = abnormal['Labels'].values
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1

    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    abnormal_label[abnormal_label != 0] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label


def read_SMD_dataset(file_name, normalize=True):
    file_name_wo_path = Path(file_name).name
    parent_path = Path(file_name).parent.parent
    train_data = pd.read_csv(str(parent_path) + '/train/' + file_name_wo_path, header=None, index_col=None)
    test_data = pd.read_csv(str(parent_path) + '/test/' + file_name_wo_path, header=None, index_col=None)
    test_label = pd.read_csv(str(parent_path) + '/test_label/' + file_name_wo_path, header=None, index_col=None)
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    test_label[test_label != 0] = -1
    test_label[test_label == 0] = 1
    return train_data.astype(dtype='float32'), test_data.astype(dtype='float32'), np.expand_dims(test_label.to_numpy(), axis=1)


def read_WADI_dataset(file_name, normalize=True, sampling=0.1):
    file_name_wo_path = Path(file_name).name
    parent_path = Path(file_name).parent.parent
    train_raw = pd.read_csv(str(parent_path) + '/train/' + file_name_wo_path, header=0, index_col=None).fillna(0)
    test_raw = pd.read_csv(str(parent_path) + '/test/' + file_name_wo_path, header=0, index_col=None).fillna(0)
    
    train_data = train_raw.iloc[::int(sampling*100)][train_raw.columns[3:]]
    test_data = test_raw.iloc[::int(sampling*100)][test_raw.columns[3:-1]]
    test_label = test_raw.iloc[::int(sampling*100)][test_raw.columns[-1]]
    
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

    return train_data.astype(dtype='float32'), test_data.astype(dtype='float32'), np.expand_dims(test_label.to_numpy(), axis=1)


def read_SWAT_dataset(file_name, normalize=True, sampling=0.1):
    file_name_wo_path = Path(file_name).name
    parent_path = Path(file_name).parent.parent
    train_raw = pd.read_csv(str(parent_path) + '/train/' + file_name_wo_path, header=0, index_col=None).fillna(0)
    test_raw = pd.read_csv(str(parent_path) + '/test/' + file_name_wo_path, header=0, index_col=None).fillna(0)
    
    train_data = train_raw.iloc[::int(sampling*100)][train_raw.columns[1:]]
    test_data = test_raw.iloc[::int(sampling*100)][test_raw.columns[1:-1]]
    test_label = test_raw.iloc[::int(sampling*100)][test_raw.columns[-1]]
    
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

    return train_data.astype(dtype='float32'), test_data.astype(dtype='float32'), np.expand_dims(test_label.to_numpy(), axis=1)


def read_SMAP_dataset(file_name, normalize=True):
    file_name_wo_path = Path(file_name).name
    file_name_wo_path_extension = Path(file_name).stem
    parent_path = Path(file_name).parent.parent
    train_data = np.load(str(parent_path) + '/train/' + file_name_wo_path)
    test_data = np.load(str(parent_path) + '/test/' + file_name_wo_path)
    test_label = pd.read_csv(str(parent_path.parent) + '/labeled_anomalies.csv', header=0, index_col=None)
    num_values = test_label.loc[test_label['chan_id'] == file_name_wo_path_extension]['num_values'].item()
    idx_anomalies = ast.literal_eval(test_label.loc[test_label['chan_id'] == file_name_wo_path_extension]['anomaly_sequences'].to_numpy()[0])
    labels = []
    j = 0
    for i in range(num_values):
        for idx in range(j, len(idx_anomalies)):
            if idx_anomalies[idx][0] < i < idx_anomalies[idx][1]:
                labels.append(-1)
                break
            else:
                labels.append(1)
                break
    labels = np.asarray(labels)
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    assert test_data.shape[0] == labels.shape[0]
    return train_data.astype(dtype='float32'), test_data.astype(dtype='float32'), np.expand_dims(labels, axis=1)

def read_MSL_dataset(file_name, normalize=True):
    file_name_wo_path = Path(file_name).name
    file_name_wo_path_extension = Path(file_name).stem
    parent_path = Path(file_name).parent.parent
    train_data = np.load(str(parent_path) + '/train/' + file_name_wo_path)
    test_data = np.load(str(parent_path) + '/test/' + file_name_wo_path)
    test_label = pd.read_csv(str(parent_path.parent) + '/labeled_anomalies.csv', header=0, index_col=None)
    num_values = test_label.loc[test_label['chan_id'] == file_name_wo_path_extension]['num_values'].item()
    idx_anomalies = ast.literal_eval(test_label.loc[test_label['chan_id'] == file_name_wo_path_extension]['anomaly_sequences'].to_numpy()[0])
    labels = []
    j = 0
    for i in range(num_values):
        for idx in range(j, len(idx_anomalies)):
            if idx_anomalies[idx][0] < i < idx_anomalies[idx][1]:
                labels.append(-1)
                break
            else:
                labels.append(1)
                break
    labels = np.asarray(labels)
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    return train_data.astype(dtype='float32'), test_data.astype(dtype='float32'), np.expand_dims(labels, axis=1)


def read_datasets(file_name, dataset):
    train_data = None
    abnormal_data = None
    abnormal_label = None
    if dataset == 0:
        abnormal_data, abnormal_label = generate_synthetic_dataset(case=0, length=512, noise=True, verbose=False)
    if dataset == 1:
        abnormal_data, abnormal_label = read_GD_dataset(file_name)
    if dataset == 2:
        abnormal_data, abnormal_label = read_HSS_dataset(file_name)
    if dataset == 3 or dataset == 31 or dataset == 32 or dataset == 33 or dataset == 34 or dataset == 35:
        abnormal_data, abnormal_label = read_S5_dataset(file_name)
    if dataset == 4 or dataset == 41 or dataset == 42 or dataset == 43 or dataset == 44 or dataset == 45 or dataset == 46:
        abnormal_data, abnormal_label = read_NAB_dataset(file_name)
    if dataset == 5 or dataset == 51 or dataset == 52 or dataset == 53 or dataset == 54 or dataset == 55 or dataset == 56 or dataset == 57:
        train_data, abnormal_data, abnormal_label = read_2D_dataset(file_name)
    if dataset == 6 or dataset == 61 or dataset == 62 or dataset == 63 or dataset == 64 or dataset == 65 or dataset == 66 or dataset == 67:
        abnormal_data, abnormal_label = read_ECG_dataset(file_name)
    if dataset == 7 or dataset == 71 or dataset == 72 or dataset == 73:
        train_data, abnormal_data, abnormal_label = read_SMD_dataset(file_name)
    if dataset == 8 or dataset == 81 or dataset == 82 or dataset == 83 or dataset == 84 or dataset == 85 or dataset == 86 or dataset == 87 or dataset == 88 or dataset == 89 or dataset == 90:
        train_data, abnormal_data, abnormal_label = read_SMAP_dataset(file_name)
    if dataset == 9 or dataset == 91 or dataset == 92 or dataset == 93 or dataset == 94 or dataset == 95 or dataset == 96 or dataset == 97:
        train_data, abnormal_data, abnormal_label = read_MSL_dataset(file_name)
    if dataset == 101:
        train_data, abnormal_data, abnormal_label = read_SWAT_dataset(file_name)
    if dataset == 11 or dataset == 111 or dataset == 112:
        train_data, abnormal_data, abnormal_label = read_WADI_dataset(file_name)
    return train_data, abnormal_data, abnormal_label


def get_data_path(dataset):
    path = None
    paths = None
    if dataset == 0:
        path = './data/synthetic'
    if dataset == 1:
        path = '../data/GD/data/Genesis_AnomalyLabels.csv'
    if dataset == 2:
        path = '../data/HSS/data/HRSS_anomalous_standard.csv'
    if dataset == 31:
        path = '../data/YAHOO/data/A1Benchmark'
    if dataset == 32:
        path = '../data/YAHOO/data/A2Benchmark'
    if dataset == 33:
        path = '../data/YAHOO/data/A3Benchmark'
    if dataset == 34:
        path = '../data/YAHOO/data/A4Benchmark'
    if dataset == 35:
        path = '../data/YAHOO/data/Vis'
    if dataset == 41:
        path = '../data/NAB/data/artificialWithAnomaly'
    if dataset == 42:
        path = '../data/NAB/data/realAdExchange'
    if dataset == 43:
        path = '../data/NAB/data/realAWSCloudwatch'
    if dataset == 44:
        path = '../data/NAB/data/realKnownCause'
    if dataset == 45:
        path = '../data/NAB/data/realTraffic'
    if dataset == 46:
        path = '../data/NAB/data/realTweets'
    if dataset == 51:
        path = '../data/2D/Comb'
    if dataset == 52:
        path = '../data/2D/Cross'
    if dataset == 53:
        path = '../data/2D/Intersection'
    if dataset == 54:
        path = '../data/2D/Pentagram'
    if dataset == 55:
        path = '../data/2D/Ring'
    if dataset == 56:
        path = '../data/2D/Stripe'
    if dataset == 57:
        path = '../data/2D/Triangle'
    if dataset == 61:
        path = '../data/ECG/chf01'
    if dataset == 62:
        path = '../data/ECG/chf13'
    if dataset == 63:
        path = '../data/ECG/ltstdb43'
    if dataset == 64:
        path = '../data/ECG/ltstdb240'
    if dataset == 65:
        path = '../data/ECG/mitdb180'
    if dataset == 66:
        path = '../data/ECG/stdb308'
    if dataset == 67:
        path = '../data/ECG/xmitdb108'
    if dataset == 71:
        path = '../data/SMD/machine1/train'
    if dataset == 72:
        path = '../data/SMD/machine2/train'
    if dataset == 73:
        path = '../data/SMD/machine3/train'
    if dataset == 81:
        path = '../data/SMAP/channel1/train'
    if dataset == 82:
        path = '../data/SMAP/channel2/train'
    if dataset == 83:
        path = '../data/SMAP/channel3/train'
    if dataset == 84:
        path = '../data/SMAP/channel4/train'
    if dataset == 85:
        path = '../data/SMAP/channel5/train'
    if dataset == 86:
        path = '../data/SMAP/channel6/train'
    if dataset == 87:
        path = '../data/SMAP/channel7/train'
    if dataset == 88:
        path = '../data/SMAP/channel8/train'
    if dataset == 89:
        path = '../data/SMAP/channel9/train'
    if dataset == 90:
        path = '../data/SMAP/channel10/train'
    if dataset == 91:
        path = '../data/MSL/channel1/train'
    if dataset == 92:
        path = '../data/MSL/channel2/train'
    if dataset == 93:
        path = '../data/MSL/channel3/train'
    if dataset == 94:
        path = '../data/MSL/channel4/train'
    if dataset == 95:
        path = '../data/MSL/channel5/train'
    if dataset == 96:
        path = '../data/MSL/channel6/train'
    if dataset == 97:
        path = '../data/MSL/channel7/train'
    if dataset == 101:
        path = '../data/SWaT/train'
    if dataset == 111:
        path = '../data/WADI/2017/train'
    if dataset == 112:
        path = '../data/WADI/2019/train'

    if dataset == 3:
        paths = ['./data/YAHOO/data/A1Benchmark', './data/YAHOO/data/A2Benchmark', './data/YAHOO/data/A3Benchmark', './data/YAHOO/data/A4Benchmark']
    if dataset == 4:
        paths = ['./data/NAB/data/artificialWithAnomaly', './data/NAB/data/realAdExchange', './data/NAB/data/realAWSCloudwatch', './data/NAB/data/realKnownCause', './data/NAB/data/realTraffic', './data/NAB/data/realTweets']
    if dataset == 5:
        paths = ['./data/2D/Comb', './data/2D/Cross', './data/2D/Intersection', './data/2D/Pentagram', './data/2D/Ring', './data/2D/Stripe', './data/2D/Triangle']
    if dataset == 6:
        paths = ['./data/ECG/chf01', './data/ECG/chf13', './data/ECG/ltstdb43', './data/ECG/ltstdb240', './data/ECG/mitdb180', './data/ECG/stdb308', './data/ECG/xmitdb108']
    if dataset == 7:
        paths = ['./data/SMD/machine1/train', './data/SMD/machine2/train', './data/SMD/machine3/train']
    if dataset == 8:
        paths = ['./data/SMAP/channel1/train', './data/SMAP/channel2/train', './data/SMAP/channel3/train', './data/SMAP/channel4/train', './data/SMAP/channel5/train', './data/SMAP/channel6/train', './data/SMAP/channel7/train', './data/SMAP/channel8/train', './data/SMAP/channel9/train', './data/SMAP/channel10/train']
    if dataset == 9:
        paths = ['./data/MSL/channel1/train', './data/MSL/channel2/train', './data/MSL/channel3/train', './data/MSL/channel4/train', './data/MSL/channel5/train', './data/MSL/channel6/train', './data/MSL/channel7/train']
    if dataset == 11:
        paths = ['./data/WADI/2017/train', './data/WADI/2019/train']

    return paths, path


def create_low_corr_tuples_h(input, k):
    '''
    :param x: time series [number of windows, widow size, number of channels]
    :return: x and y
    '''
    x = []
    y = []
    assert k % 2 == 0, 'k should be even number'
    sample_wise = np.array_split(input, k, axis=0)
    for i in range(k):
        if i < k // 2:
            x.append(sample_wise[i])
            y.append(sample_wise[i + k//2])
        else:
            x.append(sample_wise[i])
            y.append(sample_wise[i - k//2])
    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)


def realign_low_corr_tuples_h(input, k):
    '''
    :param x: time series [number of windows, widow size, number of channels]
    :return: x and reconstruct_x align
    '''
    x = []
    reconstruct_x = []
    assert k % 2 == 0, 'k should be even number'
    sample_wise = np.array_split(input, k, axis=0)
    for i in range(k):
        if i < k // 2:
            x.append(sample_wise[i])
            reconstruct_x.append(sample_wise[i + k//2])
        else:
            x.append(sample_wise[i])
            reconstruct_x.append(sample_wise[i - k//2])
    return np.concatenate(x, axis=0), np.concatenate(reconstruct_x, axis=0)


def create_low_corr_tuples_v(input):
    '''
    :param x: time series [number of windows, number of channels]
    :return: x and y
    '''
    assert input.shape[1] != 1, 'channel should be > 1'
    x = []
    y = []
    corr_matrix = np.corrcoef(input)
    channel_wise = np.split(input, axis=0)
    return 0


def create_cross_vertical_tuples(input, num_groups=1):
    '''
    :param input: time series [total length, number of channels]
    :param num_groups: number of groups
    :return: return the tuple of input and output
    '''
    # TODO use num_group
    channel_candidate = input.shape[1]
    left_channel_candidate = random.sample(sequence=channel_candidate, k=len(channel_candidate) // 2)
    right_channel_candidate = channel_candidate.remove(left_channel_candidate)

    return input[left_channel_candidate], input[right_channel_candidate]
