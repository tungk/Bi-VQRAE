import logging
import os


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    log_format = (
        '%(asctime)s - '
        '%(name)s - '
        '%(funcName)s - '
        '%(levelname)s - '
        '%(message)s'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def create_logger(dataset, train_logger_name, file_logger_name, meta_logger_name, model_name, pid):
    if not os.path.exists('./logs/{}/'.format(dataset)):
        os.makedirs('./logs/{}/'.format(dataset))

    if dataset == 0:
        # first file logger
        file_logger = setup_logger(file_logger_name + model_name + str(pid),
                                   './logs/{}/file_synthetic_{}_{}.log'.format(dataset, model_name, pid))
        # second file logger
        meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                                   './logs/{}/meta_synthetic_{}_{}.log'.format(dataset, model_name, pid))
        # third file logger
        train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                    './logs/{}/train_synthetic_{}_{}.log'.format(dataset, model_name, pid))
    elif dataset == 1:
        # first file logger
        file_logger = setup_logger(file_logger_name + model_name + str(pid),
                                   './logs/{}/file_GD_synthetic_{}_{}.log'.format(dataset, model_name, pid))
        # second file logger
        meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                                   './logs/{}/meta_GD_synthetic_{}_{}.log'.format(dataset, model_name, pid))
        # third file logger
        train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                    './logs/{}/train_GD_synthetic_{}_{}.log'.format(dataset, model_name, pid))
    elif dataset == 2:
        # first file logger
        file_logger = setup_logger(file_logger_name + model_name + str(pid),
                                   './logs/{}/file_HSS_synthetic_{}_{}.log'.format(dataset, model_name, pid))
        # second file logger
        meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                                   './logs/{}/meta_HSS_synthetic_{}_{}.log'.format(dataset, model_name, pid))
        # third file logger
        train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                    './logs/{}/train_HSS_synthetic_{}_{}.log'.format(dataset, model_name, pid))
    elif dataset == 3 or dataset == 31 or dataset == 32 or dataset == 33 or dataset == 34 or dataset == 35:
        if dataset == 3:
            subset = 'YAHOO_ALL'
        if dataset == 31:
            subset = 'A1Benchmark'
        if dataset == 32:
            subset = 'A2Benchmark'
        if dataset == 33:
            subset = 'A3Benchmark'
        if dataset == 32:
            subset = 'A4Benchmark'
        if dataset == 35:
            subset = 'Vis'
        # first file logger
        file_logger = setup_logger(file_logger_name + model_name + str(pid),
                                   './logs/{}/file_YAHOO_{}_{}.log'.format(dataset, model_name, pid))
        # second file logger
        meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                                   './logs/{}/meta_YAHOO_{}_{}.log'.format(dataset, model_name, pid))
        # third file logger
        train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                    './logs/{}/train_YAHOO_{}_{}.log'.format(dataset, model_name, pid))
    elif dataset == 4 or dataset == 41 or dataset == 42 or dataset == 43 or dataset == 44 or dataset == 45 or dataset == 46:
        if dataset == 41:
            subset = 'NAB_ALL'
        if dataset == 41:
            subset = 'artificialWithAnomaly'
        if dataset == 42:
            subset = 'realAdExchange'
        if dataset == 43:
            subset = 'realAWSCloudwatch'
        if dataset == 44:
            subset = 'realKnownCause'
        if dataset == 45:
            subset = 'realTraffic'
        if dataset == 46:
            subset = 'realTweets'
        # first file logger
        file_logger = setup_logger(file_logger_name + model_name + str(pid),
                                   './logs/{}/file_NAB_{}_{}.log'.format(dataset, model_name, pid))
        # second file logger
        meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                                   './logs/{}/meta_NAB_{}_{}.log'.format(dataset, model_name, pid))
        # third file logger
        train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                    './logs/{}/train_NAB_{}.log'.format(dataset, model_name, pid))
    elif dataset == 5 or dataset == 51 or dataset == 52 or dataset == 53 or dataset == 54 or dataset == 55 or dataset == 56 or dataset == 57:
        if dataset == 5:
            subset = '2D_ALL'
        if dataset == 51:
            subset = 'Comb'
        if dataset == 52:
            subset = 'Cross'
        if dataset == 53:
            subset = 'Intersection'
        if dataset == 54:
            subset = 'Pentagram'
        if dataset == 55:
            subset = 'Ring'
        if dataset == 56:
            subset = 'Stripe'
        if dataset == 57:
            subset = 'Triangle'
        # first file logger
        file_logger = setup_logger(file_logger_name + model_name + str(pid),
                                   './logs/{}/file_2D_{}_{}.log'.format(dataset, model_name, pid))
        # second file logger
        meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                                   './logs/{}/meta_2D_{}_{}.log'.format(dataset, model_name, pid))
        # third file logger
        train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                    './logs/{}/train_2D_{}_{}.log'.format(dataset, model_name, pid))
    elif dataset == 6 or dataset == 61 or dataset == 62 or dataset == 63 or dataset == 64 or dataset == 65 or dataset == 66 or dataset == 67:
        if dataset == 6:
            subset = 'ECG_ALL'
        if dataset == 61:
            subset = 'chf01'
        if dataset == 62:
            subset = 'chf13'
        if dataset == 63:
            subset = 'ltstdb43'
        if dataset == 64:
            subset = 'ltstdb240'
        if dataset == 65:
            subset = 'mitdb180'
        if dataset == 66:
            subset = 'stdb308'
        if dataset == 67:
            subset = 'xmitdb108'
        # first file logger
        file_logger = setup_logger(file_logger_name + model_name + str(pid),
                                   './logs/{}/file_ECG_{}_{}.log'.format(dataset, model_name, pid))
        # second file logger
        meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                                   './logs/{}/meta_ECG_{}_{}.log'.format(dataset, model_name, pid))
        # third file logger
        train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                    './logs/{}/train_ECG_{}_{}.log'.format(dataset, model_name, pid))
    elif dataset== 7 or dataset == 71 or dataset == 72 or dataset == 73:
        if dataset == 7:
            subset = 'SMD_ALL'
        if dataset == 71:
            subset = 'machine1'
        if dataset == 72:
            subset = 'machine2'
        if dataset == 73:
            subset = 'machine3'
        # first file logger
        file_logger = setup_logger(file_logger_name + model_name + str(pid),
                                   './logs/{}/file_SMD_{}_{}.log'.format(dataset, model_name, pid))
        # second file logger
        meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                                   './logs/{}/meta_SMD_{}_{}.log'.format(dataset, model_name, pid))
        # third file logger
        train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                    './logs/{}/train_SMD_{}_{}.log'.format(dataset, model_name, pid))
    elif dataset == 8 or dataset == 81 or dataset == 82 or dataset == 83 or dataset == 84 or dataset == 85 or dataset == 86 or dataset == 87 or dataset == 88 or dataset == 89 or dataset == 90:
        if dataset == 81:
            subset = 'SMAP_ALL'
        if dataset == 81:
            subset = 'channel1'
        if dataset == 82:
            subset = 'channel2'
        if dataset == 83:
            subset = 'channel3'
        if dataset == 84:
            subset = 'channel4'
        if dataset == 85:
            subset = 'channel5'
        if dataset == 86:
            subset = 'channel6'
        if dataset == 87:
            subset = 'channel7'
        if dataset == 88:
            subset = 'channel8'
        if dataset == 89:
            subset = 'channel9'
        if dataset == 90:
            subset = 'channel10'
        # first file logger
        file_logger = setup_logger(file_logger_name + model_name + str(pid),
                                   './logs/{}/file_SMAP_{}_{}.log'.format(dataset, model_name, pid))
        # second file logger
        meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                                   './logs/{}/meta_SMAP_{}_{}.log'.format(dataset, model_name, pid))
        # third file logger
        train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                    './logs/{}/train_SMAP_{}_{}.log'.format(dataset, model_name, pid))
    elif dataset == 9 or dataset == 91 or dataset == 92 or dataset == 93 or dataset == 94 or dataset == 95 or dataset == 96 or dataset == 97:
        if dataset == 9:
            subset = 'MSL_ALL'
        if dataset == 91:
            subset = 'channel1'
        if dataset == 92:
            subset = 'channel2'
        if dataset == 93:
            subset = 'channel3'
        if dataset == 94:
            subset = 'channel4'
        if dataset == 95:
            subset = 'channel5'
        if dataset == 96:
            subset = 'channel6'
        if dataset == 97:
            subset = 'channel7'
        # first file logger
        file_logger = setup_logger(file_logger_name + model_name + str(pid),
                                   './logs/{}/file_MSL_{}_{}.log'.format(dataset, model_name, pid))
        # second file logger
        meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                                   './logs/{}/meta_MSL_{}_{}.log'.format(dataset, model_name, pid))
        # third file logger
        train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                    './logs/{}/train_MSL_{}_{}.log'.format(dataset, model_name, pid))
    elif dataset == 101:
        # first file logger
        file_logger = setup_logger(file_logger_name + model_name + str(pid),
                                   './logs/{}/file_SWAT_synthetic_{}_{}.log'.format(dataset, model_name, pid))
        # second file logger
        meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                                   './logs/{}/meta_SWAT_synthetic_{}_{}.log'.format(dataset, model_name, pid))
        # third file logger
        train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                    './logs/{}/train_SWAT_synthetic_{}_{}.log'.format(dataset, model_name, pid))
    elif dataset == 11 or dataset == 111 or dataset == 112:
        if dataset == 11:
            subset = 'WADI_ALL'
        if dataset == 111:
            subset = '2017'
        if dataset == 112:
            subset = '2019'
        # first file logger
        file_logger = setup_logger(file_logger_name + model_name + str(pid),
                                   './logs/{}/file_WADI_synthetic_{}_{}.log'.format(dataset, model_name, pid))
        # second file logger
        meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                                   './logs/{}/meta_WADI_synthetic_{}_{}.log'.format(dataset, model_name, pid))
        # third file logger
        train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                    './logs/{}/train_WADI_synthetic_{}_{}.log'.format(dataset, model_name, pid))
    return train_logger, file_logger, meta_logger