import json


class VQRAEConfig(object):
    def __init__(self, dataset, x_dim, h_dim, z_dim, adversarial_training, preprocessing, use_overlapping, rolling_size,
                 epochs, milestone_epochs, lr, gamma, batch_size, weight_decay, annealing, early_stopping,
                 loss_function, lmbda, use_clip_norm, gradient_clip_norm, rnn_layers, use_PNF, PNF_layers,
                 use_bidirection, robust_coeff, display_epoch, save_output, save_figure, save_model, load_model,
                 continue_training, dropout, use_spot, use_last_point, save_config, load_config, server_run, robustness,
                 pid):

        self.dataset = dataset
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.adversarial_training = adversarial_training
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.annealing = annealing
        self.early_stopping = early_stopping
        self.loss_function = loss_function
        self.lmbda = lmbda
        self.use_clip_norm = use_clip_norm
        self.gradient_clip_norm = gradient_clip_norm
        self.rnn_layers = rnn_layers
        self.use_PNF = use_PNF
        self.PNF_layers = PNF_layers
        self.use_bidirection = use_bidirection
        self.robust_coeff = robust_coeff
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string
