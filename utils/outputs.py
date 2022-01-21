class VRAEOutput(object):
    def __init__(self, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,  best_pr_auc,
                 best_roc_auc, best_cks, training_time, testing_time, zs=None, z_infer_means=None, z_infer_stds=None, decs=None, dec_means=None,
                 dec_stds=None, kld_loss=None, nll_loss=None):
        self.zs = zs
        self.z_infer_means = z_infer_means
        self.z_infer_stds = z_infer_stds
        self.decs = decs
        self.dec_means = dec_means
        self.dec_stds = dec_stds
        self.kld_loss = kld_loss
        self.nll_loss = nll_loss
        self.best_TN = best_TN
        self.best_FP = best_FP
        self.best_FN = best_FN
        self.best_TP = best_TP
        self.best_precision = best_precision
        self.best_recall = best_recall
        self.best_fbeta = best_fbeta
        self.best_pr_auc = best_pr_auc
        self.best_roc_auc = best_roc_auc
        self.best_cks = best_cks
        self.training_time = training_time
        self.testing_time = testing_time

