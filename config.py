# All configurations are listed here

class Config():
    def __init__(self):
        self.dataset_name = 'har'
        self.manual_dataset = True  # whether manually change the anomaly percentage in unlabeled dataset

        self.train_percentage = 0.8
        self.known_anomaly_num = 60  # number of known anomalies, default is same as DPLAN
        self.contamination_rate = 0.02  # anomaly percentage in unlabeled dataset, only take effect when manual_dataset=True
        self.device = 'cuda'
        self.known_anomaly_classes = {'ann': 2, 'cov': 4, 'car': 2, 'shu': 2, 'har': 2}  # class id of the known anomaly data
        self.normalization = True  # whether normalize the data