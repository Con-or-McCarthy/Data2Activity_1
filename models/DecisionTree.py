import sklearn.tree as skt

class DecisionTree(skt.DecisionTreeClassifier):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(
            criterion=cfg.scorer.criterion,
            min_samples_split=cfg.scorer.min_samples_split,
            min_samples_leaf=cfg.scorer.min_samples_leaf,
            random_state=cfg.scorer.random_state,
            class_weight=cfg.scorer.class_weight,
        )
