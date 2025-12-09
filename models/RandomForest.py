import sklearn.ensemble as sk

class RandomForest(sk.RandomForestClassifier):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(
            n_estimators=cfg.scorer.n_estimators,
            criterion=cfg.scorer.criterion,
            max_depth=cfg.scorer.max_depth,
            min_samples_split=cfg.scorer.min_samples_split,
            min_samples_leaf=cfg.scorer.min_samples_leaf,
            max_features=cfg.scorer.max_features,
            max_leaf_nodes=cfg.scorer.max_leaf_nodes,
        )
