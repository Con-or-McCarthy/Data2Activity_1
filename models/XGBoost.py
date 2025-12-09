import xgboost as xgb

class XGBoost(xgb.XGBClassifier):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(
            device=cfg.eval.device,
            eta=cfg.scorer.learning_rate,
            max_depth=cfg.scorer.max_depth,
            n_estimators=cfg.scorer.n_estimators,
            objective=cfg.scorer.objective,
            use_label_encoder=False,
            seed=cfg.scorer.seed,
            deterministic=cfg.scorer.deterministic,
            nthread=cfg.scorer.nthread
        )
