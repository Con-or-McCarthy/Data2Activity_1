import catboost as cb

class CatBoost(cb.CatBoostClassifier):
    def __init__(self, cfg, is_multiclass=False, class_weights=None):
        self.cfg = cfg
        loss_function = cfg.scorer.loss_function_multiclass if is_multiclass else cfg.scorer.loss_function
        super().__init__(
            iterations=cfg.scorer.iterations,
            depth=cfg.scorer.depth,
            learning_rate=cfg.scorer.learning_rate,
            loss_function=loss_function,
            nan_mode=cfg.scorer.nan_mode,
            l2_leaf_reg=cfg.scorer.l2_leaf_reg,
            min_child_samples=cfg.scorer.min_child_samples,
            rsm=cfg.scorer.rsm,
            verbose=False,
            random_seed=cfg.scorer.random_seed,
            thread_count=cfg.scorer.thread_count,
            class_weights=class_weights
        )

    def fit(self, X, y, **kwargs):
        if len(self.cfg.data.categorical_vars) > 0:
            cat_features = self.cfg.data.categorical_vars
            cat_features = [feat for feat in cat_features if feat in self.cfg.data.vars_to_use]
            
            # Convert categorical features to string type if they're float
            # This ensures CatBoost treats them as categorical
            X_copy = X.copy()
            for col in cat_features:
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].astype(str)
            
            # Create Pool with categorical features
            train_pool = cb.Pool(data=X_copy, 
                                label=y, 
                                cat_features=cat_features)
            
            # Call the parent fit method with the pool
            return super().fit(train_pool, **kwargs)
        else:
            # If no categorical features, just call the parent fit method
            return super().fit(X, y, **kwargs)

    def predict_proba(self, X, **kwargs):
        if len(self.cfg.data.categorical_vars) > 0 and not isinstance(X, cb.Pool):
            cat_features = self.cfg.data.categorical_vars
            cat_features = [feat for feat in cat_features if feat in self.cfg.data.vars_to_use]
            
            # Convert categorical features to string type if they're float
            # This ensures CatBoost treats them as categorical
            X_copy = X.copy()
            for col in cat_features:
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].astype(str)
            
            # Create Pool with categorical features
            test_pool = cb.Pool(data=X_copy, 
                                cat_features=cat_features)
            
            # Call the parent predict_proba method with the pool
            return super().predict_proba(test_pool, **kwargs)
        else:
            # If no categorical features, just call the parent predict_proba method
            return super().predict_proba(X, **kwargs)

    def predict(self, X, **kwargs):
        if len(self.cfg.data.categorical_vars) > 0 and not isinstance(X, cb.Pool):
            cat_features = self.cfg.data.categorical_vars
            cat_features = [feat for feat in cat_features if feat in self.cfg.data.vars_to_use]
            
            # Convert categorical features to string type if they're float
            # This ensures CatBoost treats them as categorical
            X_copy = X.copy()
            for col in cat_features:
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].astype(str)
            
            # Create Pool with categorical features
            test_pool = cb.Pool(data=X_copy, 
                                cat_features=cat_features)

            # Call the parent predict method with the pool
            return super().predict(test_pool, **kwargs)
        else:
            # If no categorical features, just call the parent predict method
            return super().predict(X, **kwargs)
