import numpy as np
import lightgbm as lgb
from ngboost.ngboost import NGBoost
from ngboost.distns import Bernoulli, Normal, LogNormal
from ngboost.scores import MLE
from ngboost.learners import default_tree_learner
from sklearn.base import BaseEstimator


class NGBRegressorLGB(NGBoost, BaseEstimator):

    def __init__(self,
                 lgb_param,
                 Dist=Normal,
                 Score=MLE,
                 Base=default_tree_learner,
                 natural_gradient=True,
                 n_estimators=500,
                 learning_rate=0.01,
                 minibatch_frac=1.0,
                 verbose=True,
                 verbose_eval=100,
                 tol=1e-4):
        assert Dist.problem_type == "regression"
        super().__init__(Dist, Score, Base, natural_gradient, n_estimators, learning_rate,
                         minibatch_frac, verbose, verbose_eval, tol)

        self.lgb_param = lgb_param


    def dist_to_prediction(self, dist): # predictions for regression are typically conditional means
        return dist.mean()

    def fit_base(self, dataset_tr, X, grads, sample_weight=None):
        models = list()
        for g in grads.T:
            dataset_tr.set_label(g)
            f_model = lgb.train(
                    self.lgb_param,
                    dataset_tr,
                    verbose_eval=True,
                    valid_sets= [dataset_tr],
                    valid_names = ['train'],
                    num_boost_round = self.lgb_param.get('num_round', 1)
                    )
            models.append(f_model)

        fitted = np.array([m.predict(X) for m in models]).T
        self.base_models.append(models)
        return fitted

    def fit(self, X_tr, Y_tr, X_val = None, Y_val = None, 
            sample_weight = None, val_sample_weight = None,
            train_loss_monitor = None, val_loss_monitor = None, 
            early_stopping_rounds = None,
            callbacks=[]):

        X = X_tr
        Y = Y_tr
        dataset = lgb.Dataset(X)

        loss_list = []
        val_loss_list = []

        if early_stopping_rounds is not None:
            best_val_loss = np.inf

        self.fit_init_params_to_marginal(Y)

        params = self.pred_param(X)
        if X_val is not None and Y_val is not None:
            val_params = self.pred_param(X_val)

        S = self.Score

        if not train_loss_monitor:
            train_loss_monitor = lambda D,Y: S.loss(D, Y, sample_weight=sample_weight)

        if not val_loss_monitor:
            val_loss_monitor = lambda D,Y: S.loss(D, Y, sample_weight=val_sample_weight)

        for itr in range(self.n_estimators):
            self.iteration = itr
            if len(callbacks) > 0:
                for callback in callbacks:
                    callback(self)

            D = self.Dist(params.T)

            loss_list += [train_loss_monitor(D, Y)]
            loss = loss_list[-1]
            grads = S.grad(D, Y, natural=self.natural_gradient)

            proj_grad = self.fit_base(dataset, X, grads, sample_weight)
            scale = self.line_search(proj_grad, params, Y, sample_weight)

            # pdb.set_trace()
            params -= self.learning_rate * scale * np.array([m.predict(X) for m in self.base_models[-1]]).T

            val_loss = 0
            if X_val is not None and Y_val is not None:
                val_params -= self.learning_rate * scale * np.array([m.predict(X_val) for m in self.base_models[-1]]).T
                val_loss = val_loss_monitor(self.Dist(val_params.T), Y_val)
                val_loss_list += [val_loss]
                if early_stopping_rounds is not None:
                    if val_loss < best_val_loss:
                        best_val_loss, self.best_val_loss_itr = val_loss, itr
                    if best_val_loss < np.min(np.array(val_loss_list[-early_stopping_rounds:])):
                        if self.verbose:
                            print(f"== Early stopping achieved.")
                            print(f"== Best iteration / VAL {self.best_val_loss_itr} (val_loss={best_val_loss:.4f})")
                        break

            if self.verbose and int(self.verbose_eval) > 0 and itr % int(self.verbose_eval) == 0:
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} scale={scale:.4f} "
                      f"norm={grad_norm:.4f}")

            if np.linalg.norm(proj_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration / GRAD {itr}")
                break

        self.evals_result = {}
        metric = self.Score.__name__.upper()
        self.evals_result['train'] = {metric: loss_list}
        if X_val is not None and Y_val is not None:
            self.evals_result['val'] = {metric: val_loss_list}

        return self

    def pred_dist(self, X, max_iter=None):
        if max_iter is not None: # get prediction at a particular iteration if asked for
            dist = self.staged_pred_dist(X, max_iter=max_iter)[-1]
        elif self.best_val_loss_itr is not None: # this will exist if there's a validation set 
            dist = self.staged_pred_dist(X, max_iter=self.best_val_loss_itr)[-1]
        else: 
            params = np.asarray(self.pred_param(X, max_iter))
            dist = self.Dist(params.T)
        return dist

    def staged_pred_dist(self, X, max_iter=None):
        predictions = []
        m, n = X.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, s) in enumerate(zip(self.base_models, self.scalings)):
            resids = np.array([model.predict(X) for model in models]).T
            params -= self.learning_rate * resids * s
            dists = self.Dist(np.asarray(params).T)
            predictions.append(dists)
            if max_iter and i == max_iter:
                break
        return predictions

    def save_model(self, path):
        import joblib
        joblib.dump(self, path)
