import numpy as np

class BLModel:
    def __init__(self, hot_rate_matrix, return_array, tau=0.05):
        """
        hot_rate_matrix: np.ndarray, shape=(n_samples, n_features)
        return_array: np.ndarray, shape=(n_samples,)
        tau: float, BL tau parameter
        """
        self.hot_rate = hot_rate_matrix
        self.returns = return_array
        self.tau = tau
        self.fitted = False

    def fit(self):
        # 1. Market equilibrium return (prior)
        self.prior_mean = np.mean(self.returns)
        self.prior_var = np.var(self.returns)

        # 2. Multivariate linear regression: returns ~ hot_rate_1 + ... + hot_rate_5
        X = self.hot_rate
        y = self.returns
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        self.reg_intercept = beta[0]
        self.reg_coef = beta[1:]

        # 3. View prediction function
        self._predict_view = lambda hot: self.reg_intercept + np.dot(self.reg_coef, hot)

        # 4. Omega: Variance of residuals
        resid = y - (self.reg_intercept + np.dot(X, self.reg_coef))
        self.omega = np.var(resid)

        self.fitted = True

    def predict_next_return(self, next_hot_rate_vec):
        """
        next_hot_rate_vec: np.ndarray, shape=(n_features,)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")

        view = self._predict_view(next_hot_rate_vec)
        tau_var = self.tau * self.prior_var
        posterior = (tau_var * view + self.omega * self.prior_mean) / (tau_var + self.omega)
        return posterior
