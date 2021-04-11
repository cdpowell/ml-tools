from numpy import sum


class Regression(object):
    """Regression model class object.

    """

    def __init__(self, X, Y, thetas=None, alpha=0.0005, lam=None):
        """Initialization method for Regression class.

        """
        self.X = X
        self.Y = Y
        self.thetas = thetas
        self.alpha = alpha
        self.lam = lam

    def train(self, training_type="linear", iterations=10000, verbose=False):
        """Method for training the model on the self provided dataset for i iterations.

        """
        for i, _ in enumerate(range(iterations)):

            # standard linear regression
            if training_type == "linear":
                self.thetas = self._linear_train()
                mse = self.mean_squared_error()

                if verbose and (i + 1) % 100 == 0:
                    print("MSE at {:>7}: {}".format(i + 1, mse))

            # L2 regularized linear regression
            elif training_type == "regularized":
                self.thetas = self._regularized_linear_train()
                mse = self.reg_mean_squared_error()

                if verbose and (i+1) % 100 == 0:
                    print("MSE at {:>7}: {}".format(i+1, mse))

            else:
                print("Unknown training type \"{}\"".format(training_type))

    def mean_squared_error(self):
        """Method for calculating the mean squared error of the function.

        Where:
            X = [[ 1.         -1.43235849]
                 [ 1.         -1.84192139]
                 ...],
            X.shape[0] = 100,
            thetas = [[0, 0]],

        """
        return 1/self.X.shape[0] * sum((self.X.dot(self.thetas) - self.Y)**2)

    def reg_mean_squared_error(self):
        """Method for calculating the L2 regularized mean squared error of the function.

        """
        return 1/self.X.shape[0] * sum((self.X.dot(self.thetas) - self.Y)**2) + self.lam * sum(self.thetas)

    def _linear_train(self):
        """Method for training model based on linear regression.

        """
        # return self.thetas - self.alpha * 1 / self.X.shape[0] * sum((self.thetas * self.X - self.Y) * self.X, axis=0)
        return self.thetas - self.alpha * 1/self.X.shape[0] * sum((self.X.dot(self.thetas) - self.Y).dot(self.X), axis=0)

    def _regularized_linear_train(self):
        """Method for training model based on linear regression with L2 regularization.

        """
        return self.thetas - self.alpha * 1 / self.X.shape[0] * sum((self.X.dot(self.thetas) - self.Y).dot(self.X), axis=0) + self.lam * self.thetas