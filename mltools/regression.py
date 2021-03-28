from numpy import array, c_, flip, genfromtxt, hsplit, ones, subtract, sum, zeros
from matplotlib.pyplot import legend, plot, scatter, show


class Regression(object):
    """Regression model class object.

    """

    def __init__(self, X, Y, thetas=None, alpha=0.0005):
        """Initialization method for Regression class.

        """
        self.X = X
        self.Y = Y
        self.thetas = thetas
        self.alpha = alpha

    def train(self, reg_type="linear", iterations=10000, verbose=False):
        """Method for training the model on the self provided dataset for i iterations.

        """
        for i, _ in enumerate(range(iterations)):

            if reg_type == "linear":
                self.thetas = self._linear_train()
            elif reg_type == "polynomial":
                self.thetas = self._poly_train()

            if verbose and (i+1) % 100 == 0:
                mse = self.mean_squared_error()
                print("MSE at {:>7}: {}".format(i+1, mse))

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

    def _linear_train(self):
        """Method for training model using based on linear regression.

        """
        # return self.thetas - self.alpha * 1 / self.X.shape[0] * sum((self.thetas * self.X - self.Y) * self.X, axis=0)
        return self.thetas - self.alpha * 1/self.X.shape[0] * sum((self.X.dot(self.thetas) - self.Y).dot(self.X), axis=0)

    def _poly_train(self):
        """Method for training model using based on polynomial regression.

        """
        return self.thetas - self.alpha * 1 / self.X.shape[0] * sum((self.thetas * self.X - self.Y) * (self.X ** 1), axis=0)
