import numpy as np
from scipy.stats import expon
from sklearn.preprocessing import scale


class DataGenerator:
    """Generate the synthetic distributions used in the figure experiments."""

    def __init__(self, num_samples, num_dimensions):
        self.num_samples = num_samples
        self.num_dimensions = num_dimensions
        self.matrix_axis = np.identity(self.num_dimensions)

    def gaussian_factor_analysis_data(self, t=0.05):
        n_samples = self.num_samples
        d = self.num_dimensions
        if d == 2:
            u = 0.92
            covariances = [np.array([[1, u], [u, 1]]), np.array([[1, -u], [-u, 1]])]
            samples = np.vstack([
                np.random.multivariate_normal([0, 0], covariances[0], size=n_samples // 2),
                np.random.multivariate_normal([0, 0], covariances[1], size=n_samples - n_samples // 2),
            ])
            samples = scale(samples)
        else:
            covariances = [
                np.diag([1 - t if i == j else t / (d - 1) for i in range(1, d + 1)])
                for j in range(1, d + 1)
            ]
            counts = [n_samples // d] * d
            counts[-1] += n_samples - sum(counts)
            samples = np.vstack([
                np.random.multivariate_normal(np.zeros(d), covariances[j], size=counts[j])
                for j in range(d)
            ])
        np.random.shuffle(samples)
        return samples

    def exponential_factor_analysis_data(self):
        n = self.num_samples
        d = self.num_dimensions
        samples = np.random.choice([-1, 1], size=(n, d)) * expon.rvs(size=(n, d)) ** 1.3
        if d == 2:
            samples = scale(samples)
            # Keep the base double-exponential data axis-aligned. Figure scripts
            # can apply their own controlled rotations when needed.
            # rotation = np.linalg.svd([[1, -2], [-3, 1]])[0]
            # samples = samples @ rotation
        return samples

    def student_t_data(self):
        n = self.num_samples
        d = self.num_dimensions
        samples = np.random.choice([-1, 1], size=(n, d)) * np.random.standard_t(df=3, size=(n, d))
        if d == 2:
            samples = scale(samples)
        return samples
