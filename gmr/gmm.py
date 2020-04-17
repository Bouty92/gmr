import numpy as np
from .utils import check_random_state, pinvh, Protect_loop
from .mvn import MVN, invert_indices


class GMM(object):
    """Gaussian Mixture Model.

    Parameters
    ----------
    n_components : int
        Number of MVNs that compose the GMM.

    priors : array, shape (n_components,), optional
        Weights of the components.

    means : array, shape (n_components, n_features), optional
        Means of the components.

    covariances : array, shape (n_components, n_features, n_features), optional
        Covariances of the components.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState, optional (default: global random state)
        If an integer is given, it fixes the seed. Defaults to the global numpy
        random number generator.
    """
    def __init__(self, n_components, priors=None, means=None, covariances=None,
                 verbose=0, random_state=None):
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    def _check_initialized(self):
        if self.priors is None:
            raise ValueError("Priors have not been initialized")
        if self.means is None:
            raise ValueError("Means have not been initialized")
        if self.covariances is None:
            raise ValueError("Covariances have not been initialized")

    def from_samples(self, X, R_diff=1e-4, n_iter=100, plot=True):
        """MLE of the mean and covariance.

        Expectation-maximization is used to infer the model parameters. The
        objective function is non-convex. Hence, multiple runs can have
        different results.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples from the true function.

        R_diff : float
            Minimum allowed difference of responsibilities between successive
            EM iterations.

        n_iter : int
            Maximum number of iterations.

        Returns
        -------
        log_likelihood : list of floats
            log of the likelihood at each iteration.
        """
        n_samples, n_features = X.shape

        if self.priors is None:
            self.priors = np.ones(self.n_components,
                                  dtype=np.float) / self.n_components

        if self.means is None:
            # TODO k-means++
            indices = self.random_state.choice(
                np.arange(n_samples), self.n_components)
            self.means = X[indices]

        if self.covariances is None:
            self.covariances = np.empty((self.n_components, n_features,
                                         n_features))
            for k in range(self.n_components):
                self.covariances[k] = np.eye(n_features)

        with Protect_loop() as interruption :
            if plot :
                import matplotlib.pyplot as plt

            R = np.zeros((n_samples, self.n_components))
            log_likelihood = []
            for i in range(n_iter):
                print( 'Iteration %i\r' % i, end='' )

                R_prev = R

                # Expectation
                R = self.to_responsibilities(X)

                if np.linalg.norm(R - R_prev) < R_diff:
                    if self.verbose:
                        print("EM converged.")
                    break

                # Maximization
                w = R.sum(axis=0) + 10.0 * np.finfo(R.dtype).eps
                R_n = R / w
                self.priors = w / w.sum()
                self.means = R_n.T.dot(X)
                for k in range(self.n_components):
                    Xm = X - self.means[k]
                    self.covariances[k] = (R_n[:, k, np.newaxis] * Xm).T.dot(Xm)

                likelihood = self.to_probability_density( X )
                likelihood += 1e-6
                log_likelihood.append( np.log( likelihood ).mean() )

                if plot :
                    plt.figure( 'log-likelihood of the model' )
                    plt.cla()
                    plt.plot( log_likelihood )
                    plt.draw()
                    plt.pause( 0.001 )

                if interruption() :
                    break

        print( '\nFinal log-likelihood: %g' % log_likelihood[-1] )

        return log_likelihood

    def sample(self, n_samples):
        """Sample from Gaussian mixture distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Samples from the GMM.
        """
        self._check_initialized()

        mvn_indices = self.random_state.choice(
            self.n_components, size=(n_samples,), p=self.priors)
        mvn_indices.sort()
        split_indices = np.hstack(
            ((0,), np.nonzero(np.diff(mvn_indices))[0] + 1, (n_samples,)))
        clusters = np.unique(mvn_indices)
        lens = np.diff(split_indices)
        samples = np.empty((n_samples, self.means.shape[1]))
        for i, (k, n_samples) in enumerate(zip(clusters, lens)):
            samples[split_indices[i]:split_indices[i + 1]] = MVN(
                mean=self.means[k], covariance=self.covariances[k],
                random_state=self.random_state).sample(n_samples=n_samples)
        return samples

    def to_responsibilities(self, X):
        """Compute responsibilities of each MVN for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        R : array, shape (n_samples, n_components)
        """
        self._check_initialized()

        n_samples = X.shape[0]
        R = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            R[:, k] = self.priors[k] * MVN(
                mean=self.means[k], covariance=self.covariances[k],
                random_state=self.random_state).to_probability_density(X)
        R_norm = R.sum(axis=1)[:, np.newaxis]
        R_norm[np.where(R_norm == 0.0)] = 1.0
        R /= R_norm
        return R

    def to_probability_density(self, X):
        """Compute probability density.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        p : array, shape (n_samples,)
            Probability densities of data.
        """
        self._check_initialized()

        p = [MVN(mean=self.means[k], covariance=self.covariances[k],
                 random_state=self.random_state).to_probability_density(X)
             for k in range(self.n_components)]
        return np.dot(self.priors, p)

    def condition(self, indices, x):
        """Conditional distribution over given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to condition.

        x : array, shape (n_new_features,)
            Values of the features that we know.

        Returns
        -------
        conditional : GMM
            Conditional GMM distribution p(Y | X=x).
        """
        self._check_initialized()

        n_features = self.means.shape[1] - len(indices)
        priors = np.empty(self.n_components)
        means = np.empty((self.n_components, n_features))
        covariances = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
                      random_state=self.random_state)
            conditioned = mvn.condition(indices, x)
            priors[k] = (self.priors[k] *
                         mvn.marginalize(indices).to_probability_density(x))
            means[k] = conditioned.mean
            covariances[k] = conditioned.covariance
        priors_sum = priors.sum()
        if priors_sum != 0 :
            priors /= priors_sum
        else :
            print( 'DIVISION_BY_ZERO_IN_CONDITION' )
            priors *= 0
        return GMM(n_components=self.n_components, priors=priors, means=means,
                   covariances=covariances, random_state=self.random_state)

    def predict(self, indices, X):
        """Predict means of posteriors.

        Same as condition() but for multiple samples.

        Parameters
        ----------
        indices : array, shape (n_features_1,)
            Indices of dimensions that we want to condition.

        X : array, shape (n_samples, n_features_1)
            Values of the features that we know.

        Returns
        -------
        Y : array, shape (n_samples, n_features_2)
            Predicted means of missing values.
        """
        self._check_initialized()

        n_samples, n_features_1 = X.shape
        n_features_2 = self.means.shape[1] - n_features_1
        Y = np.empty((n_samples, n_features_2))
        for n in range(n_samples):
            conditioned = self.condition(indices, X[n])
            Y[n] = conditioned.priors.dot(conditioned.means)
        return Y

    def condition_derivative( self, indices, x ) :
        """Provide the derivative of posterior means
           relative to the known features.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to condition.

        x : array, shape (n_new_features,)
            Values of the features that we know.

        Returns
        -------
        dYdX : array, shape (n_samples, n_features_2)
            Derivative of the means of missing values.
        """
        self._check_initialized()

        n_features = self.means.shape[1] - len( indices )
        i1 = invert_indices( self.means.shape[1], indices )
        i2 = indices

        priors = np.empty( self.n_components )
        prec_22 = []
        for k in range( self.n_components ):
            mvn = MVN( self.means[k], self.covariances[k], random_state=self.random_state )
            priors[k] = ( self.priors[k]*mvn.marginalize( i2 ).to_probability_density( x ) )

            cov_22 = self.covariances[k][np.ix_( i2, i2 )]
            prec_22.append( pinvh( cov_22 ) )

        dYdX = np.zeros(( len( i1 ), len( i2 ) ))
        for i in range( self.n_components ):
            cov_12 = self.covariances[i][np.ix_( i1, i2 )]

            dYdX += priors[i]/priors.sum()*cov_12.dot( prec_22[i] )

            dhdX = -priors.sum()*prec_22[i].dot( x - self.means[i][i2] )
            for k in range( self.n_components ):
                dhdX += priors[k]*prec_22[k].dot( x - self.means[k][i2] )

            priors_sum2 = priors.sum()**2
            if priors_sum2 != 0 :
                dhdX *= priors[i]/priors_sum2
            else :
                print( 'DIVISION_BY_ZERO_IN_CONDITION_DERIVATIVE' )
                dhdX *= 0

            dYdX += dhdX[:,np.newaxis].dot( ( self.means[i][i1] + cov_12.dot( prec_22[i].dot( x - self.means[i][i2] ) ).T )[np.newaxis,:] ).T

        return dYdX

    def to_ellipses(self, factor=1.0):
        """Compute error ellipses.

        An error ellipse shows equiprobable points.

        Parameters
        ----------
        factor : float
            One means standard deviation.

        Returns
        -------
        ellipses : array, shape (n_components, 3)
            Parameters that describe the error ellipses of all components:
            angles, widths and heights.
        """
        self._check_initialized()

        res = []
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
                      random_state=self.random_state)
            res.append((self.means[k], mvn.to_ellipse(factor)))
        return res


def plot_error_ellipses(ax, gmm, colors=None):
    """Plot error ellipses of GMM components.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    gmm : GMM
        Gaussian mixture model.
    """
    from matplotlib.patches import Ellipse
    from itertools import cycle
    if colors is not None:
        colors = cycle(colors)
    for factor in np.linspace(0.5, 4.0, 8):
        for mean, (angle, width, height) in gmm.to_ellipses(factor):
            ell = Ellipse(xy=mean, width=width, height=height,
                          angle=np.degrees(angle))
            ell.set_alpha(0.25)
            if colors is not None:
                ell.set_color(next(colors))
            ax.add_artist(ell)
