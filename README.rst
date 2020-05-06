===
gmr
===

.. image:: https://api.travis-ci.org/AlexanderFabisch/gmr.png?branch=master
   :target: https://travis-ci.org/AlexanderFabisch/gmr
   :alt: Travis
.. image:: https://landscape.io/github/AlexanderFabisch/gmr/master/landscape.svg?style=flat
   :target: https://landscape.io/github/AlexanderFabisch/gmr/master
   :alt: Code Health

Gaussian Mixture Models (GMMs) for clustering and regression in Python.

Original repository: https://github.com/AlexanderFabisch/gmr

Changes made from the original repository:

- Implementation of a method :code:`gmm.condition_derivative( indices, x )` to compute the gradient of the conditional expectation.
- Addition of example scripts to test the computation of the gradient in 1D and 2D.
- Computation and plotting of the log-likelihood at each iteration during training.
- Interruptible training.

.. image:: https://raw.githubusercontent.com/AlexanderFabisch/gmr/master/gmr.png

Example
-------

Estimate GMM from samples and sample from GMM::

    from gmr import GMM

    gmm = GMM(n_components=3, random_state=random_state)
    gmm.from_samples(X)
    X_sampled = gmm.sample(100)


For more details, see::

    help(gmr)

How Does It Compare to scikit-learn?
------------------------------------

There is an implementation of Gaussian Mixture Models for clustering in
`scikit-learn <http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html>`_
as well. Regression could not be easily integrated in the interface of
sklearn. That is the reason why I put the code in a separate repository.

Installation
------------

Install from `PyPI`_::

    sudo pip install gmr

or from source::

    sudo python setup.py install

.. _PyPi: https://pypi.python.org/pypi
