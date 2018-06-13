import numpy as np
from sklearn.feature_selection.univariate_selection import _BaseFilter, f_classif, _clean_nans
from sklearn.utils.validation import check_is_fitted


class SelectThreshold(_BaseFilter):
    """Select features according to the threshold.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
        Default is f_classif (see below "See also"). The default function only
        works with classification tasks.

    thresh : int or "none", optional, default=0.1
        Score threshold for feature inclusion.
        The "none" option bypasses selection, for use in a parameter search.

    Attributes
    ----------
    scores_ : array-like, shape=(n_features,)
        Scores of features.

    pvalues_ : array-like, shape=(n_features,)
        p-values of feature scores, None if `score_func` returned only scores.

    Notes
    -----
    Ties between features with equal scores will be broken in an unspecified
    way.

    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    mutual_info_regression: Mutual information for a continuous target.
    SelectPercentile: Select features based on percentile of the highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    GenericUnivariateSelect: Univariate feature selector with configurable mode.
    """

    def __init__(self, score_func=f_classif, thresh=0.1):
        super(SelectThreshold, self).__init__(score_func)
        self.thresh = thresh

    def _check_params(self, X, y):
        if not (self.thresh == "none" or 0 <= self.thresh <= 1):
            raise ValueError("thresh should be >=0, <= 1; got %r."
                             "Use thresh='none' to return all features."
                             % self.thresh)

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')

        if self.thresh == 'none':
            return np.ones(self.scores_.shape, dtype=bool)
        elif self.thresh == 0:
            return np.zeros(self.scores_.shape, dtype=bool)
        else:
            scores = _clean_nans(self.scores_)
            mask = np.zeros(scores.shape, dtype=bool)

            # Request a stable sort. Mergesort takes more memory (~40MB per
            # megafeature on x86-64).
            return np.array(scores > self.thresh, dtype=bool)
