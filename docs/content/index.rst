Score Analysis
==============

Package to analyse ML model results. Contains an efficient implementation of common
metrics computations such as TPR, FPR, EER, methods for threshold setting and methods
to compute confidence intervals.

Terminology
-----------

Sometimes, we like to work with metrics based on acceptance and rejection, such as
FAR (false acceptance rate) and FRR (false rejection rate), while the standard ML
terminology talks about positive and negative classes and FPR (false positive rate) and
FNR (false negative rate).

This library adopts the standard ML terminology. The translation is simple: just replace
"accept" with "positive" and replace "reject" with "negative" and you have a dictionary
between the two worlds.

The library is also agnostic to which direction scores are pointing. It works with
scores that indicate membership of the positive (accept) class as well as with scores
that indicate membership of the negative (reject) class. The score interpretation is
set using the :code:`score_class` parameter when constructing a :code:`Scores` object.

The key is to decouple the process of computing scores from the process of interpreting
them. When we compute scores, e.g., using an ML model, some will point towards genuines,
some towards spoofs/fraud. Sometimes we use score-mappers that reverse the score
orientation. We cannot change scores. But, when we move towards interpreting them, we
should always use a fixed terminology: positive class means accept/genuine; negative
class means reject/spoof/fraud. And at the point, when we go from generating scores to
interpreting them, we set, via the `score_class` parameter, how scores are to be
interpreted.

Scores
------

We assume that we work with a binary classification problem. First, we create a `Scores`
object with the experiment results. We can do this in two ways.

.. code-block:: python

    from score_analysis import Scores

    # If we have the scores for the positive and negative classes separately
    scores = Scores(pos=[1, 2, 3], neg=[0.5, 1.5])

    # If we have an intermingled set of scores and labels
    scores = Scores.from_labels(
        labels=[1, 1, 1, 0, 0],
        scores=[1, 2, 3, 0.5, 1.5],
        # We specify the label of the positive class. All other labels are assigned to
        # the negative class.
        pos_label=1,
    )


There are two parameters, that will determine calculations of metrics:

* Do scores indicate membership of the positive or the negative class
  (:code:`score_class`)
* If a score is exactly equal to the threshold, will it be assigned to the positive
  or negative class (:code:`equal_class`)

The meaning of the parameters is summarized in the table

+-------------+-------------+-----------------------------------+
| score_class | equal_class | Decision logic for positive class |
+=============+=============+===================================+
|     pos     |     pos     |        score >= threshold         |
+-------------+-------------+-----------------------------------+
|     pos     |     neg     |         score > threshold         |
+-------------+-------------+-----------------------------------+
|     neg     |     pos     |        score <= threshold         |
+-------------+-------------+-----------------------------------+
|     neg     |     neg     |         score < threshold         |
+-------------+-------------+-----------------------------------+

We can apply a threshold to a `Scores` object to obtain a confusion matrix and then
compute metrics associated to the confusion matrix.

.. code-block:: python

    cm = scores.cm(threshold=2.5)
    print(cm.fpr())  # Print False Positive Rate

We can work with multiple thresholds at once, which leads to vectorized confusion
matrices. In fact, the :code:`threshold` parameter accepts arbitrary-shaped arrays and
all confusion matrix operations preserve the shapes.

.. code-block:: python

    threshold = np.linspace(0, 3, num=50)
    cm = scores.cm(threshold=threshold)
    fpr = cm.fpr()  # Contains FPRs at all defined thresholds
    assert fpr.shape == threshold.shape

We can also determine thresholds at specific operating points. These operations are also
fully vectorized.

.. code-block:: python

    # Calculate threshold at 30% False Positive Rate
    threshold = scores.threshold_at_fpr(fpr=0.3)

    # Calculate thresholds at logarithmically spaced FPR intervals from 0.1% to 100%
    fpr = np.logspace(-3, 0, num=50)
    threshold = scores.threshold_at_fpr(fpr)

Note that determining thresholds a fixed operating points requires interpolation, since
with a finite dataset we can measure only finitely many values for FPR, etc. If we want
to determine a threshold at any other value for the target metric, we use linear
interpolation.

Confusion matrices
------------------

Most metrics that we use are defined via confusion matrices. We can create a confusion
matrix either from vectors with labels and predictions or directly from a matrix.

.. code-block:: python

    >>> labels      = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    >>> predictions = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
    >>> cm = ConfusionMatrix(labels=labels, predictions=predictions)
    >>> cm.classes
    [0, 1, 2]
    >>> cm.matrix
    array([[3, 0, 0],
           [0, 1, 2],
           [2, 1, 3]])

A binary confusion matrix is a special case of a :code:`ConfusionMatrix`, with specially
designated positive and negative classes. The convention is that the classes are
ordered :code:`classes = [pos, neg]`. It can be created with the parameter
:code:`binary=True`.

For binary confusion matrices all metrics such as TPR are scalar. Since we have defined
which is the positive class, there is no need to use the one-vs-all strategy.

A binary confusion matrix is different from a regular confusion matrix with two classes,
since the latter does not have designated positive and negative classes.

.. code-block:: python

    >>> cm = ConfusionMatrix(matrix=[[1, 4], [2, 3]], binary=True)
    >>> cm.tpr()
    0.2
    >>> cm = ConfusionMatrix(matrix=[[1, 4], [2, 3]])
    >>> cm.tpr()  # True positive rate for each class
    array([0.2, 0.6])

Available metrics
-----------------

Basic parameters

1. TP (true positive)
2. TN (true negative)
3. FP (false positive)
4. FN (false negative)
5. P (condition positive)
6. N (condition negative)
7. TOP (test outcome positive)
8. TON (test outcome negative)
9. POP (population)

Class metrics

1. TPR (true positive rate) + confidence interval
2. TNR (true negative rate) + confidence interval
3. FPR (false positive rate) + confidence interval
4. FNR (false negative rate) + confidence interval
5. PPV (positive predictive value)
6. NPV (negative predictive value)
7. FDR (false discovery rate)
8. FOR (false omission rate)
9. Class accuracy
10. Class error rate

Overall metrics

1. Accuracy
2. Error rate

Confidence intervals
--------------------

The library implements bootstrapping to compute confidence intervals for arbitrary
(vectorized) measurements. It allows us to compute confidence intervals for arbitrary
functions

.. code-block:: python

    def metric(scores: Scores) -> np.ndarray:
        # Simple metric calculating the mean of positive scores
        return np.mean(scores.pos)

    scores = Scores(pos=[1, 2, 3], neg=[0, 2]) # Sample scores
    ci = scores.bootstrap_ci(metric=metric, alpha=0.05)

    # For metrics that are part of the Scores class we can pass their names
    ci = scores.bootstrap_ci(metric="eer")
    # Scores.eer() returns both threshold and EER value
    print(f"Threshold 95%-CI: ({ci[0, 0]:.4f}, {ci[0, 1]:.4f})")
    print(f"EER 95%-CI: ({ci[1, 0]:.3%}, {ci[1, 1]:.3%})")

Vectorized operations
---------------------

All operations are, as far as feasible, vectorized and care has been taken to ensure
consistent handling of matrix shapes.

- A (vectorized) confusion matrix has shape (X, N, N), where X can be an arbitrary
  shape, including the empty shape, and N is the number of classes.
- Calculating a metric results in an array of shape (X, Y), where Y is the shape
  defined by the metric. Most metrics are scalar, Y=(), while confidence intervals
  have shape (2,).
- A confusion matrix can be converted to a vector of binary confusion matrices using the
  one-vs-all strategy. This results in a binary confusion matrix of shape (X, N, 2, 2).
- Calculating per-class metrics implicitely uses the one-vs-all strategy, so the result
  has shape (X, N, Y).
- Whenever a result is a scalar, we return it as such. This is, e.g., the case when
  computing scalar metrics of single confusion matrices, i.e., X=Y=().

.. Hidden TOCs

.. toctree::
    :maxdepth: 2
    :caption: Concepts
    :hidden:

    concepts/confidence_intervals

.. toctree::
    :maxdepth: 2
    :caption: Score Analysis
    :hidden:

    score_analysis/cm
    score_analysis/group_scores
    score_analysis/metrics
    score_analysis/roc_curve
    score_analysis/scores
    score_analysis/utils
