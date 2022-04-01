# onfido-score-analysis

Package to analyse ML model results. Contains an efficient implementation of common
metrics computations such as TPR, FPR, EER and methods for threshold setting.

## Usage

### Terminology

At Onfido, we like to work with metrics based on acceptance and rejection, such as
FAR (false acceptance rate) and FRR (false rejection rate), while the standard ML
terminology talks about positive and negative classes and FPR (false positive rate) and
FNR (false negative rate).

This library adopts the standard ML terminology. The translation is simple: just replace
"accept" with "positive" and replace "reject" with "negative" and you have a dictionary
between the two worlds.

The library is also agnostic to which direction scores are pointing. It works with 
scores that indicate membership of the positive (accept) class as well as with scores
that indicate membership of the negative (reject) class. The score interpretation is
set using the `score_class` parameter when constructing a `Scores` object.

The key is to decouple the process of computing scores from the process of interpreting
them. When we compute scores, e.g., using an ML model, some will point towards genuines,
some towards spoofs/fraud. Sometimes we use score-mappers that reverse the score 
orientation. We cannot change scores. But, when we move towards interpreting them, we
should always use a fixed terminology: positive class means accept/genuine; negative 
class means reject/spoof/fraud. And at the point, when we go from generating scores to
interpreting them, we set, via the `score_class` parameter, how scores are to be 
interpreted.

### Scores

We assume that we work with a binary classification problem. First, we create a `Scores`
object with the experiment results. We can do this in two ways.

```python
from score_analysis import Scores

# If we have the scores for the positive and negative classes separately
scores = Scores(pos=[1, 2, 3], neg=[0.5, 1.5])
# If we have an intermingled set of scores and labels
scores = Scores.from_labels(
    scores=[1, 2, 3, 0.5, 1.5], 
    labels=[1, 1, 1, 0, 0],
    # We specify the label of the positive class. All other labels are assigned to
    # the negative class.
    pos_label=1,  
)
```

There are two parameters, that will determine calculations of metrics:

- Do scores indicate membership of the positive or the negative class (`score_class`)
- If a score is exactly equal to the threshold, will it be assigned to the positive
  or negative class (`equal_class`)

The meaning of the parameters is summarized in the table

| score_class | equal_class | Decision logic for positive class |
|:-----------:|:-----------:|:---------------------------------:|
|     pos     |     pos     |        score >= threshold         |
|     pos     |     neg     |         score > threshold         |
|     neg     |     pos     |        score <= threshold         |
|     neg     |     neg     |         score < threshold         |

We can apply a threshold to a `Scores` object to obtain a confusion matrix and then
compute metrics associated to the confusion matrix.

```python
cm = scores.cm(threshold=2.5)
print(cm.fpr())  # Print False Positive Rate
```

We can work with multiple thresholds at once, which leads to vectorized confusion
matrices. In fact, the `threshold` parameter accepts arbitrary-shaped arrays and all
confusion matrix operations preserve the shapes.

```python
threshold = np.linspace(0, 3, num=50)
cm = scores.cm(threshold=threshold)
fpr = cm.fpr()  # Contains FPRs at all defined thresholds
assert fpr.shape == threshold.shape
```

We can also determine thresholds at specific operating points. These operations are also
fully vectorized.

```python
# Calculate threshold at 30% False Positive Rate
threshold = scores.threshold_at_fpr(fpr=0.3)

# Calculate thresholds at logarithmically spaced FPR intervals from 0.1% to 100%
fpr = np.logspace(-3, 0, num=50)
threshold = scores.threshold_at_fpr(fpr)
```

Note that determining thresholds a fixed operating points requires interpolation, since
with a finite dataset we can measure only finitely many values for FPR, etc. If we want
to determine a threshold at any other value for the target metric, we use linear
interpolation.

### Confusion matrices

Most metrics that we use are defined via confusion matrices. We can create a confusion
matrix either from vectors with labels and predictions or directly from a matrix.

```python
>>> labels      = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
>>> predictions = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
>>> cm = ConfusionMatrix(labels=labels, predictions=predictions)
>>> cm.classes
[0, 1, 2]
>>> cm.matrix
array([[3, 0, 0],
       [0, 1, 2],
       [2, 1, 3]])
```

A `BinaryConfusionMatrix` is a special case of a `ConfusionMatrix`, with specially
designated positive and negative classes. The convention is that the classes are
ordered `classes = [pos, neg]`.

For binary confusion matrices all metrics such as TPR are scalar. Since we have defined
which is the positive class, there is no need to use the one-vs-all strategy.

A `BinaryConfusionMatrix` is different from a `ConfusionMatrix` with two classes, since
the latter does not have designated positive and negative classes

```python
>>> cm = BinaryConfusionMatrix(matrix=[[1, 4], [2, 3]])
>>> cm.tpr()
0.2
>>> cm = ConfusionMatrix(matrix=[[1, 4], [2, 3]])
>>> cm.tpr()  # True positive rate for each class
array([0.2, 0.6])
```

### Available metrics

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

1. TPR (true positive rate)
2. TNR (true negative rate)
3. FPR (false positive rate)
4. FNR (false negative rate)
5. PPV (positive predictive value)
6. NPV (negative predictive value)
7. FDR (false discovery rate)
8. FOR (false omission rate)
9. Class accuracy
10. Class error rate

Overall metrics

1. Accuracy
2. Error rate

## Contributing

Before submitting an MR, please run

```shell
make style
```

This will run `black`, `isort` and `flake8` on the code.

Unit tests can be executed via

```shell
make test
```

## Formatting tips

 * `# fmt: skip` for disabling formatting on a single line.
 * `# fmt: off` / `# fmt: on` for disabling formatting on a block of code.
 * `# noqa: F401` to disable flake8 warning of unused import

## Future plans

The following features are planned

- [ ] Aliases for metrics
