"""Check that basic features work.

Catch cases where e.g. files are missing so the import doesn't work. It is
recommended to check that e.g. assets are included."""

from score_analysis import Scores

scores = Scores.from_labels(labels=[0, 1, 1], scores=[3, 2, 1])
print(f"Scores object created with {scores.nb_all_pos} positive scores.")
