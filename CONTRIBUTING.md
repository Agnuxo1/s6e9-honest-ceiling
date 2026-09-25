# Contributing

Thank you for looking. The most valuable contribution is an **honest attempt to beat the ceiling**, whether it
succeeds or not.

## Proposing a result

1. Open an issue with the *Research result* template.
2. State the hypothesis, the code (fork or gist) and the numbers required by the acceptance test in
   [ROADMAP.md](ROADMAP.md): nested out-of-fold AUC per fold and the paired private-size bootstrap.
3. Negative results are welcome and will be listed with credit; they narrow the space for everyone.

## Code changes

* `pip install -e ".[test]"`, then `pytest -q` must pass.
* Keep every supervised statistic inside the outer-train partition. A pull request that touches
  `features.py`, `train.py` or `blend.py` must keep `test_nested_target_encoding_never_sees_own_label` passing and
  should add a test if it adds a new encoder.
* Do not commit competition data, predictions on the competition test set other than ours, or credentials.
* Style: standard library typing, NumPy-style docstrings where a function is not obvious, no new heavy
  dependencies without discussion.

## Conduct

Be precise and kind. Claims need numbers; numbers need code.
