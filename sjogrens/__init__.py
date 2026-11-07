"""Predicting Sjögren's syndrome from whole-blood gene expression."""

import os

# Hide TensorFlow's C++ startup logs (they're informational, not errors).
# This has to be set before TensorFlow is imported anywhere.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
