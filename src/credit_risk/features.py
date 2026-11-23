"""Row-wise feature transforms (no fitting): flags, ratios, protected and redundant drops.

Anything that learns from data (medians, scaling, encodings) belongs in the sklearn
Pipeline in train.py, so it is re-fitted inside every CV fold.

Item 5 of Stage 4. Not written yet.
"""
