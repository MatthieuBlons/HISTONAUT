"""Encode patches and slides with pathology foundation models.

(`patch_encoder_factory`) turn patches into features saved to h5;
(`slide_encoder_factory`, `aggragate_tiles_features`) turn patch
features into a slide-level embedding.
"""
