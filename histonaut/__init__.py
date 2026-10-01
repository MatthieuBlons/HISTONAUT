"""HISTONAUT: whole-slide image (WSI) processing.

Read pyramidal slides, cut them into patches and encode patches and slides with
pathology foundation models.

- `histonaut.slide`: read, mask and draw WSIs.
- `histonaut.patcher`: cut WSIs into patches and save their coordinates.
- `histonaut.encoder`: encode patches and slides with foundation models.

Exemple Pipeline: 

`OpenWSI` (slide) -> `SlidePatcher` (patcher) -> coordinates h5 ->
`TileEncoder` (encoder) -> features + coordinates h5 -> `aggragate_tiles_features` -> slide
embedding h5.
"""
