"""Classification experiments on the 60-s windows (size: SMALL / MID / LARGE; powertrain: BEV vs ICE).

Inputs are the official channels only (vtt.config.OFFICIAL_INPUTS), read from the modelling cache written by
scripts/s13_export_tensors.py; labels, splits and protocols come from the published index tables, so every number here
is reproducible from the released dataset plus this folder. Nothing in this package reads the raw library.
"""
