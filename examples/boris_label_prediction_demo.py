#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0

Example: predict manually scored BORIS behaviours from VAME latent vectors
and/or DeepLabCut pose features.

Prerequisites (see examples/demo.py):
  * vame.egocentric_alignment() or vame.csv_to_numpy()   -> data/<video>/<video>-PE-seq.npy
  * vame.create_trainset()                               -> data/<video>/<video>-PE-seq-clean.npy
  * vame.train_model() + vame.pose_segmentation()        -> latent vectors (only needed for
                                                            features containing 'latent'/'motif')
"""

import vame

config = '/YOUR/WORKING/DIRECTORY/Your-VAME-Project-Apr14-2020/config.yaml'

# Step 1: Export your BORIS observations (Observations -> Export events -> aggregated events,
# tabulated events or binary table; csv/tsv/xlsx) and copy them into
# <project>/videos/boris/<video-name>.csv  -- one file per video, named like the video --
# or export all observations into one file (matched by "Observation id" or media file name).
#
# Converts the events into per-frame labels: data/<video>/<video>-boris-labels.npy
# behaviors: optional subset; its order sets the priority when bouts overlap (last wins).
# fps: frame rate of the video; taken from the config ('boris_fps') or the BORIS file if None.
vame.boris_to_numpy(config, behaviors=None, fps=None, subject=None)

# Step 2: Train a classifier and evaluate it with cross-validation.
# features   : 'latent', 'pose', 'motif' or combinations, e.g. 'latent+pose' (default)
# classifier : 'random_forest' (default), 'logistic', 'mlp', 'gradient_boosting'
#              or any sklearn estimator instance
# cv         : 'video' = leave-one-video-out (default), 'blocked' = contiguous blocks within videos
# Saves the classifier, cv_report.json, the confusion matrix and feature importances to
# results/label_prediction/<model_name>/<features>_<classifier>/
# and (predict=True) the predicted labels for all videos of the project.
report = vame.train_label_predictor(config, features='latent+pose', classifier='random_forest')
print('Cross-validated balanced accuracy: %.3f' % report['balanced_accuracy'])

# Compare feature sources: how much do the VAME latents add on top of the raw pose?
for features in ['pose', 'latent', 'latent+pose']:
    r = vame.train_label_predictor(config, features=features, predict=False)
    print('%-12s balanced accuracy %.3f  F1 macro %.3f' % (features, r['balanced_accuracy'], r['f1_macro']))

# Step 3: Predict BORIS labels for (new, unscored) videos with a trained classifier.
# Output per video in results/<video>/<model_name>/label_prediction/:
#   boris_prediction_<tag>_<video>.npy / .csv   frame-wise labels and probabilities
#   boris_prediction_bouts_<tag>_<video>.csv    bouts (Behavior, Start (s), Stop (s), ...)
vame.predict_labels(config, features='latent+pose', classifier='random_forest')
