#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0

Supervised prediction of manually scored BORIS behaviours from VAME latent
vectors and/or DeepLabCut pose features.

Workflow
--------
1. vame.boris_to_numpy(config)             -> per-frame BORIS labels
2. vame.train_label_predictor(config, ...) -> cross-validated classifier
3. vame.predict_labels(config, ...)        -> labels for all (also unscored) videos

Feature sources (combine with '+', e.g. 'latent+pose'):
  'latent' : VAME latent vectors (results/<video>/<model>/<param>-<k>/latent_vector_<video>.npy)
  'pose'   : cleaned pose data (data/<video>/<video>-PE-seq-clean.npy) with
             velocity and windowed mean/std
  'motif'  : one-hot encoded VAME motif labels
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
from scipy.ndimage import uniform_filter1d

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)

from vame.util.auxiliary import read_config


FEATURE_SOURCES = ('latent', 'pose', 'motif')
CLASSIFIERS = ('random_forest', 'logistic', 'mlp', 'gradient_boosting')


# -----------------------------------------------------------------------------
# helpers: paths and config
# -----------------------------------------------------------------------------
def _get(cfg, key, default):
    val = cfg.get(key, None)
    if val is None or val == 'None':
        return default
    return val


def _parse_features(features):
    if isinstance(features, (list, tuple)):
        feats = [str(f).strip().lower() for f in features]
    else:
        feats = [f.strip().lower() for f in str(features).replace(',', '+').split('+') if f.strip()]
    if features == 'both' or feats == ['both']:
        feats = ['latent', 'pose']
    if features == 'all' or feats == ['all']:
        feats = list(FEATURE_SOURCES)
    for f in feats:
        if f not in FEATURE_SOURCES:
            raise ValueError("Unknown feature source '%s'. Choose from %s" % (f, FEATURE_SOURCES))
    if not feats:
        raise ValueError("No feature source given.")
    # keep canonical order
    return [f for f in FEATURE_SOURCES if f in feats]


def _run_tag(features, classifier):
    clf_name = classifier if isinstance(classifier, str) else type(classifier).__name__
    return '+'.join(features) + '_' + clf_name


def _param_dir(cfg, file):
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    parameterization = _get(cfg, 'parameterization', 'kmeans')
    return os.path.join(cfg['project_path'], 'results', file, model_name,
                        parameterization + '-' + str(n_cluster), '')


def _model_dir(cfg, features, classifier):
    return os.path.join(cfg['project_path'], 'results', 'label_prediction',
                        cfg['model_name'], _run_tag(features, classifier), '')


def _prediction_dir(cfg, file):
    return os.path.join(cfg['project_path'], 'results', file, cfg['model_name'], 'label_prediction', '')


def _latent_offset(cfg):
    """
    Frame index that latent vector 0 is assigned to. VAME embeds the window
    data[:, i:i+time_window] into latent i, and the rest of VAME (motif videos)
    assigns it to the window centre, so we do the same by default.
    """
    mode = _get(cfg, 'label_latent_alignment', 'center')
    tw = int(cfg['time_window'])
    if isinstance(mode, (int, np.integer)) and not isinstance(mode, bool):
        return int(mode)
    mode = str(mode).lower()
    if mode == 'start':
        return 0
    if mode == 'end':
        return tw
    if mode == 'center' or mode == 'centre':
        return tw // 2
    return int(mode)


# -----------------------------------------------------------------------------
# helpers: feature construction
# -----------------------------------------------------------------------------
def pose_features(pose, window=15):
    """
    Builds per-frame features from a (n_features, n_frames) pose array:
    position, velocity (first difference) and, if window > 1, rolling mean and
    rolling std over a centred window of ``window`` frames.
    Returns (n_frames, n_out) array.
    """
    X = np.asarray(pose, dtype=np.float64).T  # frames x features
    X = np.nan_to_num(X)
    vel = np.diff(X, axis=0, prepend=X[:1])
    blocks = [X, vel]
    if window is not None and int(window) > 1:
        w = int(window)
        mean = uniform_filter1d(X, size=w, axis=0, mode='nearest')
        sq = uniform_filter1d(X ** 2, size=w, axis=0, mode='nearest')
        std = np.sqrt(np.clip(sq - mean ** 2, 0, None))
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        speed_mean = uniform_filter1d(speed, size=w, axis=0, mode='nearest')
        blocks += [mean, std, speed_mean]
    return np.concatenate(blocks, axis=1)


def _pose_feature_names(n_feat, window):
    names = ['pos_%d' % i for i in range(n_feat)] + ['vel_%d' % i for i in range(n_feat)]
    if window is not None and int(window) > 1:
        names += ['mean_%d' % i for i in range(n_feat)] + ['std_%d' % i for i in range(n_feat)] + ['speed_mean']
    return names


def load_video_features(cfg, file, features, pose_window=15):
    """
    Loads and frame-aligns all requested feature sources of one video.

    Returns
    -------
    X : (n_valid, n_features) array
    frames : (n_valid,) frame indices of the pose data each row belongs to
    n_frames : total number of frames of the video
    feature_names : list of str
    """
    project_path = cfg['project_path']
    offset = _latent_offset(cfg)
    blocks, names = [], []
    n_frames = None
    lat_len = None

    pose_path = os.path.join(project_path, 'data', file, file + '-PE-seq-clean.npy')
    if not os.path.exists(pose_path):
        pose_path = os.path.join(project_path, 'data', file, file + '-PE-seq.npy')
    if os.path.exists(pose_path):
        n_frames = np.load(pose_path, mmap_mode='r').shape[1]

    if 'latent' in features or 'motif' in features:
        pdir = _param_dir(cfg, file)
        if 'latent' in features:
            lpath = os.path.join(pdir, 'latent_vector_' + file + '.npy')
            if not os.path.exists(lpath):
                raise FileNotFoundError("Latent vectors %s not found. Run vame.pose_segmentation() first." % lpath)
            latent = np.load(lpath)
            lat_len = latent.shape[0]
            blocks.append(('latent', latent))
            names += ['latent_%d' % i for i in range(latent.shape[1])]
        if 'motif' in features:
            mpath = os.path.join(pdir, str(cfg['n_cluster']) + '_km_label_' + file + '.npy')
            if not os.path.exists(mpath):
                raise FileNotFoundError("Motif labels %s not found. Run vame.pose_segmentation() first." % mpath)
            motif = np.load(mpath).astype(int)
            lat_len = motif.shape[0] if lat_len is None else min(lat_len, motif.shape[0])
            onehot = np.zeros((motif.shape[0], int(cfg['n_cluster'])), dtype=np.float64)
            valid = (motif >= 0) & (motif < onehot.shape[1])
            onehot[np.arange(motif.shape[0])[valid], motif[valid]] = 1.0
            blocks.append(('latent', onehot))
            names += ['motif_%d' % i for i in range(onehot.shape[1])]
        if n_frames is None:
            n_frames = lat_len + int(cfg['time_window'])

    if 'pose' in features:
        if not os.path.exists(pose_path):
            raise FileNotFoundError("Pose data for %s not found in %s" % (file, os.path.join(project_path, 'data', file)))
        pose = np.load(pose_path)
        feats = pose_features(pose, window=pose_window)
        blocks.append(('pose', feats))
        names += _pose_feature_names(pose.shape[0], pose_window)

    # frames that have every requested feature
    if lat_len is not None:
        lat_len = min(lat_len, max(n_frames - offset, 0))
        frames = np.arange(offset, offset + lat_len)
    else:
        frames = np.arange(n_frames)

    cols = []
    for kind, arr in blocks:
        if kind == 'latent':
            cols.append(arr[:len(frames)])
        else:
            cols.append(arr[frames])
    X = np.concatenate(cols, axis=1).astype(np.float32)
    return X, frames, n_frames, names


def load_video_labels(cfg, file):
    path = os.path.join(cfg['project_path'], 'data', file, file + '-boris-labels.npy')
    if not os.path.exists(path):
        return None, None
    labels = np.load(path).astype(int)
    meta_path = os.path.join(cfg['project_path'], 'data', file, file + '-boris-labels.json')
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    return labels, meta


# -----------------------------------------------------------------------------
# helpers: models, smoothing, evaluation
# -----------------------------------------------------------------------------
def build_classifier(classifier='random_forest', random_state=42, n_jobs=-1):
    """
    Returns an sklearn Pipeline (StandardScaler + estimator). ``classifier``
    may also be a ready-made sklearn estimator instance.
    """
    if isinstance(classifier, BaseEstimator):
        est = clone(classifier)
    else:
        name = str(classifier).lower()
        if name in ('random_forest', 'rf'):
            from sklearn.ensemble import RandomForestClassifier
            est = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                         min_samples_leaf=2, n_jobs=n_jobs, random_state=random_state)
        elif name in ('logistic', 'logistic_regression', 'lr'):
            from sklearn.linear_model import LogisticRegression
            est = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
        elif name == 'mlp':
            from sklearn.neural_network import MLPClassifier
            est = MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True,
                                max_iter=300, random_state=random_state)
        elif name in ('gradient_boosting', 'hgb', 'gb'):
            from sklearn.ensemble import HistGradientBoostingClassifier
            est = HistGradientBoostingClassifier(class_weight='balanced', random_state=random_state)
        else:
            raise ValueError("Unknown classifier '%s'. Choose from %s or pass an sklearn estimator."
                             % (classifier, CLASSIFIERS))
    return Pipeline([('scaler', StandardScaler()), ('clf', est)])


def smooth_probabilities(proba, window):
    """Moving average of class probabilities over a centred window of frames."""
    if window is None or int(window) <= 1 or proba.shape[0] < 2:
        return proba
    return uniform_filter1d(proba, size=int(window), axis=0, mode='nearest')


def _predict_proba(model, X, n_classes):
    """predict_proba over all ``n_classes`` (columns for classes unseen in training are 0)."""
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)
        classes = model.classes_ if hasattr(model, 'classes_') else model[-1].classes_
    else:
        pred = model.predict(X)
        classes = np.unique(pred)
        p = (pred[:, None] == classes[None, :]).astype(float)
    out = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    for j, c in enumerate(classes):
        out[:, int(c)] = p[:, j]
    return out


def _metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    present = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    rep = classification_report(y_true, y_pred, labels=present,
                                target_names=[class_names[i] for i in present],
                                output_dict=True, zero_division=0)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, labels=present, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, labels=present, average='weighted', zero_division=0)),
        'per_class': {k: v for k, v in rep.items() if k in class_names},
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'n_samples': int(len(y_true)),
    }


def _cv_splits(groups, y, cv='video', n_folds=5):
    """
    Yields (train_idx, test_idx) pairs.
    'video'   : leave-one-video-out (GroupKFold when more videos than folds).
                Falls back to 'blocked' if only one labelled video exists.
    'blocked' : each video is cut into n_folds contiguous blocks, fold k holds
                block k of every video (avoids leakage between neighbouring frames).
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    n = len(groups)
    if cv == 'video' and len(uniq) > 1:
        from sklearn.model_selection import GroupKFold
        k = min(int(n_folds), len(uniq))
        for tr, te in GroupKFold(n_splits=k).split(np.zeros(n), y, groups):
            yield tr, te
        return
    if cv == 'video':
        print("Only one labelled video: using blocked cross-validation instead of leave-one-video-out.")
    k = int(n_folds)
    fold_id = np.zeros(n, dtype=int)
    for g in uniq:
        idx = np.where(groups == g)[0]
        fold_id[idx] = np.minimum((np.arange(len(idx)) * k) // max(len(idx), 1), k - 1)
    for f in range(k):
        te = np.where(fold_id == f)[0]
        tr = np.where(fold_id != f)[0]
        if len(te) and len(tr):
            yield tr, te


def _plot_confusion(cm, class_names, path, title):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as err:
        print("Could not plot confusion matrix: %s" % err)
        return
    cm = np.asarray(cm, dtype=float)
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    fig, ax = plt.subplots(figsize=(1.2 + 0.6 * len(class_names), 1.0 + 0.6 * len(class_names)))
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('BORIS label')
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, '%d' % cm[i, j], ha='center', va='center',
                    color='white' if norm[i, j] > 0.5 else 'black', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bouts_from_labels(labels, class_names, fps=None):
    """Turns a per-frame label vector into a table of bouts."""
    labels = np.asarray(labels)
    rows = []
    if len(labels) == 0:
        return pd.DataFrame(rows)
    change = np.where(np.diff(labels) != 0)[0] + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [len(labels)]])
    for s, e in zip(starts, stops):
        row = {'Behavior': class_names[int(labels[s])], 'Start (frame)': int(s), 'Stop (frame)': int(e),
               'Duration (frames)': int(e - s)}
        if fps:
            row.update({'Start (s)': s / fps, 'Stop (s)': e / fps, 'Duration (s)': (e - s) / fps})
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------
def train_label_predictor(config, features=None, classifier=None, files=None, pose_window=None,
                          smoothing_window=None, cv=None, n_folds=None, ignore_background=None,
                          predict=True, random_state=42, n_jobs=-1):
    """
    Trains a classifier that predicts BORIS behaviours from VAME latent vectors
    and/or pose features, evaluates it with cross-validation and saves it.

    Parameters
    ----------
    config : path to config.yaml
    features : 'latent', 'pose', 'motif' or combinations like 'latent+pose'
        (default: config ``label_features`` or 'latent+pose')
    classifier : 'random_forest', 'logistic', 'mlp', 'gradient_boosting' or an
        sklearn estimator instance (default: config ``label_classifier``)
    files : list of labelled videos to train on (default: all videos with BORIS labels)
    pose_window : window (frames) for rolling pose statistics (config ``label_pose_window``)
    smoothing_window : window (frames) for temporal smoothing of the predicted
        probabilities (config ``label_smoothing_window``); <=1 disables it
    cv : 'video' (leave-one-video-out) or 'blocked' (config ``label_cv``)
    n_folds : number of folds (config ``label_n_folds``)
    ignore_background : drop frames without a BORIS behaviour from training and
        evaluation (config ``label_ignore_background``)
    predict : if True, run vame.predict_labels() for all videos afterwards

    Returns
    -------
    report : dict with cross-validation metrics (also saved as cv_report.json)
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)

    features = _parse_features(_get(cfg, 'label_features', 'latent+pose') if features is None else features)
    classifier = _get(cfg, 'label_classifier', 'random_forest') if classifier is None else classifier
    pose_window = int(_get(cfg, 'label_pose_window', 15)) if pose_window is None else int(pose_window)
    smoothing_window = int(_get(cfg, 'label_smoothing_window', 5)) if smoothing_window is None else int(smoothing_window)
    cv = str(_get(cfg, 'label_cv', 'video')) if cv is None else str(cv)
    n_folds = int(_get(cfg, 'label_n_folds', 5)) if n_folds is None else int(n_folds)
    if ignore_background is None:
        ignore_background = bool(_get(cfg, 'label_ignore_background', False))

    if files is None:
        files = [f for f in cfg['video_sets']
                 if os.path.exists(os.path.join(cfg['project_path'], 'data', f, f + '-boris-labels.npy'))]
    if not files:
        raise FileNotFoundError("No video with BORIS labels found. Run vame.boris_to_numpy() first.")

    print("Training BORIS label predictor for model %s" % cfg['model_name'])
    print("  features   : %s" % '+'.join(features))
    print("  classifier : %s" % (classifier if isinstance(classifier, str) else type(classifier).__name__))
    print("  videos     : %s" % files)

    X_all, y_all, g_all = [], [], []
    class_names = None
    feature_names = None
    for file in files:
        labels, meta = load_video_labels(cfg, file)
        if labels is None:
            raise FileNotFoundError("No BORIS labels for %s. Run vame.boris_to_numpy() first." % file)
        names = meta['class_names'] if meta else None
        if class_names is None:
            class_names = names
        elif names is not None and names != class_names:
            raise ValueError("Class names of %s (%s) differ from %s. Fix 'boris_behaviors' in the config "
                             "and re-run vame.boris_to_numpy() so all videos share the same classes."
                             % (file, names, class_names))
        X, frames, n_frames, feature_names = load_video_features(cfg, file, features, pose_window)
        if len(labels) != n_frames:
            print("Warning: %s has %d BORIS frames but %d pose frames, truncating." % (file, len(labels), n_frames))
        valid = frames < len(labels)
        X, frames = X[valid], frames[valid]
        y = labels[frames]
        if ignore_background:
            keep = y != 0
            X, y = X[keep], y[keep]
        X_all.append(X)
        y_all.append(y)
        g_all.append(np.full(len(y), file))
        print("  %s: %d frames, %d features" % (file, len(y), X.shape[1]))

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    groups = np.concatenate(g_all)
    if class_names is None:
        class_names = ['none'] + ['behavior_%d' % i for i in range(1, int(y.max()) + 1)]
    n_classes = len(class_names)
    if len(np.unique(y)) < 2:
        raise ValueError("Only one class present in the training labels (%s); nothing to learn."
                         % [class_names[i] for i in np.unique(y)])

    # ---- cross-validation -------------------------------------------------
    print("\nCross-validation (%s, %d folds)..." % (cv, n_folds))
    y_pred_raw = np.full(len(y), -1, dtype=int)
    y_pred_smooth = np.full(len(y), -1, dtype=int)
    fold_reports = []
    for k, (tr, te) in enumerate(_cv_splits(groups, y, cv=cv, n_folds=n_folds)):
        model = build_classifier(classifier, random_state=random_state, n_jobs=n_jobs)
        model.fit(X[tr], y[tr])
        proba = _predict_proba(model, X[te], n_classes)
        y_pred_raw[te] = proba.argmax(axis=1)
        # smooth within each video separately (test indices are contiguous per video)
        sm = np.zeros_like(proba)
        for g in np.unique(groups[te]):
            m = groups[te] == g
            sm[m] = smooth_probabilities(proba[m], smoothing_window)
        y_pred_smooth[te] = sm.argmax(axis=1)
        fm = _metrics(y[te], y_pred_smooth[te], class_names)
        fold_reports.append({'fold': k, 'test_videos': sorted(set(groups[te].tolist())),
                             'accuracy': fm['accuracy'], 'balanced_accuracy': fm['balanced_accuracy'],
                             'f1_macro': fm['f1_macro'], 'n_test': int(len(te))})
        print("  fold %d (%s): acc %.3f | bal. acc %.3f | F1 macro %.3f"
              % (k, ','.join(fold_reports[-1]['test_videos']), fm['accuracy'], fm['balanced_accuracy'], fm['f1_macro']))

    scored = y_pred_smooth >= 0
    report = _metrics(y[scored], y_pred_smooth[scored], class_names)
    report_raw = _metrics(y[scored], y_pred_raw[scored], class_names)
    report.update({
        'features': features, 'feature_names': feature_names,
        'classifier': classifier if isinstance(classifier, str) else repr(classifier),
        'class_names': class_names, 'videos': list(files), 'cv': cv, 'n_folds': n_folds,
        'pose_window': pose_window, 'smoothing_window': smoothing_window,
        'ignore_background': ignore_background, 'folds': fold_reports,
        'unsmoothed': {k: report_raw[k] for k in ('accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted')},
        'class_frame_counts': {class_names[i]: int((y == i).sum()) for i in range(n_classes)},
    })
    print("\nCross-validated performance (smoothed predictions):")
    print("  accuracy          : %.3f" % report['accuracy'])
    print("  balanced accuracy : %.3f" % report['balanced_accuracy'])
    print("  F1 macro          : %.3f" % report['f1_macro'])
    for name in class_names:
        pc = report['per_class'].get(name)
        if pc:
            print("  %-20s precision %.3f recall %.3f f1 %.3f (n=%d)"
                  % (name, pc['precision'], pc['recall'], pc['f1-score'], pc['support']))

    # ---- final model on all data ---------------------------------------------
    print("\nFitting final model on all labelled frames...")
    model = build_classifier(classifier, random_state=random_state, n_jobs=n_jobs)
    model.fit(X, y)

    out_dir = _model_dir(cfg, features, classifier)
    os.makedirs(out_dir, exist_ok=True)
    bundle = {'model': model, 'class_names': class_names, 'features': features,
              'feature_names': feature_names, 'pose_window': pose_window,
              'smoothing_window': smoothing_window, 'latent_offset': _latent_offset(cfg),
              'ignore_background': ignore_background, 'model_name': cfg['model_name'],
              'n_features': int(X.shape[1])}
    joblib.dump(bundle, os.path.join(out_dir, 'classifier.pkl'))
    with open(os.path.join(out_dir, 'cv_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(report['confusion_matrix'], index=class_names, columns=class_names).to_csv(
        os.path.join(out_dir, 'cv_confusion_matrix.csv'))
    _plot_confusion(report['confusion_matrix'], class_names, os.path.join(out_dir, 'cv_confusion_matrix.png'),
                    'BORIS label prediction (%s)' % _run_tag(features, classifier))

    est = model.named_steps['clf']
    if hasattr(est, 'feature_importances_') and feature_names is not None:
        imp = pd.DataFrame({'feature': feature_names, 'importance': est.feature_importances_})
        imp.sort_values('importance', ascending=False).to_csv(os.path.join(out_dir, 'feature_importance.csv'), index=False)
        top = imp.sort_values('importance', ascending=False).head(10)
        print("Top features: %s" % ', '.join('%s (%.3f)' % (r.feature, r.importance) for r in top.itertuples()))

    print("Classifier and cross-validation report saved to %s" % out_dir)

    if predict:
        predict_labels(config, features=features, classifier=classifier)
    return report


def predict_labels(config, features=None, classifier=None, files=None, fps=None):
    """
    Predicts BORIS behaviours for videos of the project with a classifier
    trained by vame.train_label_predictor().

    Parameters
    ----------
    config : path to config.yaml
    features, classifier : identify the trained classifier (defaults as in
        train_label_predictor)
    files : videos to predict (default: all videos in ``video_sets`` that have
        the required features)
    fps : frame rate used for the time columns of the csv output (default:
        taken from the BORIS metadata of the video or config ``boris_fps``)

    Writes (per video, in results/<video>/<model>/label_prediction/)
    ------
    boris_prediction_<tag>_<video>.npy        (n_frames,) int label per frame
    boris_prediction_proba_<tag>_<video>.npy  (n_frames, n_classes) probabilities
    boris_prediction_<tag>_<video>.csv        frame-wise table incl. probabilities
    boris_prediction_bouts_<tag>_<video>.csv  bouts in a BORIS-like table
    Frames before/after the range covered by latent vectors get the nearest prediction.
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    features = _parse_features(_get(cfg, 'label_features', 'latent+pose') if features is None else features)
    classifier = _get(cfg, 'label_classifier', 'random_forest') if classifier is None else classifier
    tag = _run_tag(features, classifier)

    model_path = os.path.join(_model_dir(cfg, features, classifier), 'classifier.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained classifier at %s. Run vame.train_label_predictor() first." % model_path)
    bundle = joblib.load(model_path)
    model = bundle['model']
    class_names = bundle['class_names']
    n_classes = len(class_names)

    if files is None:
        files = list(cfg['video_sets'])

    results = {}
    for file in files:
        try:
            X, frames, n_frames, _ = load_video_features(cfg, file, features, bundle['pose_window'])
        except FileNotFoundError as err:
            print("Skipping %s: %s" % (file, err))
            continue
        if X.shape[1] != bundle['n_features']:
            raise ValueError("%s has %d features but the classifier was trained with %d."
                             % (file, X.shape[1], bundle['n_features']))
        proba = smooth_probabilities(_predict_proba(model, X, n_classes), bundle['smoothing_window'])
        full = np.zeros((n_frames, n_classes), dtype=np.float32)
        full[frames] = proba
        if len(frames):
            full[:frames[0]] = proba[0]
            full[frames[-1] + 1:] = proba[-1]
        labels = full.argmax(axis=1)

        out_dir = _prediction_dir(cfg, file)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'boris_prediction_%s_%s.npy' % (tag, file)), labels)
        np.save(os.path.join(out_dir, 'boris_prediction_proba_%s_%s.npy' % (tag, file)), full)

        _, meta = load_video_labels(cfg, file)
        this_fps = fps or (meta.get('fps') if meta else None) or _get(cfg, 'boris_fps', None)
        table = pd.DataFrame({'frame': np.arange(n_frames), 'label': labels,
                              'behavior': [class_names[i] for i in labels]})
        if this_fps:
            table.insert(1, 'time_s', table['frame'] / float(this_fps))
        for i, name in enumerate(class_names):
            table['p_' + name] = full[:, i]
        table.to_csv(os.path.join(out_dir, 'boris_prediction_%s_%s.csv' % (tag, file)), index=False)
        _bouts_from_labels(labels, class_names, this_fps).to_csv(
            os.path.join(out_dir, 'boris_prediction_bouts_%s_%s.csv' % (tag, file)), index=False)

        usage = {class_names[i]: int((labels == i).sum()) for i in range(n_classes)}
        print("Predicted labels for %s: %s" % (file, usage))
        results[file] = labels

    if results:
        print("Predictions saved to results/<video>/%s/label_prediction/" % cfg['model_name'])
    return results
