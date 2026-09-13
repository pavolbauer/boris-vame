#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0

Import of manually scored BORIS (Behavioral Observation Research Interactive
Software, http://www.boris.unito.it) ethograms into per-frame label arrays that
line up with the pose-estimation (DLC) data of a VAME project.

Supported BORIS export formats
------------------------------
* "Aggregated events" export (csv / tsv / xlsx): one row per behavioural bout
  with ``Behavior``, ``Start (s)``, ``Stop (s)`` columns.
* "Tabulated events" export (csv / tsv): one row per event with ``Time``,
  ``Behavior`` and ``Status`` (START / STOP / POINT) columns.
* "Binary table" export (csv / tsv): one row per time bin with one 0/1 column
  per behaviour.

The resulting label vector is saved as
``<project>/data/<video>/<video>-boris-labels.npy`` (int, one entry per video
frame, ``0`` = background / unlabeled) together with a JSON file describing the
class names. A multi-hot matrix (frames x behaviours) is stored as well so that
overlapping behaviours are not lost.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

from vame.util.auxiliary import read_config


# Column names used by the different BORIS export dialects.
_BEHAVIOR_COLS = ['Behavior', 'behavior', 'Behaviour']
_START_COLS = ['Start (s)', 'Start(s)', 'start (s)', 'Start']
_STOP_COLS = ['Stop (s)', 'Stop(s)', 'stop (s)', 'Stop']
_TIME_COLS = ['Time', 'time', 'Time (s)']
_STATUS_COLS = ['Status', 'status']
_FPS_COLS = ['FPS (frame/s)', 'FPS', 'fps']
_SUBJECT_COLS = ['Subject', 'subject']
_MEDIA_COLS = ['Media file name', 'Media file', 'Media file path', 'Media file path(s)']
_OBS_COLS = ['Observation id', 'Observation ID', 'observation id']
_FRAME_START_COLS = ['Image index start', 'Frame start', 'Start (frame)']
_FRAME_STOP_COLS = ['Image index stop', 'Frame stop', 'Stop (frame)']


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_boris_file(path):
    """
    Reads a BORIS export (csv, tsv or xlsx) into a pandas DataFrame.
    BORIS exports may contain a preamble before the actual header line; the
    header is located by searching for the ``Behavior`` column.
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(path)

    sep = '\t' if ext == '.tsv' else ','
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("BORIS file %s is empty" % path)

    # auto-detect the separator from the first non-empty line
    first = next((l for l in lines if l.strip()), '')
    if first.count('\t') > first.count(','):
        sep = '\t'
    elif first.count(',') > first.count('\t'):
        sep = ','

    header_row = 0
    for i, line in enumerate(lines):
        cells = [c.strip().strip('"') for c in line.rstrip('\n').split(sep)]
        if any(c in cells for c in _BEHAVIOR_COLS) or any(c in cells for c in _TIME_COLS):
            header_row = i
            break

    df = pd.read_csv(path, sep=sep, skiprows=header_row, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_boris_format(df):
    """
    Returns ``'aggregated'``, ``'tabulated'`` or ``'binary'`` depending on the
    columns present in the DataFrame.
    """
    if _find_col(df, _BEHAVIOR_COLS) is not None and _find_col(df, _START_COLS) is not None \
            and _find_col(df, _STOP_COLS) is not None:
        return 'aggregated'
    if _find_col(df, _BEHAVIOR_COLS) is not None and _find_col(df, _TIME_COLS) is not None \
            and _find_col(df, _STATUS_COLS) is not None:
        return 'tabulated'
    if _find_col(df, _TIME_COLS) is not None or 'time' in [c.lower() for c in df.columns]:
        return 'binary'
    raise ValueError("Could not recognise the BORIS export format. Supported are the "
                     "'aggregated events', 'tabulated events' and 'binary table' exports. "
                     "Columns found: %s" % list(df.columns))


def boris_to_bouts(df, subject=None):
    """
    Converts a BORIS DataFrame (any supported dialect) into a list of bouts
    ``(behavior, start_s, stop_s)``. Point events become bouts of zero length.
    Returns the bouts and the fps stored in the file (or None).
    """
    fmt = detect_boris_format(df)
    fps = None
    fps_col = _find_col(df, _FPS_COLS)
    if fps_col is not None:
        try:
            fps_val = pd.to_numeric(df[fps_col], errors='coerce').dropna()
            if len(fps_val):
                fps = float(fps_val.iloc[0])
        except Exception:
            fps = None

    subj_col = _find_col(df, _SUBJECT_COLS)
    if subject is not None and subj_col is not None:
        df = df[df[subj_col].astype(str) == str(subject)]

    bouts = []
    if fmt == 'aggregated':
        b_col = _find_col(df, _BEHAVIOR_COLS)
        s_col = _find_col(df, _START_COLS)
        e_col = _find_col(df, _STOP_COLS)
        for _, row in df.iterrows():
            beh = str(row[b_col]).strip()
            if beh in ('', 'nan'):
                continue
            start = float(row[s_col])
            stop = float(row[e_col]) if not pd.isna(row[e_col]) else start
            bouts.append((beh, start, stop))

    elif fmt == 'tabulated':
        b_col = _find_col(df, _BEHAVIOR_COLS)
        t_col = _find_col(df, _TIME_COLS)
        st_col = _find_col(df, _STATUS_COLS)
        open_events = {}
        for _, row in df.iterrows():
            beh = str(row[b_col]).strip()
            if beh in ('', 'nan'):
                continue
            t = float(row[t_col])
            status = str(row[st_col]).strip().upper()
            if status == 'START':
                open_events.setdefault(beh, []).append(t)
            elif status == 'STOP':
                if open_events.get(beh):
                    start = open_events[beh].pop(0)
                    bouts.append((beh, start, t))
                else:
                    print("Warning: STOP event for '%s' at %.3f s without START, ignored" % (beh, t))
            elif status == 'POINT':
                bouts.append((beh, t, t))
        for beh, starts in open_events.items():
            for start in starts:
                print("Warning: START event for '%s' at %.3f s without STOP, extended to end" % (beh, start))
                bouts.append((beh, start, np.inf))

    elif fmt == 'binary':
        t_col = _find_col(df, _TIME_COLS)
        if t_col is None:
            t_col = [c for c in df.columns if c.lower() == 'time'][0]
        times = pd.to_numeric(df[t_col], errors='coerce').to_numpy()
        beh_cols = [c for c in df.columns if c != t_col and not c.lower().startswith('unnamed')]
        if len(times) > 1:
            dt = float(np.nanmedian(np.diff(times)))
        else:
            dt = 0.0
        for c in beh_cols:
            vals = pd.to_numeric(df[c], errors='coerce').fillna(0).to_numpy() > 0
            if not vals.any():
                continue
            changes = np.diff(np.concatenate([[0], vals.astype(int), [0]]))
            starts = np.where(changes == 1)[0]
            stops = np.where(changes == -1)[0]
            for s, e in zip(starts, stops):
                bouts.append((str(c).strip(), float(times[s]), float(times[e-1]) + dt))

    return bouts, fps


def bouts_to_frame_labels(bouts, n_frames, fps, behaviors=None, background_label='none',
                          frame_offset=0):
    """
    Rasterises bouts into per-frame labels.

    Parameters
    ----------
    bouts : list of (behavior, start_s, stop_s)
    n_frames : int, number of frames in the pose-estimation data
    fps : float, frame rate of the video
    behaviors : list of str or None. Order defines priority if behaviours overlap
        (the *last* one in the list wins). None -> all behaviours found, sorted.
    background_label : name for the unlabeled class (index 0)
    frame_offset : int, added to every frame index (e.g. if the scoring started
        at a different frame than the pose-estimation)

    Returns
    -------
    labels : (n_frames,) int array, 0 = background, i = behaviors[i-1]
    multi_hot : (n_frames, len(behaviors)) uint8 array
    class_names : list of str, class_names[0] == background_label
    """
    found = sorted(set(b for b, _, _ in bouts))
    if behaviors is None:
        behaviors = found
    else:
        behaviors = list(behaviors)
        missing = [b for b in behaviors if b not in found]
        if missing:
            print("Warning: behaviours %s not found in the BORIS file" % missing)

    class_names = [background_label] + behaviors
    labels = np.zeros(n_frames, dtype=np.int64)
    multi_hot = np.zeros((n_frames, len(behaviors)), dtype=np.uint8)

    for beh, start, stop in bouts:
        if beh not in behaviors:
            continue
        k = behaviors.index(beh)
        f0 = int(np.floor(start * fps)) + frame_offset
        if np.isinf(stop):
            f1 = n_frames
        else:
            f1 = int(np.ceil(stop * fps)) + frame_offset
            if f1 <= f0:
                f1 = f0 + 1  # point events occupy a single frame
        f0 = max(f0, 0)
        f1 = min(f1, n_frames)
        if f1 <= f0:
            continue
        multi_hot[f0:f1, k] = 1

    # single label per frame: highest priority (= last in list) wins
    for k in range(len(behaviors)):
        labels[multi_hot[:, k] == 1] = k + 1

    return labels, multi_hot, class_names


def _match_boris_files(boris_path, video_names):
    """
    Finds BORIS files for each video. Two layouts are supported:
      1) one file per video: <boris_path>/<video>.csv|tsv|xlsx
      2) one (aggregated) file with several observations: matched by
         'Observation id' or media file name.
    Returns dict video -> (path, observation_filter_or_None)
    """
    matches = {}
    boris_path = str(boris_path)
    exts = ('.csv', '.tsv', '.txt', '.xlsx', '.xls')

    if os.path.isfile(boris_path):
        candidates = [boris_path]
    else:
        candidates = sorted(f for f in glob.glob(os.path.join(boris_path, '*'))
                            if os.path.splitext(f)[1].lower() in exts)

    # layout 1: filename matches video name
    for video in video_names:
        for f in candidates:
            if Path(f).stem == video:
                matches[video] = (f, None)
                break

    # layout 2: multi-observation files
    remaining = [v for v in video_names if v not in matches]
    if remaining:
        for f in candidates:
            try:
                df = read_boris_file(f)
            except Exception as err:
                print("Skipping %s: %s" % (f, err))
                continue
            obs_col = _find_col(df, _OBS_COLS)
            media_col = _find_col(df, _MEDIA_COLS)
            for video in list(remaining):
                if obs_col is not None:
                    ids = df[obs_col].astype(str).unique()
                    if video in ids:
                        matches[video] = (f, (obs_col, video))
                        remaining.remove(video)
                        continue
                if media_col is not None:
                    media = df[media_col].astype(str)
                    stems = media.apply(lambda s: Path(str(s).split(';')[0].strip()).stem)
                    if (stems == video).any():
                        val = media[stems == video].iloc[0]
                        matches[video] = (f, (media_col, val))
                        remaining.remove(video)
    return matches


def boris_to_numpy(config, boris_path=None, fps=None, behaviors=None, subject=None,
                   background_label=None, frame_offset=0, files=None):
    """
    Converts BORIS annotations into per-frame label arrays for every video of
    the project that has a matching BORIS export.

    Parameters
    ----------
    config : path to the project config.yaml
    boris_path : folder with BORIS exports (default: <project>/videos/boris/) or
        a single BORIS export file containing several observations
    fps : frame rate of the videos. If None, the value of ``boris_fps`` in the
        config is used, and if this is None too, the FPS column of the BORIS file.
    behaviors : list of behaviours to keep (default: config ``boris_behaviors``
        or all behaviours found). The order defines the priority for overlapping
        bouts, the last entry wins.
    subject : only keep events of this BORIS subject (multi-animal scoring)
    background_label : name of the unlabeled class (default config
        ``boris_background_label`` or 'none')
    frame_offset : integer frame shift applied to all events
    files : subset of video names to convert (default: all in ``video_sets``)

    Writes
    ------
    data/<video>/<video>-boris-labels.npy        (n_frames,) int
    data/<video>/<video>-boris-multihot.npy      (n_frames, n_behaviors) uint8
    data/<video>/<video>-boris-labels.json       class names, fps, bouts summary
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    project_path = cfg['project_path']

    if boris_path is None:
        boris_path = cfg.get('boris_path') or os.path.join(project_path, 'videos', 'boris')
    if not os.path.isabs(str(boris_path)):
        boris_path = os.path.join(project_path, str(boris_path))
    if fps is None:
        fps = cfg.get('boris_fps')
    if behaviors is None:
        behaviors = cfg.get('boris_behaviors')
        if behaviors in ('None', 'all', ''):
            behaviors = None
    if background_label is None:
        background_label = cfg.get('boris_background_label') or 'none'
    if files is None:
        files = list(cfg['video_sets'])

    if not os.path.exists(boris_path):
        raise FileNotFoundError("BORIS folder/file %s does not exist. Export your BORIS observations "
                                "(Observations -> Export events -> aggregated events) and place them "
                                "there, one file per video named like the video." % boris_path)

    matches = _match_boris_files(boris_path, files)
    if not matches:
        raise FileNotFoundError("No BORIS export matching any of the videos %s was found in %s"
                                % (files, boris_path))

    converted = []
    for video in files:
        if video not in matches:
            print("No BORIS annotation found for %s, skipping." % video)
            continue
        path, obs_filter = matches[video]
        print("Converting BORIS annotation of %s (%s)" % (video, os.path.basename(path)))
        df = read_boris_file(path)
        if obs_filter is not None:
            col, val = obs_filter
            df = df[df[col].astype(str) == str(val)]

        pose_file = os.path.join(project_path, 'data', video, video + '-PE-seq.npy')
        if not os.path.exists(pose_file):
            raise FileNotFoundError("Pose data %s not found. Run vame.egocentric_alignment() or "
                                    "vame.csv_to_numpy() first so the number of frames is known." % pose_file)
        n_frames = np.load(pose_file, mmap_mode='r').shape[1]

        bouts, file_fps = boris_to_bouts(df, subject=subject)
        this_fps = fps if fps is not None else file_fps
        if this_fps is None:
            raise ValueError("Could not determine the frame rate for %s. Pass fps=... or set "
                             "'boris_fps' in the config.yaml." % video)
        this_fps = float(this_fps)

        labels, multi_hot, class_names = bouts_to_frame_labels(
            bouts, n_frames, this_fps, behaviors=behaviors,
            background_label=background_label, frame_offset=frame_offset)

        # keep a consistent behaviour list across videos when it was not fixed
        if behaviors is None:
            behaviors = class_names[1:]

        out_dir = os.path.join(project_path, 'data', video)
        np.save(os.path.join(out_dir, video + '-boris-labels.npy'), labels)
        np.save(os.path.join(out_dir, video + '-boris-multihot.npy'), multi_hot)
        counts = {name: int((labels == i).sum()) for i, name in enumerate(class_names)}
        meta = {'class_names': class_names, 'fps': this_fps, 'n_frames': int(n_frames),
                'n_bouts': len(bouts), 'source': os.path.abspath(path), 'frame_counts': counts,
                'frame_offset': int(frame_offset)}
        with open(os.path.join(out_dir, video + '-boris-labels.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print("  %d bouts -> %d frames, class frame counts: %s" % (len(bouts), n_frames, counts))
        converted.append(video)

    if converted:
        print("\nBORIS labels for %s were saved to data/<video>/<video>-boris-labels.npy. "
              "You can now call vame.train_label_predictor()" % converted)
    return converted
