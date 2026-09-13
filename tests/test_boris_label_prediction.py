"""
Tests for the BORIS label import and label-prediction functions.
A small synthetic VAME project is created in a temporary directory: pose
data, latent vectors and motif labels are simulated so that the tests run
without a trained VAME model.
"""
import os
import json
import numpy as np
import pandas as pd
import pytest

from vame.util import auxiliary
from vame.util.boris_to_labels import (read_boris_file, detect_boris_format, boris_to_bouts,
                                       bouts_to_frame_labels, boris_to_numpy)
from vame.analysis.label_prediction import (train_label_predictor, predict_labels,
                                            load_video_features, pose_features, _cv_splits)

FPS = 30.0
N_FRAMES = 1500
TIME_WINDOW = 30
ZDIMS = 8
N_FEATURES = 6
VIDEOS = ['video-1', 'video-2']


def _make_project(tmp_path):
    project_path = tmp_path / 'proj'
    for v in VIDEOS:
        (project_path / 'data' / v).mkdir(parents=True)
        (project_path / 'results' / v / 'VAME' / 'kmeans-3').mkdir(parents=True)
    (project_path / 'videos' / 'boris').mkdir(parents=True)

    cfg, _ = auxiliary.create_config_template()
    cfg['Project'] = 'proj'
    cfg['project_path'] = str(project_path) + '/'
    cfg['video_sets'] = VIDEOS
    cfg['model_name'] = 'VAME'
    cfg['n_cluster'] = 3
    cfg['parameterization'] = 'kmeans'
    cfg['time_window'] = TIME_WINDOW
    cfg['num_features'] = N_FEATURES
    cfg['egocentric_data'] = True
    cfg['boris_path'] = 'videos/boris/'
    cfg['boris_fps'] = FPS
    cfg['label_n_folds'] = 3
    cfg['label_smoothing_window'] = 5
    cfg['label_pose_window'] = 9
    config = project_path / 'config.yaml'
    auxiliary.write_config(str(config), cfg)

    rng = np.random.RandomState(0)
    truth = {}
    for i, v in enumerate(VIDEOS):
        # three behavioural states, in blocks of 100 frames
        state = (np.arange(N_FRAMES) // 100) % 3
        # pose and latent vectors carry a state-dependent signal
        pose = rng.randn(N_FRAMES, N_FEATURES) * 0.3 + state[:, None] * np.linspace(0.5, 1.5, N_FEATURES)
        latent = rng.randn(N_FRAMES - TIME_WINDOW, ZDIMS) * 0.3
        latent[:, :3] += np.eye(3)[state[TIME_WINDOW // 2: TIME_WINDOW // 2 + N_FRAMES - TIME_WINDOW]] * 2
        motif = state[TIME_WINDOW // 2: TIME_WINDOW // 2 + N_FRAMES - TIME_WINDOW]
        np.save(project_path / 'data' / v / (v + '-PE-seq.npy'), pose.T)
        np.save(project_path / 'data' / v / (v + '-PE-seq-clean.npy'), pose.T)
        np.save(project_path / 'results' / v / 'VAME' / 'kmeans-3' / ('latent_vector_' + v + '.npy'), latent)
        np.save(project_path / 'results' / v / 'VAME' / 'kmeans-3' / ('3_km_label_' + v + '.npy'), motif)
        truth[v] = state

    # BORIS aggregated export for video-1 (state 1 = 'rearing', state 2 = 'grooming', 0 = unscored)
    rows = []
    state = truth['video-1']
    change = np.where(np.diff(state) != 0)[0] + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [N_FRAMES]])
    names = {1: 'rearing', 2: 'grooming'}
    for s, e in zip(starts, stops):
        if state[s] in names:
            rows.append({'Observation id': 'video-1', 'Observation date': '2024-01-01', 'Media file': '/x/video-1.mp4',
                         'Total length': N_FRAMES / FPS, 'FPS': FPS, 'Subject': 'mouse', 'Behavior': names[state[s]],
                         'Behavioral category': '', 'Modifiers': '', 'Behavior type': 'STATE',
                         'Start (s)': s / FPS, 'Stop (s)': e / FPS, 'Duration (s)': (e - s) / FPS,
                         'Comment start': '', 'Comment stop': ''})
    pd.DataFrame(rows).to_csv(project_path / 'videos' / 'boris' / 'video-1.csv', index=False)
    return str(config), project_path, truth


@pytest.fixture
def project(tmp_path):
    return _make_project(tmp_path)


def test_bouts_to_frame_labels_priority_and_points():
    bouts = [('walk', 0.0, 1.0), ('rear', 0.5, 0.8), ('sniff', 2.0, 2.0)]
    labels, multi, names = bouts_to_frame_labels(bouts, n_frames=100, fps=10.0,
                                                 behaviors=['walk', 'rear', 'sniff'])
    assert names == ['none', 'walk', 'rear', 'sniff']
    assert labels[0] == 1 and labels[9] == 1
    assert labels[5] == 2  # overlap: later entry in list wins
    assert multi[5].tolist() == [1, 1, 0]
    assert labels[20] == 3 and labels[21] == 0  # point event = one frame
    assert (labels[30:] == 0).all()


def test_tabulated_and_binary_formats(tmp_path):
    tab = pd.DataFrame({'Time': [0.0, 1.0, 1.5, 2.0, 3.0], 'Media file path': 'v.mp4', 'Total length': 10,
                        'FPS': 10, 'Subject': '', 'Behavior': ['walk', 'walk', 'sniff', 'rear', 'rear'],
                        'Modifiers': '', 'Behavior type': '', 'Comment': '',
                        'Status': ['START', 'STOP', 'POINT', 'START', 'STOP']})
    p = tmp_path / 'tab.tsv'
    tab.to_csv(p, sep='\t', index=False)
    df = read_boris_file(p)
    assert detect_boris_format(df) == 'tabulated'
    bouts, fps = boris_to_bouts(df)
    assert fps == 10
    assert sorted(bouts) == [('rear', 2.0, 3.0), ('sniff', 1.5, 1.5), ('walk', 0.0, 1.0)]

    binary = pd.DataFrame({'time': np.arange(0, 1.0, 0.1), 'walk': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           'rear': [0, 0, 0, 0, 1, 1, 0, 0, 0, 1]})
    p = tmp_path / 'bin.csv'
    binary.to_csv(p, index=False)
    df = read_boris_file(p)
    assert detect_boris_format(df) == 'binary'
    bouts, _ = boris_to_bouts(df)
    bouts = sorted((b, round(s, 3), round(e, 3)) for b, s, e in bouts)
    assert bouts == [('rear', 0.4, 0.6), ('rear', 0.9, 1.0), ('walk', 0.0, 0.3)]


def test_boris_file_with_preamble(tmp_path):
    p = tmp_path / 'agg.csv'
    with open(p, 'w') as f:
        f.write('# exported by BORIS\n\n')
        f.write('Observation id,Behavior,Start (s),Stop (s),FPS\n')
        f.write('obs1,walk,0.0,0.5,25\n')
    df = read_boris_file(p)
    assert detect_boris_format(df) == 'aggregated'
    bouts, fps = boris_to_bouts(df)
    assert bouts == [('walk', 0.0, 0.5)] and fps == 25


def test_boris_to_numpy(project):
    config, project_path, truth = project
    converted = boris_to_numpy(config)
    assert converted == ['video-1']
    labels = np.load(project_path / 'data' / 'video-1' / 'video-1-boris-labels.npy')
    meta = json.load(open(project_path / 'data' / 'video-1' / 'video-1-boris-labels.json'))
    assert meta['class_names'] == ['none', 'grooming', 'rearing']
    assert len(labels) == N_FRAMES
    expected = np.array([{0: 0, 1: 2, 2: 1}[s] for s in truth['video-1']])
    assert (labels == expected).mean() > 0.98


def test_boris_to_numpy_multi_observation_file(project):
    config, project_path, truth = project
    # move the per-video file into one multi-observation file with a different name
    df = pd.read_csv(project_path / 'videos' / 'boris' / 'video-1.csv')
    os.remove(project_path / 'videos' / 'boris' / 'video-1.csv')
    df.to_csv(project_path / 'videos' / 'boris' / 'all_observations.csv', index=False)
    assert boris_to_numpy(config, behaviors=['rearing', 'grooming']) == ['video-1']
    meta = json.load(open(project_path / 'data' / 'video-1' / 'video-1-boris-labels.json'))
    assert meta['class_names'] == ['none', 'rearing', 'grooming']


def test_feature_alignment(project):
    config, project_path, _ = project
    cfg = auxiliary.read_config(config)
    X, frames, n_frames, names = load_video_features(cfg, 'video-1', ['latent', 'pose', 'motif'], pose_window=9)
    assert n_frames == N_FRAMES
    assert frames[0] == TIME_WINDOW // 2 and len(frames) == N_FRAMES - TIME_WINDOW
    assert X.shape == (N_FRAMES - TIME_WINDOW, ZDIMS + 3 + len(names) - ZDIMS - 3)
    assert names[:ZDIMS] == ['latent_%d' % i for i in range(ZDIMS)]
    Xp, frames_p, _, _ = load_video_features(cfg, 'video-1', ['pose'], pose_window=1)
    assert Xp.shape == (N_FRAMES, 2 * N_FEATURES) and len(frames_p) == N_FRAMES


def test_pose_features_shape():
    pose = np.random.randn(4, 50)
    assert pose_features(pose, window=5).shape == (50, 4 * 4 + 1)
    assert pose_features(pose, window=1).shape == (50, 8)


def test_cv_splits_blocked_and_video():
    groups = np.array(['a'] * 100 + ['b'] * 100)
    y = np.zeros(200)
    splits = list(_cv_splits(groups, y, cv='video', n_folds=5))
    assert len(splits) == 2
    for tr, te in splits:
        assert len(set(groups[tr]) & set(groups[te])) == 0
    splits = list(_cv_splits(groups, y, cv='blocked', n_folds=4))
    assert len(splits) == 4
    tr, te = splits[0]
    assert set(te.tolist()) == set(range(0, 25)) | set(range(100, 125))


@pytest.mark.parametrize('features', ['latent', 'pose', 'latent+pose', 'motif'])
def test_train_and_predict(project, features):
    config, project_path, truth = project
    boris_to_numpy(config)
    report = train_label_predictor(config, features=features, classifier='logistic',
                                   cv='blocked', n_folds=3, predict=False)
    assert report['class_names'] == ['none', 'grooming', 'rearing']
    assert report['accuracy'] > 0.85, report['accuracy']
    assert len(report['folds']) == 3
    model_dir = project_path / 'results' / 'label_prediction' / 'VAME' / (features + '_logistic')
    assert (model_dir / 'classifier.pkl').exists()
    assert (model_dir / 'cv_report.json').exists()
    assert (model_dir / 'cv_confusion_matrix.csv').exists()

    results = predict_labels(config, features=features, classifier='logistic')
    assert set(results) == set(VIDEOS)
    for v in VIDEOS:
        pred = results[v]
        assert pred.shape == (N_FRAMES,)
        expected = np.array([{0: 0, 1: 2, 2: 1}[s] for s in truth[v]])
        assert (pred == expected).mean() > 0.85, (v, (pred == expected).mean())
        out = project_path / 'results' / v / 'VAME' / 'label_prediction'
        csv = pd.read_csv(out / ('boris_prediction_%s_logistic_%s.csv' % (features, v)))
        assert list(csv.columns[:4]) == ['frame', 'time_s', 'label', 'behavior']
        assert len(csv) == N_FRAMES
        bouts = pd.read_csv(out / ('boris_prediction_bouts_%s_logistic_%s.csv' % (features, v)))
        assert 'Start (s)' in bouts.columns and bouts['Duration (frames)'].sum() == N_FRAMES


def test_train_random_forest_feature_importance(project):
    config, project_path, _ = project
    boris_to_numpy(config)
    report = train_label_predictor(config, features='latent+pose', classifier='random_forest',
                                   n_folds=2, predict=True)
    assert report['accuracy'] > 0.85
    model_dir = project_path / 'results' / 'label_prediction' / 'VAME' / 'latent+pose_random_forest'
    imp = pd.read_csv(model_dir / 'feature_importance.csv')
    assert len(imp) == len(report['feature_names'])
    assert (project_path / 'results' / 'video-2' / 'VAME' / 'label_prediction'
            / 'boris_prediction_latent+pose_random_forest_video-2.npy').exists()


def test_ignore_background(project):
    config, project_path, truth = project
    boris_to_numpy(config)
    report = train_label_predictor(config, features='latent', classifier='logistic', cv='blocked',
                                   n_folds=2, ignore_background=True, predict=False)
    assert report['class_frame_counts']['none'] == 0
    assert 'none' not in report['per_class']
