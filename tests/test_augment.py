import random

import numpy as np

from mv3d.detection.augment import crop, horizontal_flip, photometric, rotate, vertical_flip
from mv3d.detection.train import LabeledFrame, leave_one_camera_out, write_augmented_dataset


def test_horizontal_flip_mirrors_box():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    flipped, boxes = horizontal_flip(image, [(10.0, 2.0, 15.0, 8.0)])
    assert flipped.shape == image.shape
    assert boxes == [(25.0, 2.0, 30.0, 8.0)]


def test_vertical_flip_mirrors_box():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    _flipped, boxes = vertical_flip(image, [(4.0, 1.0, 12.0, 6.0)])
    assert boxes == [(4.0, 14.0, 12.0, 19.0)]


def test_zero_rotation_keeps_box():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    _, boxes = rotate(image, [(10.0, 20.0, 30.0, 50.0)], 0.0)
    assert len(boxes) == 1
    for got, expected in zip(boxes[0], (10.0, 20.0, 30.0, 50.0), strict=True):
        assert abs(got - expected) < 1.0


def test_crop_shifts_box_and_drops_outside():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    cropped, boxes = crop(
        image,
        [(10.0, 10.0, 40.0, 40.0), (90.0, 80.0, 110.0, 95.0)],
        (5, 5),
        (50, 50),
    )
    assert cropped.shape == (50, 50, 3)
    assert boxes == [(5.0, 5.0, 35.0, 35.0)]


def test_photometric_keeps_shape():
    image = np.full((16, 16, 3), 40, dtype=np.uint8)
    adjusted = photometric(image, random.Random(1))
    assert adjusted.shape == image.shape


def test_dataset_writer_augments_train_only(tmp_path):
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    box = [(8.0, 8.0, 40.0, 40.0)]
    train_frames = [LabeledFrame("cam1", index, frame, box) for index in range(2)]
    valid_frames = [LabeledFrame("cam2", 0, frame, box)]
    root = write_augmented_dataset(train_frames, valid_frames, tmp_path / "truck", seed=2)
    train = list((root / "train").glob("*.jpg"))
    valid = list((root / "valid").glob("*.jpg"))
    assert len(valid) == 1
    assert valid[0].name.endswith("_orig.jpg")
    assert len(train) > len(train_frames)
    assert (root / "train" / "_annotations.coco.json").is_file()


def test_leave_one_camera_out_holds_out_each_camera():
    frames = [
        LabeledFrame(camera, index, np.zeros((8, 8, 3), dtype=np.uint8), [(1.0, 1.0, 5.0, 5.0)])
        for camera in ("cam1", "cam2", "cam3", "cam4", "cam5")
        for index in range(2)
    ]
    folds = leave_one_camera_out(frames)
    assert [held for held, _train, _valid in folds] == ["cam1", "cam2", "cam3", "cam4", "cam5"]
    for held, train, valid in folds:
        assert {frame.camera_id for frame in valid} == {held}
        assert held not in {frame.camera_id for frame in train}
        assert len(valid) == 2
        assert len(train) == 8
