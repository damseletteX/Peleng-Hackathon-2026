import numpy as np

from mv3d.ui.scene import render_space


def test_space_view_draws_the_shared_path():
    fused = [np.array([x, x * 0.2, 300.0 + x * 0.01]) for x in range(0, 500, 20)]
    cameras = {
        "cam1": [point + np.array([30.0, -10.0, 0.0]) for point in fused],
        "cam2": [point + np.array([-20.0, 15.0, 5.0]) for point in fused],
    }
    image = render_space(fused, cameras, width=640, height=420, cursor=len(fused) // 2)
    assert image.shape == (420, 640, 3)
    assert int(image[:, :, 1].max()) > 180
