import pytest
import torch

from somato.sensors import SENSOR_MODELS, SensorSuite


def pressure_steps(levels, T_each=100, N=3):
    x = torch.zeros(1, T_each * len(levels), N, 3)
    for i, p in enumerate(levels):
        x[:, i * T_each : (i + 1) * T_each, :, 0] = p
    return x


@pytest.mark.parametrize("name", ["ideal_tactile", "fsr", "capacitive"])
def test_tactile_monotonic(name):
    model = SENSOR_MODELS.build(name, dt=1e-3)
    if hasattr(model, "noise_std"):
        model.noise_std = 0.0
    if hasattr(model, "drift_std"):
        model.drift_std = 0.0
    gen = torch.Generator().manual_seed(0)
    out, _ = model(pressure_steps([0, 5e3, 2e4, 6e4]), generator=gen)
    means = [float(out[0, i * 100 + 80 : (i + 1) * 100, :, 0].mean()) for i in range(4)]
    assert means == sorted(means) and means[-1] > means[0]


def test_fsr_threshold_saturation_and_quantization():
    m = SENSOR_MODELS.build("fsr", dt=1e-3, noise_std=0.0)
    out, _ = m(pressure_steps([500.0, 1e7]), generator=torch.Generator().manual_seed(0))  # 0.05 N (< threshold), 1 kN
    assert float(out[0, 50:100].max()) == 0.0
    assert 0.9 < float(out[0, -1].min()) <= 1.0
    levels = out.unique()
    assert torch.allclose(levels * 4095, (levels * 4095).round(), atol=1e-3)


def test_fsr_hysteresis_and_creep():
    m = SENSOR_MODELS.build("fsr", dt=1e-3, noise_std=0.0, gain_spread=0.0, adc_bits=24)
    x = pressure_steps([0.0, 3e4, 0.0], T_each=200)
    out, _ = m(x)
    v = out[0, :, 0, 0]
    rise = int((v[200:] > 0.5 * v[399]).nonzero()[0])
    fall = int((v[400:] < 0.5 * v[399]).nonzero()[0])
    assert fall > rise  # unloading is slower than loading
    assert v[399] > v[230]  # creep: reading keeps growing under constant load


def test_sensor_state_continuity():
    """Processing a sequence in two chunks with carried state equals processing it at once."""
    for name in ["fsr", "capacitive", "motor"]:
        m = SENSOR_MODELS.build(name, dt=1e-3)
        for attr in ("noise_std", "drift_std", "torque_noise"):
            if hasattr(m, attr):
                setattr(m, attr, 0.0)
        C = 4 if m.kind == "joint" else 3
        x = torch.rand(2, 60, 5, C) * 2e4
        if m.kind == "joint":
            x = torch.randn(2, 60, 5, C)
        state = m.init_state(x, torch.Generator().manual_seed(0))
        full, _ = m(x, {k: v.clone() for k, v in state.items()})
        a, st = m(x[:, :25], {k: v.clone() for k, v in state.items()})
        b, _ = m(x[:, 25:], st)
        assert torch.allclose(full, torch.cat([a, b], 1), atol=1e-5), name


def test_imu_noise_statistics():
    m = SENSOR_MODELS.build("mems_imu", dt=1e-3, acc_bias_std=0.0, gyro_bias_std=0.0, scale_std=0.0)
    x = torch.zeros(1, 4000, 1, 6)
    x[..., 2] = 9.81
    out, _ = m(x, generator=torch.Generator().manual_seed(0))
    assert abs(float(out[..., 2].mean()) - 9.81) < 0.02
    expected = m.acc_noise_density / (1e-3) ** 0.5
    assert abs(float(out[..., 0].std()) / expected - 1) < 0.15


def test_motor_quantization_and_error_channel():
    m = SENSOR_MODELS.build("motor", dt=2e-3, counts_per_rev=1024, offset_std=0.0, torque_noise=0.0)
    x = torch.zeros(1, 50, 2, 4)
    x[..., 0] = 0.1234
    x[..., 3] = 0.2
    out, _ = m(x)
    step = 2 * torch.pi / 1024
    assert torch.allclose(out[..., 0] / step, (out[..., 0] / step).round(), atol=1e-4)
    assert torch.allclose(out[..., 3], 0.2 - out[..., 0])
    assert m.output_channels == ("pos", "vel", "torque", "error")


def test_suite_from_config(small_layout):
    rates = {"tactile": 1000, "joint": 500, "imu": 500}
    suite = SensorSuite.from_config({"tactile": {"model": "capacitive"}, "joint": {"model": "ideal_joint"},
                                     "imu": {"model": "ideal_imu"}}, small_layout, rates)
    stim = {"tactile": torch.rand(2, 10, 60, 3) * 1e4, "joint": torch.randn(2, 5, 4, 4), "imu": torch.randn(2, 5, 1, 6)}
    readings, _ = suite(stim)
    assert readings["tactile"].shape == (2, 10, 60, 1)
    assert suite.output_channels["joint"] == ("pos", "vel", "torque", "error")
    with pytest.raises(ValueError):
        SensorSuite.from_config({"tactile": {"model": "motor"}, "joint": {"model": "motor"},
                                 "imu": {"model": "mems_imu"}}, small_layout, rates)
