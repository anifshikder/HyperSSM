"""Continuous inputs: learn to integrate a (miscalibrated) gyroscope with one quaternion channel.

    python examples/continuous_inputs.py

A body turns with angular velocity w_k; its orientation after T steps is the ORDERED product of
the small rotations exp(w_k dt / 2). The simulated sensor reports a scaled, biased reading.
HyperSSM's channel has transition exp(W u_k + b): the learned W and b are the sensor's
calibration, read straight off the weights. A commutative memory cannot do this, because the
same rotations in a different order end in a different orientation.
"""
import torch

from hyperssm import HyperSSM, Quaternion, prefix_scan

torch.manual_seed(0)
dt, T_train, T_test = 0.02, 50, 1000
true_scale, true_bias = torch.tensor([1.05, 0.97, 1.02]), torch.tensor([0.03, -0.02, 0.01])


def batch(B, T):
    w = torch.cumsum(0.3 * torch.randn(B, T, 3), 1).clamp(-3, 3)          # smooth angular velocity
    reading = w / true_scale + true_bias                                     # what the sensor reports
    H = Quaternion()
    q = prefix_scan(H.transition(w * dt / 2), H.mul)[:, -1]                 # true orientation
    return reading, q


def predict(model, reading):
    return model.states(reading)[-1][:, -1, 0]                               # the channel's state


model = HyperSSM(1, layers=["H"], d_in=3, init_scale=0.001)                 # (the readout head is unused)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(3000):
    reading, q = batch(64, T_train)
    loss = (1 - (predict(model, reading) * q).sum(-1) ** 2).mean()          # ignores the sign of q
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    reading, q = batch(256, T_test)
    err = torch.rad2deg(2 * torch.arccos((predict(model, reading) * q).sum(-1).abs().clamp(max=1)))
    W, b = model.layers[0].generator.weight, model.layers[0].generator.bias
    print(f"trained on {T_train} steps; median orientation error at {T_test} steps: {err.median():.3f} degrees")
    print("learned W / (dt/2):", (torch.diag(W) / (dt / 2)).numpy().round(3), " true scale:", true_scale.numpy())
    print("learned bias correction:", (-(b / torch.diag(W))).numpy().round(3), " true bias:", true_bias.numpy())
