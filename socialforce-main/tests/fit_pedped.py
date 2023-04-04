import math
import random

import numpy as np
import pytest
import torch
import scipy.optimize
import socialforce


OPTIMIZER_OPT = {'eps': 1e-4, 'gtol': 1e-4, 'maxcor': 30, 'maxls': 10, 'disp': True}


def visualize(file_prefix, V, initial_parameters, final_parameters, fit_result=None, V_gen=None):
    b = np.linspace(0, 3, 200)
    y_ref = 2.1 * np.exp(-1.0 * b / 0.3)
    if V_gen is not None:
        y_ref = V_gen.v0 * np.exp(-1.0 * b / V_gen.sigma)

    V.set_parameters(torch.tensor(initial_parameters))
    y_initial = V.value_b(torch.from_numpy(b).float()).detach().numpy()
    y_initial -= y_initial[-1]

    if not isinstance(final_parameters, torch.Tensor):
        final_parameters = torch.tensor(final_parameters)
    V.set_parameters(final_parameters)
    y_mlp = V.value_b(torch.from_numpy(b).float()).detach().numpy()
    y_mlp -= y_mlp[-1]

    with socialforce.show.graph(file_prefix + 'v.png') as ax:
        ax.set_xlabel('$b$ [m]')
        ax.set_ylabel('$V$')
        ax.plot(b, y_ref, label=r'true $V_0 e^{-b/\sigma}$', color='C0')
        ax.axvline(0.3, color='C0', linestyle='dotted', label=r'true $\sigma$')
        ax.plot(b, y_initial, label=r'initial MLP($b$)',
                linestyle='dashed', color='orange')
        ax.plot(b, y_mlp, label=r'MLP($b$)', color='orange')
        ax.legend()

    with socialforce.show.graph(file_prefix + 'gradv.png') as ax:
        ax.set_xlabel(r'$b$ [m]')
        ax.set_ylabel(r'$\nabla V$')
        delta_b = b[1:] - b[:-1]
        average_b = 0.5 * (b[:-1] + b[1:])
        ax.plot(average_b, (y_ref[1:] - y_ref[:-1]) / delta_b,
                label=r'true $V_0 e^{-b/\sigma}$', color='C0')
        ax.axvline(0.3, color='C0', linestyle='dotted', label=r'true $\sigma$')
        ax.plot(average_b, (y_initial[1:] - y_initial[:-1]) / delta_b,
                label=r'initial MLP($b$)',
                linestyle='dashed', color='orange')
        ax.plot(average_b, (y_mlp[1:] - y_mlp[:-1]) / delta_b,
                label=r'MLP($b$)', color='orange')
        ax.set_ylim(-4.9, 0.5)
        ax.legend()


def test_opposing(lr=0.5):
    initial_state = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.0, 10.0],
        [-0.3, 10.0, 0.0, -1.0, -0.3, 0.0],
    ])
    generator = socialforce.Simulator(initial_state)
    truth = torch.stack([generator.step().state[:, 0:2].clone() for _ in range(21)]).detach()

    print('============DONE WITH GENERATION===============')

    v0 = torch.tensor(1.2, requires_grad=True)
    sigma_v = torch.tensor(0.1, requires_grad=True)
    V = socialforce.PedPedPotential(0.4, v0, sigma_v)

    # training
    for _ in range(100):
        print('loop')
        s = socialforce.Simulator(initial_state, None, V)
        g = torch.stack([s.step().state[:, 0:2].clone() for _ in range(21)])
        loss = (g - truth).pow(2).sum()

        v0_grad, sigma_grad = torch.autograd.grad(loss, [v0, sigma_v])
        print('v0', v0, v0_grad)
        print('sigma', sigma_v, sigma_grad)

        with torch.no_grad():
            v0 -= lr * v0_grad
            sigma_v -= lr * sigma_grad

    assert v0.item() == pytest.approx(2.1, abs=0.01)
    assert sigma_v.item() == pytest.approx(0.3, abs=0.01)


def test_opposing_scipy():
    initial_state = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.0, 10.0],
        [-0.3, 10.0, 0.0, -1.0, -0.3, 0.0],
    ])
    generator = socialforce.Simulator(initial_state)
    truth = torch.stack([generator.step().state[:, 0:2].clone() for _ in range(21)]).detach()

    print('============DONE WITH GENERATION===============')

    # training
    def f(x):
        v0 = torch.tensor(x[0], requires_grad=True)
        sigma_v = float(x[1])
        V = socialforce.PedPedPotential(0.4, v0, sigma_v)
        s = socialforce.Simulator(initial_state, None, V)
        g = torch.stack([s.step().state[:, 0:2].clone() for _ in range(21)])

        # average euclidean distance loss
        loss = (g - truth).pow(2).sum()
        return loss

    parameters = np.array([1.2, 0.1])
    res = scipy.optimize.minimize(f, parameters, method='L-BFGS-B',
                                  options=OPTIMIZER_OPT)
    print(res)
    assert res.x == pytest.approx(np.array([2.1, 0.3]), abs=0.01)


def test_opposing_mlp_scipy():
    torch.manual_seed(42)
    np.random.seed(42)

    initial_state = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.0, 10.0],
        [-0.3, 10.0, 0.0, -1.0, -0.3, 0.0],
    ])
    generator = socialforce.Simulator(initial_state)
    truth = torch.stack([generator.step().state[:, 0:2].clone() for _ in range(21)]).detach()

    print('============DONE WITH GENERATION===============')

    V = socialforce.PedPedPotentialMLP(0.4)
    parameters = V.get_parameters().clone().detach().numpy()
    initial_parameters = parameters.copy()

    # training
    def f(x):
        V.set_parameters(torch.from_numpy(x))
        s = socialforce.Simulator(initial_state, None, V)
        g = torch.stack([s.step().state[:, 0:2].clone() for _ in range(21)])

        # average euclidean distance loss
        diff = g - truth
        loss = torch.mul(diff, diff).sum(dim=-1).mean()

        return float(loss)

    res = scipy.optimize.minimize(f, parameters, method='L-BFGS-B',
                                  options=OPTIMIZER_OPT)
    print(res)

    # make plots of result
    visualize('docs/mlp_scipy_', V, initial_parameters, res.x)


@pytest.mark.parametrize('n', [1, 5, 20])
def test_circle_mlp(n, lr=0.1, delta_t=0.2):
    torch.manual_seed(42)
    np.random.seed(42)

    # ped0 always left to right
    ped0 = np.array([-5.0, 0.0, 1.0, 0.0, 5.0, 0.0])

    generator_initial_states = []
    for theta in np.random.rand(n) * 2.0 * math.pi:
        # ped1 at a random angle with +/-20% speed variation
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([[c, -s], [s, c]])
        ped1 = np.concatenate((
            np.matmul(r, ped0[0:2]),
            np.matmul(r, ped0[2:4]),  # * (0.8 + random.random() * 0.4)),
            np.matmul(r, ped0[4:6]),
        ))
        generator_initial_states.append(
            torch.tensor(np.stack((ped0, ped1))).float()
        )
    if n == 1:  # override for n=1
        generator_initial_states = [torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 0.0, 10.0],
            [-0.3, 10.0, 0.0, -1.0, -0.3, 0.0],
        ])]

    # generator_v0 = 1.5 if n != 1 else 2.1
    generator_ped_ped = socialforce.PedPedPotential(delta_t, 2.1)
    true_scenes = []
    for initial_state in generator_initial_states:
        generator = socialforce.Simulator(initial_state, ped_ped=generator_ped_ped, delta_t=delta_t)
        true_scenes.append(
            [generator.step().state.clone().detach() for _ in range(42)]
        )
    true_experience = socialforce.Optimizer.scenes_to_experience(true_scenes)

    print('============DONE WITH GENERATION===============')

    V = socialforce.PedPedPotentialMLP(delta_t)
    initial_parameters = V.get_parameters().clone().detach().numpy()

    def simulator_factory(initial_state):
        s = socialforce.Simulator(initial_state, ped_ped=V, delta_t=delta_t)
        s.desired_speeds = generator.desired_speeds
        s.max_speeds = generator.max_speeds
        return s

    opt = socialforce.Optimizer(simulator_factory, V.parameters(), true_experience)
    for i in range(100 // n):
        loss = opt.epoch()
        print('epoch {}: {}'.format(i, loss))

    # make plots of result
    visualize('docs/mlp_circle_n{}_'.format(n), V, initial_parameters, V.get_parameters().clone(), V_gen=generator_ped_ped)
