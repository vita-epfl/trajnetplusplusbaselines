import torch


def test_slice_assignment():
    a = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(1.0, requires_grad=True)
    y = a * x
    grad_y, = torch.autograd.grad(y, x, create_graph=True)
    print(grad_y)

    xnew = torch.ones(2)
    xnew[0] = grad_y  # assign to slice
    print(xnew[0])

    grad_ya, = torch.autograd.grad(xnew.sum(), a)
    print('grad_ya', grad_ya)
    assert grad_ya == 1.0


def test_product():
    a = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor([1.0, 0.0], requires_grad=True)
    y = a * x * torch.norm(x, dim=0, keepdim=True)
    grad_y, = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)
    print(grad_y)

    grad_ya, = torch.autograd.grad(grad_y.sum(), a)
    print('grad_ya', grad_ya)
    assert grad_ya == 3.0


def test_norm():
    a = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor([[
        [1.0, 0.0],
    ]], requires_grad=True)
    y = a * torch.norm(x, dim=-1, keepdim=True)
    print(y)
    grad_y, = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)
    print(grad_y)

    grad_ya, = torch.autograd.grad(grad_y.sum(), a)
    print('grad_ya', grad_ya)
    assert grad_ya == 1.0


def test_mlp_bias():
    lin1 = torch.nn.Linear(1, 5)
    lin2 = torch.nn.Linear(5, 1)

    # initialize
    torch.nn.init.normal_(lin1.weight, std=0.01)
    torch.nn.init.normal_(lin1.bias, std=0.01)
    torch.nn.init.normal_(lin2.weight, std=0.01)
    torch.nn.init.normal_(lin2.bias, std=0.00000001)

    mlp = torch.nn.Sequential(lin1, torch.nn.Tanh(), lin2)
    v = mlp(torch.ones(1, 1))
    grads = torch.autograd.grad(v, [lin1.weight, lin1.bias, lin2.weight, lin2.bias])
    print(grads)
