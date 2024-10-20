import numpy as np

def initialize_weights(hidden_sizes):
    # hidden_sizes = [input_dim, hid_1, hid_2, ..., 1]
    weights = {}
    for i in range(len(hidden_sizes) - 1):
        weights[i+1] = np.random.randn(hidden_sizes[i+1], hidden_sizes[i])
    return weights


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def forward(weights, X):
    acts = {0: X}
    for i, weight in weights.items():
        inp = np.dot(weight, acts[i-1])
        acts[i] = sigmoid(inp)
    return acts, acts[len(weights)]


def compute_loss(y_gt, y_hat):
    return (y_gt - y_hat)**2


def compute_gradients(y_true, y_hat, weights, acts):
    num_layers = len(weights)
    dweights = {}
    for l in range(num_layers, 0, -1):
        if l == num_layers:
            delta = np.dot(acts[l] * (1. - acts[l]), 2 * (y_hat - y_true))
        else:
            delta = acts[l] * (1 - acts[l]) * np.dot(weights[l+1].T, delta)
        dweights[l] = np.dot(delta, acts[l-1].T)
    return dweights


def main():
    input_dim = 3
    output_dim = 1
    assert output_dim == 1, f'output_dim: {output_dim}'
    batch_size = 50
    X_batch = np.random.random((batch_size, input_dim, 1))
    Y_gt = np.random.random((batch_size, output_dim))
    print(f'X shape: {X_batch.shape}. Y shape: {Y_gt.shape}')
    hidden_sizes = [input_dim, 4, 12, 8, output_dim]
    weights = initialize_weights(hidden_sizes)
    for i, weight in weights.items():
        print(f'layer: {i}, weight shape: {weight.shape}')

    for step in range(20):
        dweights_cum = {} 
        loss_cum = 0.
        for i, X in enumerate(X_batch):
            y_gt = Y_gt[i]
            acts, y_hat = forward(weights, X)
            # for i, act in acts.items():
            #     print(f'layer: {i}, activation shape: {act.shape}')
            loss = compute_loss(y_gt, y_hat)
            loss_cum += loss[0][0]
            assert loss.shape == (1, 1), f'loss shape: {loss.shape}'
            dweights = compute_gradients(y_gt, y_hat, weights, acts)
            for l, dweight in dweights.items():
                dweights_cum[l] = dweights_cum.get(l, 0) + dweight
        loss_cum /= len(X_batch)
        print(f'step: {step}, loss: {loss_cum:0.6f}')
        for l, dweight in dweights_cum.items():
            weights[l] -= 0.1 * dweight


if __name__ == '__main__':
    main()