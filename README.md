# aricode-ml

Neural network primitives for the
[aricode](https://github.com/Lynx-Boss/aricode) compiler: dense layers,
stochastic gradient descent, and mean-squared-error loss.

## Layout

```
aricode-ml/
├── dense.ari         — dense_forward, dense_apply_relu, mse_loss
├── optimizer.ari     — sgd_update, zero_gradients, clip_gradients
├── examples/
│   └── one_step.ari  — forward + manual grad + one SGD step
└── tests/
    ├── test_dense.ari
    └── test_optimizer.ari
```

## Public API

### dense.ari

| Function | Description |
|----------|-------------|
| `dense_forward(w, b, x, y, n_in, n_out)` | Fully-connected forward pass, `y = W·x + b`. Arrays are heap pointers returned by `arr_f64_new`. |
| `dense_apply_relu(buf, n)` | In-place ReLU over a length-`n` f64 array. |
| `mse_loss(pred, target, n) -> f64` | `(1/n) · Σ (pred - target)²`. |

### optimizer.ari

| Function | Description |
|----------|-------------|
| `sgd_update(w, g, n, lr)` | `w[i] -= lr · g[i]`. |
| `zero_gradients(g, n)` | Reset gradient buffer. |
| `clip_gradients(g, n, max)` | Clamp each entry to `[-max, max]`. |

## Usage

```
import "aricode-ml/dense.ari" as dense;
import "aricode-ml/optimizer.ari" as opt;

fn main() -> i32 {
    let w: i32 = arr_f64_new(6);
    let b: i32 = arr_f64_new(2);
    let x: i32 = arr_f64_new(3);
    let y: i32 = arr_f64_new(2);

    dense.dense_forward(w, b, x, y, 3, 2);
    dense.dense_apply_relu(y, 2);

    let g: i32 = arr_f64_new(6);
    opt.sgd_update(w, g, 6, 0.01);
    return 0;
}
```

See [`examples/one_step.ari`](examples/one_step.ari) for a full training
step with manual gradient computation.

## Running the tests

```
aric tests/test_dense.ari     -o /tmp/test_dense     && /tmp/test_dense
aric tests/test_optimizer.ari -o /tmp/test_optimizer && /tmp/test_optimizer
```

## Dependencies

This module relies on the `arr_f64_*` heap-array builtins (`arr_f64_new`,
`arr_f64_get`, `arr_f64_set`) provided by the aricode compiler.

## License

Copyright (c) 2026 Edwin F. Veliz Jaramillo. All rights reserved.
