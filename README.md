# aricode-stdlib-ml

Machine learning primitives for Aricode -- dense layers, loss functions, and optimizers.

## Functions

### dense.ari

| Function | Signature | Description |
|----------|-----------|-------------|
| `dense_forward` | `fn dense_forward(weights: i32, bias: i32, input: i32, output: i32, n_in: i32, n_out: i32) -> i32` | Forward pass for a fully connected layer |
| `dense_apply_relu` | `fn dense_apply_relu(arr: i32, n: i32) -> i32` | In-place ReLU activation over an array |
| `mse_loss` | `fn mse_loss(pred: i32, target: i32, n: i32) -> f64` | Mean Squared Error loss |

### optimizer.ari

| Function | Signature | Description |
|----------|-----------|-------------|
| `sgd_update` | `fn sgd_update(weights: i32, gradients: i32, n: i32, lr: f64) -> i32` | Stochastic Gradient Descent step |
| `zero_gradients` | `fn zero_gradients(gradients: i32, n: i32) -> i32` | Zero out a gradient array |
| `clip_gradients` | `fn clip_gradients(gradients: i32, n: i32, max_val: f64) -> i32` | Clamp gradients to [-max_val, max_val] |

## Usage

```aricode
import "ml/dense.ari";
import "ml/optimizer.ari";

fn main() -> i32 {
    let w: i32 = arr_f64_new(6);
    let b: i32 = arr_f64_new(2);
    let input: i32 = arr_f64_new(3);
    let output: i32 = arr_f64_new(2);

    dense_forward(w, b, input, output, 3, 2);
    dense_apply_relu(output, 2);

    let grad: i32 = arr_f64_new(6);
    sgd_update(w, grad, 6, 0.01);
    return 0;
}
```

## License

Copyright (c) 2026 Edwin F. Veliz Jaramillo. All rights reserved.
