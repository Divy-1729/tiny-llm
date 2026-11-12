import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # Supplied for Day 1; a manual implementation is an optional bonus exercise.
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    pre_bias = mx.matmul(x, w.T)
    if bias is not None:
        return pre_bias + bias
    return pre_bias
    


def silu(x: mx.array) -> mx.array:
    pass
