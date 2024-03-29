import functools
import aikit
import aikit.functional.frontends.tensorflow as tf_frontend
from aikit.functional.frontends.tensorflow.func_wrapper import (
    _aikit_array_to_tensorflow,
    _to_aikit_array,
    to_aikit_arrays_and_back,
)


def bias_add(x, bias, data_format=None):
    if data_format is None:
        data_format = "channels_last"
    bias_shape = bias.shape
    if len(bias_shape) == 1:
        if data_format == "channels_first":
            return tf_frontend.nn.bias_add(x, bias, data_format="NC...")
        return tf_frontend.nn.bias_add(x, bias, data_format="N...C")
    if x.ndim in (3, 4, 5):
        if data_format == "channels_first":
            bias_reshape_axis = (1, bias_shape[-1]) + bias_shape[:-1]
            return x + tf_frontend.reshape(bias, bias_reshape_axis)
        return x + tf_frontend.reshape(bias, (1,) + bias_shape)
    return tf_frontend.nn.bias_add(x, bias)


@to_aikit_arrays_and_back
def dot(x, y):
    return aikit.dot(x, y)


@to_aikit_arrays_and_back
def rnn(
    step_function,
    inputs,
    initial_states,
    go_backwards=False,
    mask=None,
    constants=None,
    unroll=False,
    input_length=None,
    time_major=False,
    zero_output_for_mask=False,
    return_all_outputs=True,
):
    @functools.wraps(step_function)
    def _new_step_function(*args, **kwargs):
        frontend_args = aikit.nested_map(
            _aikit_array_to_tensorflow, args, include_derived=True, shallow=False
        )
        frontend_kwargs = aikit.nested_map(
            _aikit_array_to_tensorflow, kwargs, include_derived=True, shallow=False
        )
        ret = step_function(*frontend_args, **frontend_kwargs)
        return aikit.nested_map(_to_aikit_array, ret, include_derived=True)

    return aikit.rnn(
        _new_step_function,
        inputs,
        initial_states,
        go_backwards=go_backwards,
        mask=mask,
        constants=constants,
        unroll=unroll,
        input_length=input_length,
        time_major=time_major,
        zero_output_for_mask=zero_output_for_mask,
        return_all_outputs=return_all_outputs,
    )
