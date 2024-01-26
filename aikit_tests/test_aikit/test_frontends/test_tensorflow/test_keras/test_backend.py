# global
from hypothesis import assume, strategies as st
from aikit.func_wrapper import output_to_native_arrays

# local
import aikit_tests.test_aikit.helpers as helpers
from aikit_tests.test_aikit.helpers import handle_frontend_test
from aikit_tests.test_aikit.test_functional.test_experimental.test_core.test_linalg import (
    _generate_dot_dtype_and_arrays,
)
from aikit_tests.test_aikit.test_frontends.test_tensorflow.test_nn import (
    _generate_bias_data,
)
from aikit_tests.test_aikit.test_functional.test_experimental.test_nn.test_layers import (
    _lstm_helper,
)
import aikit
from aikit.functional.frontends.tensorflow.func_wrapper import (
    inputs_to_aikit_arrays,
    outputs_to_frontend_arrays,
)
import aikit.functional.frontends.tensorflow as tf_frontend


@handle_frontend_test(
    fn_tree="tensorflow.keras.backend.dot",
    data=_generate_dot_dtype_and_arrays(min_num_dims=2),
)
def test_tensorflow_dot(*, data, on_device, fn_tree, frontend, test_flags, backend_fw):
    (input_dtypes, x) = data
    return helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        rtol=0.5,
        atol=0.5,
        x=x[0],
        y=x[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.keras.backend.bias_add",
    data=_generate_bias_data(keras_backend_fn=True),
    test_with_out=st.just(False),
)
def test_tensorflow_keras_backend_bias_add(
    *,
    data,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    data_format, dtype, x, bias = data
    helpers.test_frontend_function(
        input_dtypes=dtype * 2,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        bias=bias,
        data_format=data_format,
    )


@handle_frontend_test(
    fn_tree="tensorflow.keras.backend.rnn",
    rnn_args=_lstm_helper(),
    test_with_out=st.just(False),
)
def test_tensorflow_keras_backend_rnn(
    *,
    rnn_args,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    (
        input_dtypes,
        inputs,
        kernel_orig,
        recurrent_kernel_orig,
        bias_orig,
        recurrent_bias_orig,
        initial_states,
        go_backwards,
        mask,
        unroll,
        input_length,
        time_major,
        zero_output_for_mask,
        return_all_outputs,
    ) = rnn_args

    # unsupported dtype of float16 is in our _lstm_step function
    # so can't be inferred through aikit.function_unsupported_devices_and_dtypes
    assume(not (backend_fw == "torch" and input_dtypes[0] == "float16"))

    def _lstm_step(cell_inputs, cell_states):
        nonlocal kernel_orig, recurrent_kernel_orig, bias_orig, recurrent_bias_orig
        kernel = aikit.array(kernel_orig)
        recurrent_kernel = aikit.array(recurrent_kernel_orig)
        bias = aikit.array(bias_orig)
        recurrent_bias = aikit.array(recurrent_bias_orig)

        h_tm1 = cell_states[0]  # previous memory state
        c_tm1 = cell_states[1]  # previous carry state

        z = aikit.dot(cell_inputs, kernel) + bias
        z += aikit.dot(h_tm1, recurrent_kernel) + recurrent_bias

        z0, z1, z2, z3 = aikit.split(z, num_or_size_splits=4, axis=-1)

        i = aikit.sigmoid(z0)  # input
        f = aikit.sigmoid(z1)  # forget
        c = f * c_tm1 + i * aikit.tanh(z2)
        o = aikit.sigmoid(z3)  # output

        h = o * aikit.tanh(c)
        return h, [h, c]

    np_vals = [inputs, *initial_states, mask]

    if mask is None:
        np_vals.pop(-1)

    with aikit.utils.backend.ContextManager(backend_fw):
        _lstm_step_backend = outputs_to_frontend_arrays(
            inputs_to_aikit_arrays(_lstm_step)
        )
        vals = [aikit.array(val) for val in np_vals]
        if len(vals) > 3:
            inputs, init_h, init_c, mask = vals
        else:
            inputs, init_h, init_c = vals
        initial_states = [init_h, init_c]

        args = (_lstm_step_backend, inputs, initial_states)
        kwargs = {
            "go_backwards": go_backwards,
            "mask": mask,
            "constants": None,
            "unroll": unroll,
            "input_length": input_length,
            "time_major": time_major,
            "zero_output_for_mask": zero_output_for_mask,
            "return_all_outputs": return_all_outputs,
        }
        ret = tf_frontend.keras.backend.rnn(*args, **kwargs)
        aikit_ret = aikit.nested_map(lambda x: x.aikit_array, ret, shallow=False)
        aikit_idxs = aikit.nested_argwhere(aikit_ret, aikit.is_aikit_array)
        aikit_vals = aikit.multi_index_nest(aikit_ret, aikit_idxs)
        ret_np_flat = [x.to_numpy() for x in aikit_vals]

    with aikit.utils.backend.ContextManager(frontend):
        _lstm_step_gt = output_to_native_arrays(inputs_to_aikit_arrays(_lstm_step))
        import tensorflow as tf

        vals = [aikit.array(val).data for val in np_vals]
        if len(vals) > 3:
            inputs, init_h, init_c, mask = vals
        else:
            inputs, init_h, init_c = vals
        initial_states = [init_h, init_c]

        args = (_lstm_step_gt, inputs, initial_states)
        kwargs = {
            "go_backwards": go_backwards,
            "mask": mask,
            "constants": None,
            "unroll": unroll,
            "input_length": input_length,
            "time_major": time_major,
            "zero_output_for_mask": zero_output_for_mask,
            "return_all_outputs": return_all_outputs,
        }
        ret = tf.keras.backend.rnn(*args, **kwargs)
        native_idxs = aikit.nested_argwhere(ret, lambda x: isinstance(x, aikit.NativeArray))
        native_vals = aikit.multi_index_nest(ret, native_idxs)
        frontend_ret_np_flat = [x.numpy() for x in native_vals]

    helpers.value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=frontend_ret_np_flat,
        rtol=1e-1,
        atol=1e-1,
        backend=backend_fw,
        ground_truth_backend=frontend,
    )
