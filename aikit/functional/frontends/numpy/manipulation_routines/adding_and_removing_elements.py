# local
from collections import namedtuple
import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def append(arr, values, axis=None):
    if axis is None:
        return aikit.concat((aikit.flatten(arr), aikit.flatten(values)), axis=0)
    else:
        return aikit.concat((arr, values), axis=axis)


@to_aikit_arrays_and_back
def trim_zeros(filt, trim="fb"):
    first = 0
    trim = trim.upper()
    if "F" in trim:
        for i in filt:
            if i != 0.0:
                break
            else:
                first = first + 1
    last = aikit.shape(filt)[0]
    if "B" in trim:
        for i in filt[::-1]:
            if i != 0.0:
                break
            else:
                last = last - 1
    return filt[first:last]


@to_aikit_arrays_and_back
def unique(
    array, /, return_index=False, return_inverse=False, return_counts=False, axis=None
):
    results = aikit.unique_all(array, axis=axis)

    fields = ["values"]
    if return_index:
        fields.append("indices")
    if return_inverse:
        fields.append("inverse_indices")
    if return_counts:
        fields.append("counts")

    Results = namedtuple("Results", fields)

    values = [results.values]
    if return_index:
        values.append(results.indices)
    if return_inverse:
        values.append(results.inverse_indices)
    if return_counts:
        values.append(results.counts)

    return Results(*values)
