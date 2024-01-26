# local
import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def array_split(ary, indices_or_sections, axis=0):
    return aikit.split(
        ary, num_or_size_splits=indices_or_sections, axis=axis, with_remainder=True
    )


@to_aikit_arrays_and_back
def dsplit(ary, indices_or_sections):
    if isinstance(indices_or_sections, (list, tuple, aikit.Array)):
        indices_or_sections = (
            aikit.diff(indices_or_sections, prepend=[0], append=[ary.shape[2]])
            .astype(aikit.int8)
            .to_list()
        )
    return aikit.dsplit(ary, indices_or_sections)


@to_aikit_arrays_and_back
def hsplit(ary, indices_or_sections):
    if isinstance(indices_or_sections, (list, tuple, aikit.Array)):
        if ary.ndim == 1:
            indices_or_sections = (
                aikit.diff(indices_or_sections, prepend=[0], append=[ary.shape[0]])
                .astype(aikit.int8)
                .to_list()
            )
        else:
            indices_or_sections = (
                aikit.diff(indices_or_sections, prepend=[0], append=[ary.shape[1]])
                .astype(aikit.int8)
                .to_list()
            )
    return aikit.hsplit(ary, indices_or_sections)


@to_aikit_arrays_and_back
def split(ary, indices_or_sections, axis=0):
    if isinstance(indices_or_sections, (list, tuple, aikit.Array)):
        indices_or_sections = (
            aikit.diff(indices_or_sections, prepend=[0], append=[ary.shape[axis]])
            .astype(aikit.int8)
            .to_list()
        )
    return aikit.split(
        ary, num_or_size_splits=indices_or_sections, axis=axis, with_remainder=True
    )


@to_aikit_arrays_and_back
def vsplit(ary, indices_or_sections):
    if isinstance(indices_or_sections, (list, tuple, aikit.Array)):
        indices_or_sections = (
            aikit.diff(indices_or_sections, prepend=[0], append=[ary.shape[0]])
            .astype(aikit.int8)
            .to_list()
        )
    return aikit.vsplit(ary, indices_or_sections)
