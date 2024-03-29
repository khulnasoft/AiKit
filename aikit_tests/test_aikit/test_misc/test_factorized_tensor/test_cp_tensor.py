import aikit

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("shape", "rank"),
    [
        (
            (3, 4, 5),
            4,
        )
    ],
)
def test_cp_flip_sign(shape, rank):
    cp_tensor = aikit.random_cp(shape, rank)
    weights, factors = aikit.CPTensor.cp_flip_sign(cp_tensor)

    assert aikit.all(aikit.mean(factors[1], axis=0) > 0)
    assert aikit.all(aikit.mean(factors[2], axis=0) > 0)
    assert cp_tensor.rank == cp_tensor.rank
    assert np.allclose(cp_tensor.weights, weights)
    assert np.allclose(
        aikit.CPTensor.cp_to_tensor((weights, factors)),
        aikit.CPTensor.cp_to_tensor(cp_tensor),
    )


@pytest.mark.parametrize(
    ("shape", "rank"),
    [
        (
            (8, 5, 6, 4),
            25,
        )
    ],
)
def test_cp_lstsq_grad(shape, rank):
    """Validate the gradient calculation between a CP and dense tensor."""
    cp_tensor = aikit.random_cp(shape, rank, normalise_factors=False)

    # If we're taking the gradient of comparison with self it should be 0
    cp_grad = aikit.CPTensor.cp_lstsq_grad(
        cp_tensor, aikit.CPTensor.cp_to_tensor(cp_tensor)
    )
    assert aikit.CPTensor.cp_norm(cp_grad) <= 10e-5

    # Check that we can solve for a direction of descent
    dense = aikit.random_cp(shape, rank, full=True, normalise_factors=False)
    cost_before = aikit.sqrt(
        aikit.sum(aikit.square(aikit.CPTensor.cp_to_tensor(cp_tensor) - dense))
    )

    cp_grad = aikit.CPTensor.cp_lstsq_grad(cp_tensor, dense)
    cp_new = aikit.CPTensor(cp_tensor)
    for ii in range(len(shape)):
        cp_new.factors[ii] = cp_tensor.factors[ii] - 1e-3 * cp_grad.factors[ii]

    cost_after = aikit.sqrt(
        aikit.sum(aikit.square(aikit.CPTensor.cp_to_tensor(cp_new) - dense))
    )
    assert cost_before > cost_after


@pytest.mark.parametrize(
    ("shape", "rank"),
    [
        (
            (5, 4, 6),
            3,
        )
    ],
)
def test_cp_mode_dot(shape, rank):
    cp_ten = aikit.random_cp(shape, rank, orthogonal=True, full=False)
    full_tensor = aikit.CPTensor.cp_to_tensor(cp_ten)
    # matrix for mode 1
    matrix = aikit.random_uniform(shape=(7, shape[1]))
    # vec for mode 2
    vec = aikit.random_uniform(shape=shape[2])

    # Test cp_mode_dot with matrix
    res = aikit.CPTensor.cp_mode_dot(cp_ten, matrix, mode=1, copy=True)
    # Note that if copy=True is not respected, factors will be changes
    # And the next test will fail
    res = aikit.CPTensor.cp_to_tensor(res)
    true_res = aikit.mode_dot(full_tensor, matrix, mode=1)
    assert np.allclose(true_res, res, atol=1e-3, rtol=1e-3)

    # Check that the data was indeed copied
    rec = aikit.CPTensor.cp_to_tensor(cp_ten)
    assert np.allclose(full_tensor, rec)

    # Test cp_mode_dot with vec
    res = aikit.CPTensor.cp_mode_dot(cp_ten, vec, mode=2, copy=True)
    res = aikit.CPTensor.cp_to_tensor(res)
    true_res = aikit.mode_dot(full_tensor, vec, mode=2)
    assert res.shape == true_res.shape
    assert np.allclose(true_res, res)


@pytest.mark.parametrize(
    ("shape", "rank", "tol"),
    [
        (
            (8, 5, 6, 4),
            25,
            10e-5,
        )
    ],
)
def test_cp_norm(shape, rank, tol):
    cp_tensor = aikit.random_cp(shape, rank, full=False, normalise_factors=True)
    rec = aikit.CPTensor.cp_to_tensor(cp_tensor)
    true_res = aikit.sqrt(aikit.sum(aikit.square(rec)))
    res = aikit.CPTensor.cp_norm(cp_tensor)
    assert aikit.abs(true_res - res) <= tol


# These tests have been adapetd from Tensorly
# https://github.com/tensorly/tensorly/blob/main/tensorly/tests/test_cp_tensor.py


@pytest.mark.parametrize(
    ("shape", "rank"),
    [
        (
            (3, 4, 5),
            4,
        )
    ],
)
def test_cp_normalize(shape, rank):
    cp_tensor = aikit.random_cp(shape, rank)
    weights, factors = aikit.CPTensor.cp_normalize(cp_tensor)
    expected_norm = aikit.ones((rank,))
    for f in factors:
        norm = aikit.sqrt(aikit.sum(aikit.square(f), axis=0))
        assert np.allclose(norm, expected_norm)
    assert np.allclose(
        aikit.CPTensor.cp_to_tensor((weights, factors)),
        aikit.CPTensor.cp_to_tensor(cp_tensor),
    )


@pytest.mark.parametrize(
    ("shapeU1", "shapeU2", "shapeU3", "shapeU4", "true_res", "columns", "rows"),
    [
        (
            (3, 3),
            (4, 3),
            (2, 3),
            (2, 3),
            [
                [
                    [[46754.0, 51524.0], [52748.0, 58130.0]],
                    [[59084.0, 65114.0], [66662.0, 73466.0]],
                    [[71414.0, 78704.0], [80576.0, 88802.0]],
                    [[83744.0, 92294.0], [94490.0, 104138.0]],
                ],
                [
                    [[113165.0, 124784.0], [127790.0, 140912.0]],
                    [[143522.0, 158264.0], [162080.0, 178730.0]],
                    [[173879.0, 191744.0], [196370.0, 216548.0]],
                    [[204236.0, 225224.0], [230660.0, 254366.0]],
                ],
                [
                    [[179576.0, 198044.0], [202832.0, 223694.0]],
                    [[227960.0, 251414.0], [257498.0, 283994.0]],
                    [[276344.0, 304784.0], [312164.0, 344294.0]],
                    [[324728.0, 358154.0], [366830.0, 404594.0]],
                ],
            ],
            4,
            (3, 4, 2),
        )
    ],
)
def test_cp_to_tensor(shapeU1, shapeU2, shapeU3, shapeU4, true_res, columns, rows):
    U1 = aikit.reshape(aikit.arange(1, 10, dtype=float), shapeU1)
    U2 = aikit.reshape(aikit.arange(10, 22, dtype=float), shapeU2)
    U3 = aikit.reshape(aikit.arange(22, 28, dtype=float), shapeU3)
    U4 = aikit.reshape(aikit.arange(28, 34, dtype=float), shapeU4)
    U = [aikit.array(t) for t in [U1, U2, U3, U4]]
    true_res = aikit.array(true_res)
    res = aikit.CPTensor.cp_to_tensor((aikit.ones(shape=(3,)), U))
    assert np.allclose(res, true_res)

    matrices = [
        aikit.arange(k * columns, dtype=float).reshape((k, columns)) for k in rows
    ]
    tensor = aikit.CPTensor.cp_to_tensor((aikit.ones(shape=(columns,)), matrices))
    for i in range(len(rows)):
        unfolded = aikit.unfold(tensor, mode=i)
        U_i = matrices.pop(i)
        reconstructed = aikit.matmul(
            U_i, aikit.permute_dims(aikit.khatri_rao(matrices), (1, 0))
        )
        assert np.allclose(reconstructed, unfolded)
        matrices.insert(i, U_i)


@pytest.mark.parametrize(("shape", "expected"), [((2, 2), [[-2, -2], [6, 10]])])
def test_cp_to_tensor_with_weights(shape, expected):
    A = aikit.reshape(aikit.arange(1, 5, dtype=float), shape)
    B = aikit.reshape(aikit.arange(5, 9, dtype=float), shape)
    weights = aikit.array([2, -1], dtype=A.dtype)

    out = aikit.CPTensor.cp_to_tensor((weights, [A, B]))
    expected = aikit.array(expected)  # computed by hand
    assert np.allclose(out, expected)

    (weights, factors) = aikit.random_cp((5, 5, 5), 5, normalise_factors=True, full=False)
    true_res = aikit.matmul(
        aikit.matmul(factors[0], aikit.diag(weights)),
        aikit.permute_dims(aikit.khatri_rao(factors[1:]), (1, 0)),
    )
    true_res = aikit.fold(true_res, 0, (5, 5, 5))
    res = aikit.CPTensor.cp_to_tensor((weights, factors))
    assert np.allclose(true_res, res)


@pytest.mark.parametrize(
    ("shapeU1", "shapeU2", "shapeU3", "shapeU4"), [((3, 3), (4, 3), (2, 3), (2, 3))]
)
def test_cp_to_unfolded(shapeU1, shapeU2, shapeU3, shapeU4):
    U1 = aikit.reshape(aikit.arange(1, 10, dtype=float), shapeU1)
    U2 = aikit.reshape(aikit.arange(10, 22, dtype=float), shapeU2)
    U3 = aikit.reshape(aikit.arange(22, 28, dtype=float), shapeU3)
    U4 = aikit.reshape(aikit.arange(28, 34, dtype=float), shapeU4)
    U = [aikit.array(t) for t in [U1, U2, U3, U4]]
    cp_tensor = aikit.CPTensor((aikit.ones((3,)), U))

    full_tensor = aikit.CPTensor.cp_to_tensor(cp_tensor)
    for mode in range(4):
        true_res = aikit.unfold(full_tensor, mode)
        res = aikit.CPTensor.cp_to_unfolded(cp_tensor, mode)
        assert np.allclose(
            true_res,
            res,
        )


@pytest.mark.parametrize(
    ("shapeU1", "shapeU2", "shapeU3", "shapeU4"), [((3, 3), (4, 3), (2, 3), (2, 3))]
)
def test_cp_to_vec(shapeU1, shapeU2, shapeU3, shapeU4):
    """Test for cp_to_vec."""
    U1 = np.reshape(np.arange(1, 10, dtype=float), shapeU1)
    U2 = np.reshape(np.arange(10, 22, dtype=float), shapeU2)
    U3 = np.reshape(np.arange(22, 28, dtype=float), shapeU3)
    U4 = np.reshape(np.arange(28, 34, dtype=float), shapeU4)
    U = [aikit.array(t) for t in [U1, U2, U3, U4]]
    cp_tensor = aikit.CPTensor(
        (
            aikit.ones(
                (3),
            ),
            U,
        )
    )
    full_tensor = aikit.CPTensor.cp_to_tensor(cp_tensor)
    true_res = aikit.reshape(full_tensor, (-1))
    res = aikit.CPTensor.cp_to_vec(cp_tensor)
    assert np.allclose(true_res, res)


@pytest.mark.parametrize(
    ("shape", "rank"),
    [
        (
            (10, 10, 10, 4),
            5,
        )
    ],
)
def test_unfolding_dot_khatri_rao(shape, rank):
    tensor = aikit.random_uniform(shape=shape)
    weights, factors = aikit.random_cp(shape, rank, full=False, normalise_factors=True)

    for mode in range(4):
        # Version forming explicitly the khatri-rao product
        unfolded = aikit.unfold(tensor, mode)
        kr_factors = aikit.khatri_rao(factors, weights=weights, skip_matrix=mode)
        true_res = aikit.matmul(unfolded, kr_factors)

        # Efficient sparse-safe version
        res = aikit.CPTensor.unfolding_dot_khatri_rao(tensor, (weights, factors), mode)
        assert np.allclose(true_res, res)


@pytest.mark.parametrize("size", [4])
def test_validate_cp_rank(size):
    tensor_shape = tuple(aikit.randint(1, 100, shape=(size,)))
    n_param_tensor = aikit.prod(tensor_shape)

    # Rounding = floor
    rank = aikit.CPTensor.validate_cp_rank(tensor_shape, rank="same", rounding="floor")
    n_param = aikit.CPTensor.cp_n_param(tensor_shape, rank)
    assert n_param <= n_param_tensor

    # Rounding = ceil
    rank = aikit.CPTensor.validate_cp_rank(tensor_shape, rank="same", rounding="ceil")
    n_param = aikit.CPTensor.cp_n_param(tensor_shape, rank)
    assert n_param >= n_param_tensor


@pytest.mark.parametrize(
    ("true_shape", "true_rank"),
    [
        (
            (3, 4, 5),
            3,
        )
    ],
)
def test_validate_cp_tensor(true_shape, true_rank):
    cp_tensor = aikit.random_cp(true_shape, true_rank)
    (weights, factors) = aikit.CPTensor.cp_normalize(cp_tensor)

    # Check correct rank and shapes are returned
    shape, rank = aikit.CPTensor.validate_cp_tensor((weights, factors))
    np.testing.assert_equal(
        true_shape,
        shape,
        err_msg=f"Returned incorrect shape (got {shape}, expected {true_shape})",
    )
    np.testing.assert_equal(
        rank,
        true_rank,
        err_msg=f"Returned incorrect rank (got {rank}, expected {true_rank})",
    )

    # One of the factors has the wrong rank
    factors[0], copy = aikit.random_uniform(shape=(4, 4)), factors[0]
    with np.testing.assert_raises(ValueError):
        aikit.CPTensor.validate_cp_tensor((weights, factors))

    # Not the correct amount of weights
    factors[0] = copy
    wrong_weights = weights[1:]
    with np.testing.assert_raises(ValueError):
        aikit.CPTensor.validate_cp_tensor((wrong_weights, factors))

    # Not enough factors
    with np.testing.assert_raises(ValueError):
        aikit.CPTensor.validate_cp_tensor((weights[:1], factors[:1]))
