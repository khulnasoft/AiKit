# global
import abc
from typing import Optional, Union, Tuple, List, Sequence, Literal

# local
import aikit


class _ArrayWithLinearAlgebraExperimental(abc.ABC):
    def eigh_tridiagonal(
        self: Union[aikit.Array, aikit.NativeArray],
        beta: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        eigvals_only: bool = True,
        select: str = "a",
        select_range: Optional[
            Union[Tuple[int, int], List[int], aikit.Array, aikit.NativeArray]
        ] = None,
        tol: Optional[float] = None,
    ) -> Union[aikit.Array, Tuple[aikit.Array, aikit.Array]]:
        """aikit.Array instance method variant of aikit.eigh_tridiagonal. This
        method simply wraps the function, and so the docstring for
        aikit.eigh_tridiagonal also applies to this method with minimal changes.

        Parameters
        ----------
        self
            An array of real or complex arrays each of shape (n),
            the diagonal elements of the matrix.
        beta
            An array or of real or complex arrays each of shape (n-1),
            containing the elements of the first super-diagonal of the matrix.
        eigvals_only
            If False, both eigenvalues and corresponding eigenvectors are computed.
            If True, only eigenvalues are computed. Default is True.
        select
            Optional string with values in {'a', 'v', 'i'}
            (default is 'a') that determines which eigenvalues
            to calculate: 'a': all eigenvalues. 'v': eigenvalues
            in the interval (min, max] given by select_range.
            'i': eigenvalues with indices min <= i <= max.
        select_range
            Size 2 tuple or list or array specifying the range of
            eigenvalues to compute together with select. If select
            is 'a', select_range is ignored.
        tol
            Optional scalar. Ignored when backend is not Tensorflow. The
            absolute tolerance to which each eigenvalue is required. An
            eigenvalue (or cluster) is considered to have converged if
            it lies in an interval of this width. If tol is None (default),
            the value eps*|T|_2 is used where eps is the machine precision,
            and |T|_2 is the 2-norm of the matrix T.

        Returns
        -------
        eig_vals
            The eigenvalues of the matrix in non-decreasing order.
        eig_vectors
            If eigvals_only is False the eigenvectors are returned in the second
            output argument.

        Examples
        --------
        >>> alpha = aikit.array([0., 1., 2.])
        >>> beta = aikit.array([0., 1.])
        >>> y = alpha.eigh_tridiagonal(beta)
        >>> print(y)
        aikit.array([0., 0.38196, 2.61803])
        """
        return aikit.eigh_tridiagonal(
            self._data,
            beta,
            eigvals_only=eigvals_only,
            select=select,
            select_range=select_range,
            tol=tol,
        )

    def diagflat(
        self: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        offset: int = 0,
        padding_value: float = 0,
        align: str = "RIGHT_LEFT",
        num_rows: int = -1,
        num_cols: int = -1,
        out: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.diagflat. This method
        simply wraps the function, and so the docstring for aikit.diagflat also
        applies to this method with minimal changes.

        Examples
        --------
        >>> x = aikit.array([1,2])
        >>> x.diagflat(k=1)
        aikit.array([[0, 1, 0],
                   [0, 0, 2],
                   [0, 0, 0]])
        """
        return aikit.diagflat(
            self._data,
            offset=offset,
            padding_value=padding_value,
            align=align,
            num_rows=num_rows,
            num_cols=num_cols,
            out=out,
        )

    def kron(
        self: aikit.Array,
        b: aikit.Array,
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.kron. This method simply
        wraps the function, and so the docstring for aikit.kron also applies to
        this method with minimal changes.

        Examples
        --------
        >>> a = aikit.array([1,2])
        >>> b = aikit.array([3,4])
        >>> a.diagflat(b)
        aikit.array([3, 4, 6, 8])
        """
        return aikit.kron(self._data, b, out=out)

    def matrix_exp(self: aikit.Array, /, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.kron. This method simply
        wraps the function, and so the docstring for aikit.matrix_exp also
        applies to this method with minimal changes.

        Examples
        --------
        >>> x = aikit.array([[[1., 0.],
                            [0., 1.]],
                            [[2., 0.],
                            [0., 2.]]])
        >>> aikit.matrix_exp(x)
        aikit.array([[[2.7183, 1.0000],
                    [1.0000, 2.7183]],
                    [[7.3891, 1.0000],
                    [1.0000, 7.3891]]])
        """
        return aikit.matrix_exp(self._data, out=out)

    def eig(
        self: aikit.Array,
        /,
    ) -> Tuple[aikit.Array, ...]:
        """aikit.Array instance method variant of aikit.eig. This method simply
        wraps the function, and so the docstring for aikit.eig also applies to
        this method with minimal changes.

        Examples
        --------
        >>> x = aikit.array([[1,2], [3,4]])
        >>> x.eig()
        (
        aikit.array([-0.37228132+0.j,  5.37228132+0.j]),
        aikit.array([[-0.82456484+0.j, -0.41597356+0.j],
                   [ 0.56576746+0.j, -0.90937671+0.j]])
        )
        """
        return aikit.eig(self._data)

    def eigvals(
        self: aikit.Array,
        /,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.eigvals. This method simply
        wraps the function, and so the docstring for aikit.eigvals also applies
        to this method with minimal changes.

        Examples
        --------
        >>> x = aikit.array([[1,2], [3,4]])
        >>> x.eigvals()
        aikit.array([-0.37228132+0.j,  5.37228132+0.j])
        """
        return aikit.eigvals(self._data)

    def adjoint(
        self: aikit.Array,
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.adjoint. This method simply
        wraps the function, and so the docstring for aikit.adjoint also applies
        to this method with minimal changes.

        Examples
        --------
        >>> x = np.array([[1.-1.j, 2.+2.j],
                          [3.+3.j, 4.-4.j]])
        >>> x = aikit.array(x)
        >>> x.adjoint()
        aikit.array([[1.+1.j, 3.-3.j],
                   [2.-2.j, 4.+4.j]])
        """
        return aikit.adjoint(
            self._data,
            out=out,
        )

    def multi_dot(
        self: aikit.Array,
        x: Sequence[Union[aikit.Array, aikit.NativeArray]],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.multi_dot. This method
        simply wraps the function, and so the docstring for aikit.multi_dot also
        applies to this method with minimal changes.

        Examples
        --------
        >>> A = aikit.arange(2 * 3).reshape((2, 3))
        >>> B = aikit.arange(3 * 2).reshape((3, 2))
        >>> C = aikit.arange(2 * 2).reshape((2, 2))
        >>> A.multi_dot((B, C))
        aikit.array([[ 26,  49],
                   [ 80, 148]])
        """
        return aikit.multi_dot((self._data, *x), out=out)

    def cond(
        self: aikit.Array, /, *, p: Optional[Union[int, float, str]] = None
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.cond. This method simply
        wraps the function, and so the docstring for aikit.cond also applies to
        this method with minimal changes.

        Examples
        --------
        >>> x = aikit.array([[1,2], [3,4]])
        >>> x.cond()
        aikit.array(14.933034373659268)

        >>> x = aikit.array([[1,2], [3,4]])
        >>> x.cond(p=aikit.inf)
        aikit.array(21.0)
        """
        return aikit.cond(self._data, p=p)

    def mode_dot(
        self: Union[aikit.Array, aikit.NativeArray],
        /,
        matrix_or_vector: Union[aikit.Array, aikit.NativeArray],
        mode: int,
        transpose: Optional[bool] = False,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.mode_dot. This method
        simply wraps the function, and so the docstring for aikit.mode_dot also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            tensor of shape ``(i_1, ..., i_k, ..., i_N)``
        matrix_or_vector
            1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
            matrix or vectors to which to n-mode multiply the tensor
        mode
            int in the range(1, N)
        transpose
            If True, the matrix is transposed.
            For complex tensors, the conjugate transpose is used.
        out
            optional output array, for writing the result to.
            It must have a shape that the result can broadcast to.

        Returns
        -------
        aikit.Array
            `mode`-mode product of `tensor` by `matrix_or_vector`
            * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)`
            if matrix_or_vector is a matrix
            * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)`
            if matrix_or_vector is a vector
        """
        return aikit.mode_dot(self._data, matrix_or_vector, mode, transpose, out=out)

    def multi_mode_dot(
        self: Union[aikit.Array, aikit.NativeArray],
        mat_or_vec_list: Sequence[Union[aikit.Array, aikit.NativeArray]],
        /,
        modes: Optional[Sequence[int]] = None,
        skip: Optional[Sequence[int]] = None,
        transpose: Optional[bool] = False,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        r"""aikit.Array instance method variant of aikit.multi_mode_dot. This method
        simply wraps the function, and so the docstring for aikit.multi_mode_dot
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            the input tensor

        mat_or_vec_list
            sequence of matrices or vectors of length ``tensor.ndim``

        skip
            None or int, optional, default is None
            If not None, index of a matrix to skip.

        modes
            None or int list, optional, default is None

        transpose
            If True, the matrices or vectors in in the list are transposed.
            For complex tensors, the conjugate transpose is used.
        out
            optional output array, for writing the result to.
            It must have a shape that the result can broadcast to.

        Returns
        -------
        aikit.Array
            tensor times each matrix or vector in the list at mode `mode`

        Notes
        -----
        If no modes are specified, just assumes there is one matrix or vector per mode and returns:
        :math:`\\text{x  }\\times_0 \\text{ matrix or vec list[0] }\\times_1 \\cdots \\times_n \\text{ matrix or vec list[n] }`
        """  # noqa: E501
        return aikit.multi_mode_dot(
            self._data, mat_or_vec_list, modes, skip, transpose, out=out
        )

    def svd_flip(
        self: Union[aikit.Array, aikit.NativeArray],
        V: Union[aikit.Array, aikit.NativeArray],
        /,
        u_based_decision: Optional[bool] = True,
    ) -> Tuple[aikit.Array, aikit.Array]:
        """aikit.Array instance method variant of aikit.svd_flip. This method
        simply wraps the function, and so the docstring for aikit.svd_flip also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            left singular matrix output of SVD
        V
            right singular matrix output of SVD
        u_based_decision
            If True, use the columns of u as the basis for sign flipping.
            Otherwise, use the rows of v. The choice of which variable to base the
            decision on is generally algorithm dependent.

        Returns
        -------
        u_adjusted, v_adjusted : arrays with the same dimensions as the input.
        """
        return aikit.svd_flip(self._data, V, u_based_decision)

    def make_svd_non_negative(
        self: Union[aikit.Array, aikit.NativeArray],
        U: Union[aikit.Array, aikit.NativeArray],
        S: Union[aikit.Array, aikit.NativeArray],
        V: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        nntype: Optional[Literal["nndsvd", "nndsvda"]] = "nndsvd",
    ) -> Tuple[aikit.Array, aikit.Array]:
        """aikit.Array instance method variant of aikit.make_svd_non_negative. This
        method simply wraps the function, and so the docstring for
        aikit.make_svd_non_negative also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            tensor being decomposed.
        U
            left singular matrix from SVD.
        S
            diagonal matrix from SVD.
        V
            right singular matrix from SVD.
        nntype
            whether to fill small values with 0.0 (nndsvd),
            or the tensor mean (nndsvda, default).

        [1]: Boutsidis & Gallopoulos. Pattern Recognition, 41(4): 1350-1362, 2008.
        """
        return aikit.make_svd_non_negative(self._data, U, S, V, nntype=nntype)

    def tensor_train(
        self: Union[aikit.Array, aikit.NativeArray],
        rank: Union[int, Sequence[int]],
        /,
        svd: Optional[Literal["truncated_svd"]] = "truncated_svd",
        verbose: Optional[bool] = False,
    ) -> aikit.TTTensor:
        """aikit.Array instance method variant of aikit.tensor_train. This method
        simply wraps the function, and so the docstring for aikit.tensor_train
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input tensor
        rank
            maximum allowable TT rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor
        svd
            function to use to compute the SVD
        verbose
            level of verbosity

        Returns
        -------
        aikit.TTTensor
        """
        return aikit.tensor_train(self._data, rank, svd=svd, verbose=verbose)

    def truncated_svd(
        self: Union[aikit.Array, aikit.NativeArray],
        /,
        compute_uv: bool = True,
        n_eigenvecs: Optional[int] = None,
    ) -> Union[aikit.Array, Tuple[aikit.Array, aikit.Array, aikit.Array]]:
        """aikit.Array instance method variant of aikit.make_svd_non_negative. This
        method simply wraps the function, and so the docstring for
        aikit.make_svd_non_negative also applies to this method with minimal
        changes.

        Parameters
        ----------
        x
            2D-array
        compute_uv
            If ``True`` then left and right singular vectors will
            be computed and returnedv in ``U`` and ``Vh``
            respectively. Otherwise, only the singular values will
            be computed, which can be significantly faster.
        n_eigenvecs
            if specified, number of eigen[vectors-values] to return
            else full matrices will be returned

        Returns
        -------
        ret
            a namedtuple ``(U, S, Vh)``
            Each returned array must have the same floating-point data type as ``x``.
        """
        return aikit.truncated_svd(self._data, compute_uv, n_eigenvecs)

    def initialize_tucker(
        self: Union[aikit.Array, aikit.NativeArray],
        rank: Sequence[int],
        modes: Sequence[int],
        /,
        *,
        init: Optional[Union[Literal["svd", "random"], aikit.TuckerTensor]] = "svd",
        seed: Optional[int] = None,
        svd: Optional[Literal["truncated_svd"]] = "truncated_svd",
        non_negative: Optional[bool] = False,
        mask: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        svd_mask_repeats: Optional[int] = 5,
    ) -> Tuple[aikit.Array, Sequence[aikit.Array]]:
        """aikit.Array instance method variant of aikit.initialize_tucker. This
        method simply wraps the function, and so the docstring for
        aikit.initialize_tucker also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input tensor
        rank
            number of components
        modes
            modes to consider in the input tensor
        seed
            Used to create a random seed distribution
            when init == 'random'
        init
            initialization scheme for tucker decomposition.
        svd
            function to use to compute the SVD
        non_negative
            if True, non-negative factors are returned
        mask
            array of booleans with the same shape as ``tensor`` should be 0 where
            the values are missing and 1 everywhere else. Note:  if tensor is
            sparse, then mask should also be sparse with a fill value of 1 (or
            True).
        svd_mask_repeats
            number of iterations for imputing the values in the SVD matrix when
            mask is not None

        Returns
        -------
        core
            initialized core tensor
        factors
            list of factors
        """
        return aikit.initialize_tucker(
            self._data,
            rank,
            modes,
            seed=seed,
            init=init,
            svd=svd,
            non_negative=non_negative,
            mask=mask,
            svd_mask_repeats=svd_mask_repeats,
        )

    def partial_tucker(
        self: Union[aikit.Array, aikit.NativeArray],
        rank: Optional[Sequence[int]] = None,
        modes: Optional[Sequence[int]] = None,
        /,
        *,
        n_iter_max: Optional[int] = 100,
        init: Optional[Union[Literal["svd", "random"], aikit.TuckerTensor]] = "svd",
        svd: Optional[Literal["truncated_svd"]] = "truncated_svd",
        seed: Optional[int] = None,
        mask: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        svd_mask_repeats: Optional[int] = 5,
        tol: Optional[float] = 10e-5,
        verbose: Optional[bool] = False,
        return_errors: Optional[bool] = False,
    ) -> Tuple[aikit.Array, Sequence[aikit.Array]]:
        """aikit.Array instance method variant of aikit.partial_tucker. This method
        simply wraps the function, and so the docstring for aikit.partial_tucker
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            the  input tensor
        rank
            size of the core tensor, ``(len(ranks) == tensor.ndim)``
            if int, the same rank is used for all modes
            if None, original tensors size will be preserved.
        modes
            list of the modes on which to perform the decomposition
        n_iter_max
            maximum number of iteration
        init
            {'svd', 'random'}, or TuckerTensor optional
            if a TuckerTensor is provided, this is used for initialization
        svd
            str, default is 'truncated_svd'
            function to use to compute the SVD,
        seed
            Used to create a random seed distribution
            when init == 'random'
        mask
            array of booleans with the same shape as ``tensor`` should be 0 where
            the values are missing and 1 everywhere else. Note:  if tensor is
            sparse, then mask should also be sparse with a fill value of 1 (or
            True).
        svd_mask_repeats
            number of iterations for imputing the values in the SVD matrix when
            mask is not None
        tol
            tolerance: the algorithm stops when the variation in
            the reconstruction error is less than the tolerance.
        verbose
            if True, different in reconstruction errors are returned at each
            iteration.
        return_erros
            if True, list of reconstruction errors are returned.

        Returns
        -------
        core : ndarray
                core tensor of the Tucker decomposition
        factors : ndarray list
                list of factors of the Tucker decomposition.
                with ``core.shape[i] == (tensor.shape[i], ranks[i]) for i in modes``
        """
        return aikit.partial_tucker(
            self._data,
            rank,
            modes,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            seed=seed,
            mask=mask,
            svd_mask_repeats=svd_mask_repeats,
            tol=tol,
            verbose=verbose,
            return_errors=return_errors,
        )

    def tucker(
        self: Union[aikit.Array, aikit.NativeArray],
        rank: Optional[Sequence[int]] = None,
        /,
        *,
        fixed_factors: Optional[Sequence[int]] = None,
        n_iter_max: Optional[int] = 100,
        init: Optional[Union[Literal["svd", "random"], aikit.TuckerTensor]] = "svd",
        svd: Optional[Literal["truncated_svd"]] = "truncated_svd",
        seed: Optional[int] = None,
        mask: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        svd_mask_repeats: Optional[int] = 5,
        tol: Optional[float] = 10e-5,
        verbose: Optional[bool] = False,
        return_errors: Optional[bool] = False,
    ):
        """aikit.Array instance method variant of aikit.tucker. This method simply
        wraps the function, and so the docstring for aikit.tucker also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input tensor
        rank
            size of the core tensor, ``(len(ranks) == tensor.ndim)``
            if int, the same rank is used for all modes
        fixed_factors
            if not None, list of modes for which to keep the factors fixed.
            Only valid if a Tucker tensor is provided as init.
        n_iter_max
            maximum number of iteration
        init
            {'svd', 'random'}, or TuckerTensor optional
            if a TuckerTensor is provided, this is used for initialization
        svd
            str, default is 'truncated_svd'
            function to use to compute the SVD,
        seed
            Used to create a random seed distribution
            when init == 'random'
        mask
            array of booleans with the same shape as ``tensor`` should be 0 where
            the values are missing and 1 everywhere else. Note:  if tensor is
            sparse, then mask should also be sparse with a fill value of 1 (or
            True).
        svd_mask_repeats
            number of iterations for imputing the values in the SVD matrix when
            mask is not None
        tol
            tolerance: the algorithm stops when the variation in
            the reconstruction error is less than the tolerance
        verbose
            if True, different in reconstruction errors are returned at each
            iteration.

        return_errors
            Indicates whether the algorithm should return all reconstruction errors
            and computation time of each iteration or not
            Default: False


        Returns
        -------
            aikit.TuckerTensor or aikit.TuckerTensor and
            list of reconstruction errors if return_erros is True.

        References
        ----------
        .. [1] tl.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
        SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
        """
        return aikit.tucker(
            self._data,
            rank,
            fixed_factors=fixed_factors,
            n_iter_max=n_iter_max,
            init=init,
            return_errors=return_errors,
            seed=seed,
            mask=mask,
            svd=svd,
            svd_mask_repeats=svd_mask_repeats,
            tol=tol,
            verbose=verbose,
        )

    def tt_matrix_to_tensor(
        self: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """Aikit.Array instance method variant of aikit.tt_matrix_to_tensor. This
        method simply wraps the function, and so the docstring for
        aikit.tt_matrix_to_tensor also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
                array of 4D-arrays
                TT-Matrix factors (known as core) of shape
                (rank_k, left_dim_k, right_dim_k, rank_{k+1})

        out
            Optional output array. If provided, the output array to store the result.

        Returns
        -------
        output_tensor: array
                    tensor whose TT-Matrix decomposition was given by 'factors'
         --------
         >>> a = aikit.array([[[[[0.49671414],
         ...                      [-0.1382643]],
         ...
         ...                     [[0.64768857],
         ...                      [1.5230298]]]],
         ...                   [[[[-0.23415337],
         ...                      [-0.23413695]],
         ...
         ...                     [[1.57921278],
         ...                      [0.76743472]]]]])
         >>> a.tt_matrix_to_tensor()
         aikit.array([[[[-0.1163073 , -0.11629914],
          [ 0.03237505,  0.03237278]],

         [[ 0.78441733,  0.38119566],
          [-0.21834874, -0.10610882]]],


        [[[-0.15165846, -0.15164782],
          [-0.35662258, -0.35659757]],

         [[ 1.02283812,  0.49705869],
          [ 2.40518808,  1.16882598]]]])
        """
        return aikit.tt_matrix_to_tensor(self._data, out=out)

    def dot(
        self: Union[aikit.Array, aikit.NativeArray],
        b: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ):
        """Compute the dot product between two arrays `a` and `b` using the
        current backend's implementation. The dot product is defined as the sum
        of the element- wise product of the input arrays.

        Parameters
        ----------
        self
            First input array.
        b
            Second input array.
        out
            Optional output array. If provided, the output array to store the result.

        Returns
        -------
        ret
            The dot product of the input arrays.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> a = aikit.array([1, 2, 3])
        >>> b = aikit.array([4, 5, 6])
        >>> result = aikit.dot(a, b)
        >>> print(result)
        aikit.array(32)

        >>> a = aikit.array([[1, 2], [3, 4]])
        >>> b = aikit.array([[5, 6], [7, 8]])
        >>> c = aikit.empty_like(a)
        >>> aikit.dot(a, b, out=c)
        >>> print(c)
        aikit.array([[19, 22],
            [43, 50]])

        >>> a = aikit.array([[1.1, 2.3, -3.6]])
        >>> b = aikit.array([[-4.8], [5.2], [6.1]])
        >>> c = aikit.zeros((1, 1))
        >>> aikit.dot(a, b, out=c)
        >>> print(c)
        aikit.array([[-15.28]])
        """
        return aikit.dot(self._data, b, out=out)

    def general_inner_product(
        self: Union[aikit.Array, aikit.NativeArray],
        b: Union[aikit.Array, aikit.NativeArray],
        n_modes: Optional[int] = None,
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.general_inner_product. This
        method simply wraps the function, and so the docstring for
        aikit.general_inner_product also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            first input tensor.
        b
            second input tensor.
        n_modes
            int, default is None. If None, the traditional inner product is returned
            (i.e. a float) otherwise, the product between the `n_modes` last modes of
            `a` and the `n_modes` first modes of `b` is returned. The resulting tensor's
            order is `len(a) - n_modes`.
        out
            Optional output array. If provided, the output array to store the result.

        Returns
        -------
            The inner product of the input arrays.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> a = aikit.array([1, 2, 3])
        >>> b = aikit.array([4, 5, 6])
        >>> result = a.general_inner_product(b, n_modes=1)
        >>> print(result)
        aikit.array(32)

        >>> a = aikit.array([1, 2])
        >>> b = aikit.array([4, 5])
        >>> result = a.general_inner_product(b)
        >>> print(result)
        aikit.array(14)

        >>> a = aikit.array([[1, 1], [1, 1]])
        >>> b = aikit.array([[1, 2, 3, 4],[1, 1, 1, 1]])
        >>> result = a.general_inner_product(b, n_modes=1)
        >>> print(result)
        aikit.array([[2, 3, 4, 5],
            [2, 3, 4, 5]])
        """
        return aikit.general_inner_product(self, b, n_modes, out=out)

    def higher_order_moment(
        self: Union[aikit.Array, aikit.NativeArray],
        order: int,
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.higher_order_moment. This
        method simply wraps the function, and so the docstring for
        aikit.higher_order_moment also applies to this method with minimal
        changes.

        Parameters
        ----------
        x
            matrix of size (n_samples, n_features)
            or tensor of size(n_samples, D1, ..., DN)

        order
            number of the higher-order moment to compute

        Returns
        -------
        tensor
            if tensor is a matrix of size (n_samples, n_features),
            tensor of size (n_features, )*order

        Examples
        --------
        >>> a = aikit.array([[1, 2], [3, 4]])
        >>> result = aikit.higher_order_moment(a, 3)
        >>> print(result)
        aikit.array([[
            [14, 19],
            [19, 26]],
           [[19, 26],
            [26, 36]
        ]])
        """
        return aikit.higher_order_moment(self._data, order, out=out)

    def batched_outer(
        self: aikit.Array,
        tensors: Sequence[Union[aikit.Array, aikit.NativeArray]],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """Aikit Array instance method variant of aikit.batched_outer. This method
        simply wraps the function, and so the docstring for aikit.batched_outer
        also applies to this method with minimal changes.

        Parameters
        ----------
        tensors
            list of tensors of shape (n_samples, J1, ..., JN) ,
            (n_samples, K1, ..., KM) ...

        Returns
        -------
        outer product of tensors
            of shape (n_samples, J1, ..., JN, K1, ..., KM, ...)

        Examples
        --------
        >>> a = aikit.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> b = aikit.array([[[.1, .2], [.3, .4]], [[.5, .6], [.7, .8]]])
        >>> result = aikit.batched_outer(a, b)
        >>> print(result)
        aikit.array([[[[[0.1, 0.2],
              [0.30000001, 0.40000001]],
             [[0.2       , 0.40000001],
              [0.60000002, 0.80000001]]],
            [[[0.3       , 0.60000001],
              [0.90000004, 1.20000002]],
             [[0.40000001, 0.80000001],
              [1.20000005, 1.60000002]]]],
           [[[[2.5       , 3.00000012],
              [3.49999994, 4.00000006]],
             [[3.        , 3.60000014],
              [4.19999993, 4.80000007]]],
            [[[3.5       , 4.20000017],
              [4.89999992, 5.60000008]],
             [[4.        , 4.80000019],
              [5.5999999 , 6.4000001 ]]]]])
        """
        return aikit.batched_outer((self._data, *tensors), out=out)
