# global
import abc
from typing import Optional, Tuple, Union, List, Sequence

# local
import aikit


# ToDo: implement all methods here as public instance methods

# ToDo: update docstrings and typehints according to aikit\layers


class _ArrayWithLayers(abc.ABC):
    def linear(
        self: aikit.Array,
        weight: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        bias: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.linear. This method simply
        wraps the function, and so the docstring for aikit.linear also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            The input array to compute linear transformation on.
            *[outer_batch_shape,inner_batch_shape,in_features]*
        weight
            The weight matrix. *[outer_batch_shape,out_features,in_features]*
        bias
            The bias vector, default is ``None``. *[outer_batch_shape,out_features]*
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Result array of the linear transformation.
            *[outer_batch_shape,inner_batch_shape,out_features]*

        Examples
        --------
        >>> x = aikit.array([[1.1, 2.2, 3.3], \
                           [4.4, 5.5, 6.6], \
                           [7.7, 8.8, 9.9]])
        >>> w = aikit.array([[1., 2., 3.], \
                           [4., 5., 6.], \
                           [7., 8., 9.]])
        >>> b = aikit.array([1., 0., -1.])
        >>> y = x.linear(w, bias=b)
        >>> print(y)
        aikit.array([[ 16.4,  35.2,  54. ],
                   [ 36.2,  84.7, 133. ],
                   [ 56. , 134. , 212. ]])
        """
        return aikit.linear(
            self._data,
            weight,
            bias=bias,
            out=out,
        )

    def dropout(
        self: aikit.Array,
        prob: float,
        /,
        *,
        scale: bool = True,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        training: bool = True,
        seed: Optional[int] = None,
        noise_shape: Optional[Sequence[int]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.dropout. This method simply
        wraps the function, and so the docstring for aikit.dropout also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            The input array x to perform dropout on.
        prob
            The probability of zeroing out each array element, float between 0 and 1.
        scale
            Whether to scale the output by `1/(1-prob)`, default is ``True``.
        dtype
            output array data type. If dtype is None, the output array data type
            must be inferred from x. Default: ``None``.
        training
            Turn on dropout if training, turn off otherwise. Default is ``True``.
        seed
            Set a default seed for random number generating (for
            reproducibility).Default is ``None``.
        noise_shape
            a sequence representing the shape of the binary dropout mask that will be
            multiplied with the input.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Result array of the output after dropout is performed.

        Examples
        --------
        With :class:`aikit.Array` instances:

        >>> x = aikit.array([[1., 2., 3.],
        ...                [4., 5., 6.],
        ...                [7., 8., 9.],
        ...                [10., 11., 12.]])
        >>> y = x.dropout(0.3)
        >>> print(y)
        aikit.array([[ 1.42857146,  2.85714293,  4.28571415],
                   [ 5.71428585,  7.14285755,  8.5714283 ],
                   [ 0.        , 11.4285717 , 12.8571434 ],
                   [14.2857151 ,  0.        ,  0.        ]])

        >>> x = aikit.array([[1., 2., 3.],
        ...                [4., 5., 6.],
        ...                [7., 8., 9.],
        ...                [10., 11., 12.]])
        >>> y = x.dropout(0.3, scale=False)
        >>> print(y)
        aikit.array([[ 1.,  2., 3.],
                   [ 4.,  5., 0.],
                   [ 7.,  0., 9.],
                   [10., 11., 0.]])
        """
        return aikit.dropout(
            self._data,
            prob,
            scale=scale,
            dtype=dtype,
            training=training,
            seed=seed,
            noise_shape=noise_shape,
            out=out,
        )

    def dropout1d(
        self: aikit.Array,
        prob: float,
        /,
        *,
        training: bool = True,
        data_format: str = "NWC",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.dropout1d. This method
        simply wraps the function, and so the docstring for aikit.droput1d also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input array x to perform dropout on.
        prob
            The probability of zeroing out each array element, float between 0 and 1.
        training
            Turn on dropout if training, turn off otherwise. Default is ``True``.
        data_format
            "NWC" or "NCW". Default is ``"NWC"``.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Result array of the output after dropout is performed.

        Examples
        --------
        >>> x = aikit.array([1, 1, 1]).reshape([1, 1, 3])
        >>> y = x.dropout1d(0.5)
        >>> print(y)
        aikit.array([[[2., 0, 2.]]])
        """
        return aikit.dropout1d(
            self._data,
            prob,
            training=training,
            data_format=data_format,
            out=out,
        )

    def dropout2d(
        self: aikit.Array,
        prob: float,
        /,
        *,
        training: bool = True,
        data_format: str = "NHWC",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.dropout2d. This method
        simply wraps the function, and so the docstring for aikit.droput1d also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input array x to perform dropout on.
        prob
            The probability of zeroing out each array element, float between 0 and 1.
        training
            Turn on dropout if training, turn off otherwise. Default is ``True``.
        data_format
            "NHWC" or "NCHW". Default is ``"NHWC"``.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Result array of the output after dropout is performed.

        Examples
        --------
        >>> x = aikit.array([[1, 1, 1], [2, 2, 2]])
        >>> y = x.dropout2d(0.5)
        >>> print(y)
        aikit.array([[0., 0., 2.],
               [4., 4., 4.]])
        """
        return aikit.dropout2d(
            self._data,
            prob,
            training=training,
            data_format=data_format,
            out=out,
        )

    def dropout3d(
        self: aikit.Array,
        prob: float,
        /,
        *,
        training: bool = True,
        data_format: str = "NDHWC",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.dropout3d. This method
        simply wraps the function, and so the docstring for aikit.droput3d also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input array x to perform dropout on.
        prob
            The probability of zeroing out each array element, float between 0 and 1.
        training
            Turn on dropout if training, turn off otherwise. Default is ``True``.
        data_format
            "NDHWC" or "NCDHW". Default is ``"NDHWC"``.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Result array of the output after dropout is performed.
        """
        return aikit.dropout3d(
            self._data,
            prob,
            training=training,
            data_format=data_format,
            out=out,
        )

    def scaled_dot_product_attention(
        self: aikit.Array,
        key: Union[aikit.Array, aikit.NativeArray],
        value: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        scale: Optional[float] = None,
        mask: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        dropout_p: Optional[float] = 0.0,
        is_causal: Optional[bool] = False,
        training: Optional[bool] = False,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of
        aikit.scaled_dot_product_attention. This method simply wraps the
        function, and so the docstring for aikit.scaled_dot_product_attention
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The queries input array. The shape of queries input array should be in
            *[batch_shape,num_queries,feat_dim]*. The queries input array should
            have the same size as keys and values.
        key
            The keys input array. The shape of keys input array should be in
            *[batch_shape,num_keys,feat_dim]*. The keys input array should have
            the same size as queries and values.
        value
            The values input array. The shape of values input should be in
            *[batch_shape,num_keys,feat_dim]*. The values input array should
            have the same size as queries and keys.
        scale
            The scale float value.
            The scale float value is used to scale the query-key pairs before softmax.
        mask
            The mask input array. The mask to apply to the query-key values.
            Default is None. The shape of mask input should be in
            *[batch_shape,num_queries,num_keys]*.
        dropout_p
            Specifies the dropout probability, if greater than 0.0, dropout is applied
        is_causal
            If true, assumes causal attention masking and errors if both `mask` and
            `is_causal` are set.
        training
            If True, dropout is used, otherwise dropout is not activated.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The output following application of scaled dot-product attention.
            The output array is the weighted sum produced by the attention score
            and value. The shape of output array is
            *[batch_shape,num_queries,feat_dim]* .

        Examples
        --------
        With :class:`aikit.Array` input:

        >>> q = aikit.array([[[0.2, 1.], [2.2, 3.],[4.4, 5.6]]])
        >>> k = aikit.array([[[0.6, 1.5], [2.4, 3.3],[4.2, 5.1]]])
        >>> v = aikit.array([[[0.4, 1.3], [2.2, 3.1],[4.3, 5.3]]])
        >>> result = aikit.scaled_dot_product_attention(q, k, v, scale=1, dropout_p=0.1,
        ...                                           is_causal=True, training=True)
        >>> print(result)
        aikit.array([[[0.40000001, 1.29999995],
                    [2.19994521, 3.09994531],
                    [4.30000019, 5.30000019]]])

        >>> q = aikit.array([[[0.2, 1.], [2.2, 3.],[4.4, 5.6]]])
        >>> k = aikit.array([[[0.6, 1.5], [2.4, 3.3],[4.2, 5.1]]])
        >>> v = aikit.array([[[0.4, 1.3], [2.2, 3.1],[4.3, 5.3]]])
        >>> mask = aikit.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]])
        >>> result = aikit.scaled_dot_product_attention(q,k,v,scale=1, mask=mask)
        >>> print(result)
        aikit.array([[[0.40000001, 1.29999995],
                    [2.19994521, 3.09994531],
                    [4.30000019, 5.30000019]]])

        >>> q = aikit.array([[[0.2, 1.], [2.2, 3.], [4.4, 5.6]]])
        >>> k = aikit.array([[[0.6, 1.5], [2.4, 3.3], [4.2, 5.1]]])
        >>> v = aikit.array([[[0.4, 1.3], [2.2, 3.1], [4.3, 5.3]]])
        >>> out = aikit.zeros(shape=(1, 3, 2))
        >>> aikit.scaled_dot_product_attention(q, k, v, scale=1, dropout_p=0.1,
        ...                                  is_causal=True, training=True, out=out)
        >>> print(out)
        aikit.array([[[0.40000001, 1.29999995],
                    [2.19994521, 3.09994531],
                    [4.30000019, 5.30000019]]])
        """
        return aikit.scaled_dot_product_attention(
            self._data,
            key,
            value,
            scale=scale,
            mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            training=training,
            out=out,
        )

    def multi_head_attention(
        self: aikit.Array,
        /,
        *,
        key: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        value: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        num_heads: int = 8,
        scale: Optional[float] = None,
        attention_mask: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        in_proj_weights: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        q_proj_weights: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        k_proj_weights: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        v_proj_weights: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        out_proj_weights: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        in_proj_bias: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        out_proj_bias: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        is_causal: bool = False,
        key_padding_mask: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        bias_k: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        bias_v: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        static_k: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        static_v: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        add_zero_attn: bool = False,
        return_attention_weights: bool = False,
        average_attention_weights: bool = True,
        dropout: float = 0.0,
        training: bool = False,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        return aikit.multi_head_attention(
            self._data,
            key=key,
            value=value,
            num_heads=num_heads,
            scale=scale,
            attention_mask=attention_mask,
            in_proj_weights=in_proj_weights,
            q_proj_weights=q_proj_weights,
            k_proj_weights=k_proj_weights,
            v_proj_weights=v_proj_weights,
            out_proj_weights=out_proj_weights,
            in_proj_bias=in_proj_bias,
            out_proj_bias=out_proj_bias,
            is_causal=is_causal,
            key_padding_mask=key_padding_mask,
            bias_k=bias_k,
            bias_v=bias_v,
            static_k=static_k,
            static_v=static_v,
            add_zero_attn=add_zero_attn,
            return_attention_weights=return_attention_weights,
            average_attention_weights=average_attention_weights,
            dropout=dropout,
            training=training,
            out=out,
        )

    def conv1d(
        self: aikit.Array,
        filters: Union[aikit.Array, aikit.NativeArray],
        strides: Union[int, Tuple[int]],
        padding: str,
        /,
        *,
        data_format: str = "NWC",
        filter_format: str = "channel_last",
        x_dilations: Union[int, Tuple[int]] = 1,
        dilations: Union[int, Tuple[int]] = 1,
        bias: Optional[aikit.Array] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.conv1d. This method simply
        wraps the function, and so the docstring for aikit.conv1d also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,w,d_in]* or *[batch_size,d_in,w]*.
        filters
            Convolution filters *[fw,d_in,d_out]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        data_format
            "NWC" or "NCW". Defaults to "NWC".
        filter_format
            Either "channel_first" or "channel_last". Defaults to "channel_last".
         x_dilations
            The dilation factor for each dimension of input. (Default value = 1)
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        bias
            Bias array of shape *[d_out]*.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

        Examples
        --------
        >>> x = aikit.array([[[1., 2.], [3., 4.], [6., 7.], [9., 11.]]])  # NWC
        >>> filters = aikit.array([[[0., 1.], [1., 1.]]])  # WIO (I == C)
        >>> result = x.conv1d(filters, (1,), 'VALID')
        >>> print(result)
        aikit.array([[[ 2.,  3.],
        ...         [ 4.,  7.],
        ...         [ 7., 13.],
        ...         [11., 20.]]])
        """
        return aikit.conv1d(
            self._data,
            filters,
            strides,
            padding,
            data_format=data_format,
            filter_format=filter_format,
            x_dilations=x_dilations,
            dilations=dilations,
            bias=bias,
            out=out,
        )

    def conv1d_transpose(
        self: aikit.Array,
        filters: Union[aikit.Array, aikit.NativeArray],
        strides: Union[int, Tuple[int]],
        padding: str,
        /,
        *,
        output_shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
        filter_format: str = "channel_last",
        data_format: str = "NWC",
        dilations: Union[int, Tuple[int]] = 1,
        bias: Optional[aikit.Array] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.conv1d_transpose. This
        method simply wraps the function, and so the docstring for
        aikit.conv1d_transpose also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,w,d_in]* or *[batch_size,d_in,w]*.
        filters
            Convolution filters *[fw,d_out,d_in]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            either the string ‘SAME’ (padding with zeros evenly), the string ‘VALID’ (no
            padding), or a sequence of n (low, high) integer pairs that give the padding
            to apply before and after each spatial dimension.
        output_shape
            Shape of the output (Default value = None)
        filter_format
            Either "channel_first" or "channel_last". "channel_first" corresponds
            to "IOW",input data formats, while "channel_last" corresponds to "WOI".
        data_format
            The ordering of the dimensions in the input, one of "NWC" or "NCW". "NWC"
            corresponds to input with shape (batch_size, width, channels), while "NCW"
            corresponds to input with shape (batch_size, channels, width).
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        bias
            Bias array of shape *[d_out]*.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the transpose convolution operation.

        Examples
        --------
        >>> x = aikit.array([[[1., 2.], [3., 4.], [6., 7.], [9., 11.]]])  # NWC
        >>> filters = aikit.array([[[0., 1.], [1., 1.]]])  # WIO (I == C)
        >>> result = x.conv1d_transpose(filters, (1,), 'VALID')
        >>> print(result)
        aikit.array([[[ 2.,  3.],
        ...         [ 4.,  7.],
        ...         [ 7., 13.],
        ...         [11., 20.]]])
        """
        return aikit.conv1d_transpose(
            self._data,
            filters,
            strides,
            padding,
            output_shape=output_shape,
            filter_format=filter_format,
            data_format=data_format,
            dilations=dilations,
            bias=bias,
            out=out,
        )

    def depthwise_conv2d(
        self: aikit.Array,
        filters: Union[aikit.Array, aikit.NativeArray],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[str, List[int]],
        /,
        *,
        data_format: str = "NHWC",
        dilations: Union[int, Tuple[int], Tuple[int, int]] = 1,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.depthwise_conv2d. This
        method simply wraps the function, and so the docstring for
        aikit.depthwise_conv2d also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,h,w,d]*.
        filters
            Convolution filters *[fh,fw,d_in]*. (d_in must be the same as d from self)
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

        Examples
        --------
        >>> x = aikit.randint(0, 255, shape=(1, 128, 128, 3)).astype(aikit.float32) / 255.0
        >>> filters = aikit.random_normal(mean=0, std=1, shape=[3, 3, 3])
        >>> y = x.depthwise_conv2d(filters, 2, 'SAME')
        >>> print(y.shape)
        (1, 64, 64, 3)
        """
        return aikit.depthwise_conv2d(
            self._data,
            filters,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
            out=out,
        )

    def conv2d(
        self: aikit.Array,
        filters: Union[aikit.Array, aikit.NativeArray],
        strides: Union[int, Tuple[int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NHWC",
        filter_format: str = "channel_last",
        x_dilations: Union[int, Tuple[int, int]] = 1,
        dilations: Union[int, Tuple[int, int]] = 1,
        bias: Optional[aikit.Container] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of `aikit.conv2d`. This method
        simply wraps the function, and so the docstring for `aikit.conv2d` also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,h,w,d_in]* or *[batch_size,d_in,h,w]*.
        filters
            Convolution filters *[fh,fw,d_in,d_out]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        filter_format
            Either "channel_first" or "channel_last". Defaults to "channel_last".
        x_dilations
            The dilation factor for each dimension of input. (Default value = 1)
        bias
            Bias array of shape *[d_out]*.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

        Examples
        --------
        >>> x = aikit.array([[[[1.], [2.0],[3.]],
        ...                 [[1.], [2.0],[3.]],
        ...                 [[1.], [2.0],[3.]]]]) #NHWC
        >>> filters = aikit.array([[[[0.]], [[1.]], [[0.]]],
        ...                      [[[0.]], [[1.]], [[0.]]],
        ...                      [[[0.]], [[1.]], [[0.]]]]) #HWIO
        >>> result = x.conv2d(filters, 1, 'SAME', data_format='NHWC',
        ...    dilations= 1)
        >>> print(result)
        aikit.array([[
                  [[2.],[4.],[6.]],
                  [[3.],[6.],[9.]],
                  [[2.],[4.],[6.]]
                  ]])
        """
        return aikit.conv2d(
            self,
            filters,
            strides,
            padding,
            data_format=data_format,
            filter_format=filter_format,
            x_dilations=x_dilations,
            dilations=dilations,
            bias=bias,
            out=out,
        )

    def conv2d_transpose(
        self: aikit.Array,
        filters: Union[aikit.Array, aikit.NativeArray],
        strides: Union[int, Tuple[int, int]],
        padding: str,
        /,
        *,
        output_shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
        filter_format: str = "channel_last",
        data_format: str = "NHWC",
        dilations: Union[int, Tuple[int, int]] = 1,
        out: Optional[aikit.Array] = None,
        bias: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of `aikit.conv2d_transpose`. This
        method simply wraps the function, and so the docstring for
        `aikit.conv2d_transpose` also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            Input image *[batch_size,h,w,d_in]* or *[batch_size,d_in,h,w]*.
        filters
            Convolution filters *[fh,fw,d_out,d_in]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        output_shape
            Shape of the output (Default value = None)
        filter_format
            Either "channel_first" or "channel_last". "channel_first" corresponds
            to "IOHW",input data formats, while "channel_last" corresponds to "HWOI".
        data_format
            The ordering of the dimensions in the input, one of "NHWC" or "NCHW". "NHWC"
            corresponds to inputs with shape (batch_size, height, width, channels),
            while "NCHW" corresponds to input with shape (batch_size, channels, height,
            width). Default is ``"NHWC"``.
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        bias
            Bias array of shape *[d_out]*.
        out
            Optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the transpose convolution operation.

        Examples
        --------
        >>> x = aikit.random_normal(mean=0, std=1, shape=[1, 28, 28, 3])
        >>> filters = aikit.random_normal(mean=0, std=1, shape=[3, 3, 6, 3])
        >>> y = x.conv2d_transpose(filters,2,'SAME',)
        >>> print(y.shape)
        (1, 56, 56, 6)
        """
        return aikit.conv2d_transpose(
            self._data,
            filters,
            strides,
            padding,
            output_shape=output_shape,
            filter_format=filter_format,
            data_format=data_format,
            dilations=dilations,
            out=out,
            bias=bias,
        )

    def conv3d(
        self: aikit.Array,
        filters: Union[aikit.Array, aikit.NativeArray],
        strides: Union[int, Tuple[int, int, int]],
        padding: str,
        /,
        *,
        data_format: str = "NDHWC",
        filter_format: str = "channel_last",
        x_dilations: Union[int, Tuple[int, int, int]] = 1,
        dilations: Union[int, Tuple[int, int, int]] = 1,
        bias: Optional[aikit.Array] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of `aikit.conv3d`. This method
        simply wraps the function, and so the docstring for `aikit.conv3d` also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input volume *[batch_size,d,h,w,d_in]*.
        filters
            Convolution filters *[fd,fh,fw,d_in,d_out]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        data_format
            "NDHWC" or "NCDHW". Defaults to "NDHWC".
        filter_format
            Either "channel_first" or "channel_last". Defaults to "channel_last".
        x_dilations
            The dilation factor for each dimension of input. (Default value = 1)
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        bias
            Bias array of shape *[d_out]*.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

        Examples
        --------
        >>> x = aikit.ones((1, 3, 3, 3, 1)).astype(aikit.float32)

        >>> filters = aikit.ones((1, 3, 3, 1, 1)).astype(aikit.float32)

        >>> result = x.conv3d(filters, 2, 'SAME')
        >>> print(result)
        aikit.array([[[[[4.],[4.]],[[4.],[4.]]],[[[4.],[4.]],[[4.],[4.]]]]])
        """
        return aikit.conv3d(
            self._data,
            filters,
            strides,
            padding,
            data_format=data_format,
            filter_format=filter_format,
            x_dilations=x_dilations,
            dilations=dilations,
            bias=bias,
            out=out,
        )

    def conv3d_transpose(
        self: aikit.Array,
        filters: Union[aikit.Array, aikit.NativeArray],
        strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
        padding: Union[str, List[int]],
        /,
        *,
        output_shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
        filter_format: str = "channel_last",
        data_format: str = "NDHWC",
        dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
        bias: Optional[aikit.Array] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of `aikit.conv3d_transpose`. This
        method simply wraps the function, and so the docstring for
        `aikit.conv3d_transpose` also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            Input volume *[batch_size,d,h,w,d_in]* or *[batch_size,d_in,d,h,w]*.
        filters
            Convolution filters *[fd,fh,fw,d_out,d_in]*.
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating
            the per-dimension paddings.
        output_shape
            Shape of the output (Default value = None)
        filter_format
            Either "channel_first" or "channel_last". "channel_first" corresponds
            to "IODHW",input data formats, while "channel_last" corresponds to "DHWOI".
        data_format
            The ordering of the dimensions in the input, one of "NDHWC" or
            "NCDHW". "NDHWC" corresponds to inputs with shape (batch_size,
             depth, height, width, channels), while "NCDHW" corresponds
             to input with shape (batch_size, channels, depth, height,
             width).
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        bias
            Bias array of shape *[d_out]*.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the transpose convolution operation.

        Examples
        --------
        >>> x = aikit.random_normal(mean=0, std=1, shape=[1, 3, 28, 28, 3])
        >>> filters = aikit.random_normal(mean=0, std=1, shape=[3, 3, 3, 6, 3])
        >>> y = x.conv3d_transpose(filters, 2, 'SAME')
        >>> print(y.shape)
        (1, 6, 56, 56, 6)
        """
        return aikit.conv3d_transpose(
            self._data,
            filters,
            strides,
            padding,
            output_shape=output_shape,
            filter_format=filter_format,
            data_format=data_format,
            dilations=dilations,
            bias=bias,
            out=out,
        )

    def lstm_update(
        self: aikit.Array,
        init_h: Union[aikit.Array, aikit.NativeArray],
        init_c: Union[aikit.Array, aikit.NativeArray],
        kernel: Union[aikit.Array, aikit.NativeArray],
        recurrent_kernel: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        bias: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        recurrent_bias: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
    ) -> Tuple[aikit.Array, aikit.Array]:
        """aikit.Array instance method variant of aikit.lstm_update. This method
        simply wraps the function, and so the docstring for aikit.lstm_update
        also applies to this method with minimal changes.

        Parameters
        ----------
        init_h
            initial state tensor for the cell output *[batch_shape, out]*.
        init_c
            initial state tensor for the cell hidden state *[batch_shape, out]*.
        kernel
            weights for cell kernel *[in, 4 x out]*.
        recurrent_kernel
            weights for cell recurrent kernel *[out, 4 x out]*.
        bias
            bias for cell kernel *[4 x out]*. (Default value = None)
        recurrent_bias
            bias for cell recurrent kernel *[4 x out]*. (Default value = None)

        Returns
        -------
        ret
            hidden state for all timesteps *[batch_shape,t,out]* and cell state for last
            timestep *[batch_shape,out]*

        Examples
        --------
        >>> x = aikit.randint(0, 20, shape=(6, 20, 3))
        >>> h_i = aikit.random_normal(shape=(6, 5))
        >>> c_i = aikit.random_normal(shape=(6, 5))
        >>> kernel = aikit.random_normal(shape=(3, 4 * 5))
        >>> rc = aikit.random_normal(shape=(5, 4 * 5))
        >>> result = x.lstm_update(h_i, c_i, kernel, rc)

        >>> result[0].shape
        (6, 20, 5)
        >>> result[1].shape
        (6, 5)
        """
        return aikit.lstm_update(
            self._data,
            init_h,
            init_c,
            kernel,
            recurrent_kernel,
            bias=bias,
            recurrent_bias=recurrent_bias,
        )
