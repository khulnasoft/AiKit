One liners
----------

.. grid:: 1 1 3 3
    :gutter: 4

    .. grid-item-card:: ``aikit.trace_graph()``
        :link: one_liners/trace.rst

        Traces a ``Callable`` or set of them into an Aikit graph.

    .. grid-item-card:: ``aikit.transpile()``
        :link: one_liners/transpile.rst

        Transpiles a ``Callable`` or set of them from a ``source`` framework to another
        framework.

    .. grid-item-card:: ``aikit.unify()``
        :link: one_liners/unify.rst

        Transpiles an object into Aikit code. It's an alias to
        ``aikit.transpile(..., to="aikit", ...)``

.. toctree::
    :hidden:
    :maxdepth: -1

    one_liners/trace.rst
    one_liners/transpile.rst
    one_liners/unify.rst
