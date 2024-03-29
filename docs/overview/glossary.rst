Glossary
========

All of these new words can get confusing! We've created a glossary to help nail down some Aikit terms that you might find tricky.

.. glossary::
    :sorted:

    Pipeline
        A pipeline is a means of automating the machine learning workflow by enabling data to be transformed and correlated into a model that can then be analyzed to achieve outputs.

    Aikit Backends
        Aikit Backends are supported frameworks that Aikit can convert code to.
        The default is NumPy.

    Aikit Frontends
        Aikit Frontends are supported frameworks that Aikit can convert code from.

    Framework
        Frameworks are interfaces that allow scientists and developers to build and deploy machine learning models faster and easier.
        E.g. Tensorflow and PyTorch.

    Aikit Transpiler
        The transpiler allows framework to framework code conversions for supported frameworks.

    Aikit Container
        An Aikit class which inherits from :code:`dict` allows for storing nested data.

    Aikit Compiler
        A wrapper function around native compiler functions, which uses lower level compilers such as XLA to compile to lower level languages such as C++, CUDA, TorchScript, etc.

    Graph Compiler
        Graph Compilers map the high-level computational graph coming from frameworks to operations that are executable on a specific device.

    Aikit Tracer
        Aikit's Tracer creates a graph as a composition of functions in the functional API in Python.

    Aikit Functional API
        Is used for defining complex models, the Aikit functional API does not implement its own backend but wraps around other frameworks functional APIs and brings them into alignment.

    Framework Handler
    Backend Handler
        Used to control which framework Aikit is converting code to.

    Automatic Code Conversions
        Allows code to be converted from one framework to another whilst retaining its functional assets.

    Primary Functions
        Primary functions are the lowest level building blocks in Aikit and are generally implemented as light wrapping around an existing function in the backend framework, which serves a near-identical purpose.

    Compositional Functions
        Compositional functions are functions that are implemented as a composition of other Aikit functions,

    Mixed Functions
        Mixed functions are functions that have some backend-specific implementations but not for all backends.

    Standalone Functions
        Standalone functions are functions that do not reference any other primary, compositional, or mixed functions whatsoever.
        These are mainly convenience functions.

    Nestable Functions
        Nestable functions are functions that can accept :class:`aikit.Container` instances in place of any of the arguments.

    Convenience Functions
        Convenience functions can be used to organize and improve the code for other functions.

    Native Array
        The :class:`aikit.NativeArray` is simply a placeholder class for a backend-specific array class, such as :class:`np.ndarray`, :class:`tf.Tensor` or :class:`torch.Tensor`.

    Aikit Array
        The :class:`aikit.Array` is a simple wrapper class, which wraps around the :class:`aikit.NativeArray`.

    Submodule Helper Functions
        These are standalone/convenience functions that are specific to a submodule.
