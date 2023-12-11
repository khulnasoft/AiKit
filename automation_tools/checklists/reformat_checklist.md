
## Reformatting Task Checklist
#### IMPORTANT NOTICE 🚨:
The [Aikit Docs](https://unify.ai/docs/aikit/) represent the ground truth for the task descriptions and this checklist should only be used as a supplementary item to aid with the review process.

#### LEGEND 🗺:
- ❌ :  Check item is not completed.
- ✅ :  Check item is ready for review.
- 🆘 :  Stuck/Doubting implementation (PR author should add comments explaining why).
- ⏩ :  Check is not applicable to function (skip).
- 🆗 :  Check item is already implemented and does not require any edits.

#### CHECKS 📑:
1. - [ ] ❌:  Remove all lambda and direct bindings for the backend functions in:
       - [ ] ❌: [aikit/functional/backends/jax/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/tree/main/aikit/functional/backends/jax/{{ .category_name }}.py).
       - [ ] ❌: [aikit/functional/backends/numpy/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/tree/main/aikit/functional/backends/numpy/{{ .category_name }}.py).
       - [ ] ❌: [aikit/functional/backends/tensorflow/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/tree/main/aikit/functional/backends/tensorflow/{{ .category_name }}.py).
       - [ ] ❌: [aikit/functional/backends/torch/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/tree/main/aikit/functional/backends/torch/{{ .category_name }}.py).
2. - [ ] ❌: Implement the following if they don't exist:
       1. - [ ]  ❌: The `aikit.Array` instance method in [aikit/data_classes/array/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/array/{{ .category_name }}.py).
       2. - [ ]  ❌: The `aikit.Array` special method in [aikit/data_classes/array/array.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/array/array.py).
       3. - [ ]  ❌: The `aikit.Array` reverse special method in [aikit/data_classes/array/array.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/array/array.py).
       4. - [ ] ❌: The `aikit.Container` static method in [aikit/data_classes/container/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/container/{{ .category_name }}.py).
       5. - [ ] ❌: The `aikit.Container` instance method in [aikit/data_classes/container/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/container/{{ .category_name }}.py).
       6. - [ ]  ❌:  The `aikit.Container` special method in [aikit/data_classes/container/container.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/container/container.py).
       7. - [ ]  ❌: The `aikit.Container` reverse special method in [aikit/data_classes/container/container.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/container/container.py).
3. - [ ] ❌:  Make sure that the aforementioned methods are added into the correct category-specific parent class, such as  `aikit.ArrayWithElementwise`,  `aikit.ContainerWithManipulation`  etc.
4. - [ ] ❌:  Correct all of the  [Function Arguments and the type hints](https://unify.ai/docs/aikit/overview/deep_dive/function_arguments.html#function-arguments) for every function  **and**  its  _relevant methods_, including those you did not implement yourself.
5. - [ ] ❌: Add the correct  [Docstrings](https://unify.ai/docs/aikit/overview/deep_dive/docstrings.html#docstrings)  to every function  **and**  its  _relevant methods_, including those you did not implement yourself. The following should be added:
       1. - [ ] ❌:   <a name="ref1"></a> The function's [Array API standard](https://data-apis.org/array-api/latest/index.html) description in [aikit/functional/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/blob/main/aikit/functional/aikit/{{ .category_name }}.py). If the function is not part of the Array API standard then a description of similar style should be added to the same file.
	The following modifications should be made to the description:
              - [ ] ❌:  Remove type definitions in the `Parameters` and `Returns` sections.
              - [ ] ❌:  Add `out` to the `Parameters` section if function accepts an `out` argument.
              - [ ] ❌:  Replace `out` with `ret` in the `Returns` section.
       2. - [ ] ❌:  Reference to docstring for aikit.function_name ([5.a](#ref1)) for the function description **and** modified `Parameters` and `Returns` sections as described in [the docs](https://unify.ai/docs/aikit/overview/deep_dive/docstrings.html#docstrings) in:
              - [ ] ❌:  [aikit/array/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/array/{{ .category_name }}.py).
              - [ ] ❌:  [aikit/container/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/container/{{ .category_name }}.py) (in the static and instance method versions).
              - [ ] ❌:   [aikit/array/array.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/array/array.py) if the function has a special method  ( like `__function_name__` ).
              - [ ] ❌:  [aikit/array/array.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/array/array.py) if the function has a reverse special method  ( like `__rfunction_name__` ).
              - [ ] ❌: [aikit/container/container.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/container/container.py) if the function has a special method ( like `__function_name__` ).
              - [ ] ❌:  [aikit/container/container.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/container/container.py) if the function has a reverse special method  ( like `__rfunction_name__` ).
6. - [ ] ❌: Add thorough  [Docstring Examples](https://unify.ai/docs/aikit/overview/deep_dive/docstring_examples.html#docstring-examples)  for every function  **and**  its  _relevant methods_  and ensure they pass the docstring tests.

		**Functional Examples** in [aikit/functional/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/blob/main/aikit/functional/aikit/{{ .category_name }}.py).

		1. - [ ] ❌: Cover all possible variants for each of the arguments independently (not combinatorily).
	 	2. - [ ] ❌: Vary the values and input shapes considerably between examples.
	 	3. - [ ] ❌: Start out simple and get more complex with each example.
	 	4. - [ ] ❌: Show an example with:
			   - [ ] ❌: `out` unused.
			   - [ ] ❌: `out` used to update a new array y.
			   - [ ] ❌: `out` used to inplace update the input array x (if x has the same dtype and shape as the return).
	 	5. - [ ] ❌: If broadcasting is relevant for the function, then show examples which highlight this.

		**Nestable Function Examples** in [aikit/functional/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/blob/main/aikit/functional/aikit/{{ .category_name }}.py).
		Only if the function supports nestable operations.

	 	6. - [ ] ❌: <a name="ref2"></a> Add an example that passes in an  `aikit.Container`  instance in place of one of the arguments.
	 	7. - [ ] ❌: <a name="ref3"></a> Add an example passes in  `aikit.Container`  instances for multiple arguments.

		**Container Static Method Examples** in [aikit/container/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/container/{{ .category_name }}.py).

	 	8. - [ ] ❌: The example from point ([6.f](#ref2)) should be replicated, but added to the  `aikit.Container`  **static method** docstring in with  `aikit.<func_name>`  replaced with  `aikit.Container.static_<func_name>`  in the example.
	 	9. - [ ] ❌: The example from point ([6.g](#ref3)) should be replicated, but added to the  `aikit.Container`  **static method** docstring, with  `aikit.<func_name>`  replaced with  `aikit.Container.static_<func_name>`  in the example.

		**Array Instance Method Example** in [aikit/array/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/array/{{ .category_name }}).

		10. - [ ] ❌: Call this instance method of the  `aikit.Array`  class.

		**Container Instance Method Example** in [aikit/container/{{ .category_name }}.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/container/{{ .category_name }}.py).

		11. - [ ] ❌: Call this instance method of the  `aikit.Container`  class.

		**Array Operator Examples** in [aikit/array/array.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/array/array.py).

		12. - [ ] ❌: Call the operator on two  `aikit.Array`  instances.
	 	13. - [ ] ❌: Call the operator with an  `aikit.Array`  instance on the left and  `aikit.Container`  on the right.

		**Array Reverse Operator Example** in [aikit/array/array.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/array/array.py).

		14.  - [ ] ❌: Call the operator with a  `Number`  on the left and an  `aikit.Array`  instance on the right.

		**Container Operator Examples** in [aikit/container/container.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/container/container.py).

		15. - [ ] ❌: Call the operator on two `aikit.Container` instances containing Number instances at the leaves.
	 	16. - [ ] ❌: Call the operator on two `aikit.Container` instances containing `aikit.Array` instances at the leaves.
	 	17. - [ ] ❌: Call the operator with an `aikit.Container` instance on the left and `aikit.Array` on the right.

		**Container Reverse Operator Example** in [aikit/container/container.py](https://github.com/khulnasoft/aikit/blob/main/aikit/data_classes/container/container.py).

		18. - [ ] ❌: Following example in the [`aikit.Container.__radd__`](https://github.com/khulnasoft/aikit/blob/e28a3cfd8a4527066d0d92d48a9e849c9f367a39/aikit/container/container.py#L173) docstring, with the operator called with a `Number` on the left and an `aikit.Container` instance on the right.

		**Tests**

		19. - [ ] ❌: Docstring examples tests passing.
		20. - [ ] ❌: Lint checks passing.
