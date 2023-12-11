.. title:: Home

.. include:: ../README.md
  :parser: myst_parser.sphinx_


.. toctree::
  :hidden:
  :maxdepth: -1

  Home <self>


.. toctree::
  :hidden:
  :maxdepth: -1
  :caption: The Basics

  overview/get_started.rst
  demos/quickstart.ipynb


.. toctree::
  :hidden:
  :maxdepth: -1
  :caption: Demos

  demos/learn_the_basics.rst
  demos/guides.rst
  demos/examples_and_demos.rst


.. toctree::
  :hidden:
  :maxdepth: -1
  :caption: Background

  overview/motivation.rst
  overview/related_work.rst


.. toctree::
  :hidden:
  :maxdepth: -1
  :caption: Contributors

  overview/design.rst
  overview/contributing.rst
  overview/deep_dive.rst
  overview/glossary.rst
  overview/faq.rst


.. toctree::
  :hidden:
  :maxdepth: -1
  :caption: API Reference

  overview/one_liners.rst


.. autosummary::
  :toctree: docs/functional
  :template: top_functional_toc.rst
  :recursive:
  :hide-table:

  aikit.functional.aikit


.. autosummary::
  :toctree: docs/data_classes
  :template: top_data_toc.rst
  :recursive:
  :hide-table:

  aikit.data_classes


.. autosummary::
  :toctree: docs
  :template: top_aikit_toc.rst
  :recursive:
  :hide-table:

  aikit.stateful
  aikit.utils
  aikit_tests.test_aikit.helpers
