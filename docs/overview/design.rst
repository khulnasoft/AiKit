Design
======

.. _`Deep Dive`: deep_dive.rst

This section is aimed at general users, who would like to learn how to use Aikit, and are less concerned about how it all works under the hood 🔧

The `Deep Dive`_ section is more targeted at potential contributors, and at users who would like to dive deeper into the weeds of the framework🌱, and gain a better understanding of what is actually going on behind the scenes 🎬

If that sounds like you, feel free to check out the `Deep Dive`_ section after you've gone through the higher level overview which is covered in this *design* section!

| So, starting off with our higher level *design* section, Aikit can fulfill two distinct purposes:
|
| 1. enable automatic code conversions between frameworks
| 2. serve as a new ML framework with multi-framework support
|
| The Aikit codebase can then be split into three categories which are labelled (a),
| (b) and (c) below, and can be further split into 8 distinct submodules.
| The eight submodules are Aikit API, Backend Handler, Backend API, Aikit Array,
| Aikit Container, Aikit Stateful API, and finally Frontend API.

| All eight fall into one of the three categories as follows:

.. image:: https://github.com/khulnasoft/khulnasoft.github.io/blob/main/img/externally_linked/design/submodule_dependency_graph.png?raw=true
   :align: center
   :width: 100%

| (a) `Building Blocks <design/building_blocks.rst>`_
| back-end functional APIs ✅
| Aikit functional API ✅
| Framework Handler ✅
| Aikit Tracer 🚧
|
| (b) `Aikit as a Transpiler <design/aikit_as_a_transpiler.rst>`_
| front-end functional APIs 🚧
|
| (c) `Aikit as a Framework <design/aikit_as_a_framework.rst>`_
| Aikit stateful API ✅
| Aikit Container ✅
| Aikit Array ✅

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Design

   design/building_blocks.rst
   design/aikit_as_a_transpiler.rst
   design/aikit_as_a_framework.rst
