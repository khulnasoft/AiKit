{% extends "top_level_toc_recursive.rst" %}

{% set aikit_module_map = {
    "aikit.stateful": "Framework classes",
    "aikit.nested_array": "Nested array",
    "aikit.utils": "Utils",
    "aikit_tests.test_aikit.helpers": "Testing",
} %}

{% block name %}{{aikit_module_map[fullname] | escape | underline}}{% endblock %}
