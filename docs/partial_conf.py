discussion_channel_map = {
    "aikit.data_classes.array.array": ["933380487353872454"],
    "aikit.data_classes.container.container": ["982738042886422598"],
    "aikit.functional.aikit.activations": ["1000043490329251890"],
    "aikit.functional.aikit.compilation": ["1000043526849056808"],
    "aikit.functional.aikit.constants": ["1000043627961135224"],
    "aikit.functional.aikit.creation": ["1000043690254946374"],
    "aikit.functional.aikit.data_type": ["1000043749088436315"],
    "aikit.functional.aikit.device": ["1000043775021826229"],
    "aikit.functional.aikit.elementwise": ["1000043825085026394"],
    "aikit.functional.aikit.experimental": ["1028272402624434196"],
    "aikit.functional.aikit.extensions": ["1028272402624434196"],
    "aikit.functional.aikit.general": ["1000043859973247006"],
    "aikit.functional.aikit.gradients": ["1000043921633722509"],
    "aikit.functional.aikit.layers": ["1000043967989162005"],
    "aikit.functional.aikit.linear_algebra": ["1000044022942933112"],
    "aikit.functional.aikit.losses": ["1000044049333485648"],
    "aikit.functional.aikit.manipulation": ["1000044082489466951"],
    "aikit.functional.aikit.meta": ["1000044106959044659"],
    "aikit.functional.aikit.nest": ["1000044136000393326"],
    "aikit.functional.aikit.norms": ["1000044163070447626"],
    "aikit.functional.aikit.random": ["1000044191658815569"],
    "aikit.functional.aikit.searching": ["1000044227247484980"],
    "aikit.functional.aikit.set": ["1000044247606644786"],
    "aikit.functional.aikit.sorting": ["1000044274148184084"],
    "aikit.functional.aikit.statistical": ["1000044336479731872"],
    "aikit.functional.aikit.utility": ["1000044369044312164"],
    "aikit.stateful.activations": ["1000043360297439272"],
    "aikit.stateful.converters": ["1000043009758474310"],
    "aikit.stateful.initializers": ["1000043132706115654"],
    "aikit.stateful.layers": ["1000043206840426686"],
    "aikit.stateful.module": ["1000043315267387502"],
    "aikit.stateful.norms": ["1000043235802107936"],
    "aikit.stateful.optimizers": ["1000043277870964747"],
    "aikit.stateful.sequential": ["1000043078381473792"],
}

# Only generate docs for index.rst
# That resolved a bug of autosummary generating docs for code-block examples
# of autosummary
autosummary_generate = ["index.rst"]

skippable_method_attributes = [{"__qualname__": "_wrap_function.<locals>.new_function"}]

autosectionlabel_prefix_document = True

# Retrieve html_theme_options from docs/conf.py
from docs.conf import html_theme_options

html_theme_options["switcher"]["json_url"] = "https://khulnasoft.com/docs/versions/aikit.json"
html_sidebars = {"**": ["custom-toc-tree"]}

repo_name = "aikit"

# Retrieve demos specific configuration
from docs.demos.demos_conf import *  # noqa
