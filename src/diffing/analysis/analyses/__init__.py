"""Plug-and-play analysis functions for ADLExplorer.

To add a new analysis: create a new file, import its function here,
and add it to ``all_analyses``.

Each key in ``all_analyses`` is a group name selectable via ``--analyses``.
The value is a dict of ``{output_name: callable}``.
"""

from .first_letter_a_n_proportion import (
    an_start_logit_lens_plot,
    an_start_logit_lens_table,
    an_start_patchscope_plot,
)
from .first_letter_f_proportion import (
    f_start_logit_lens_plot,
    f_start_logit_lens_table,
    f_start_patchscope_plot,
)

all_analyses: dict[str, dict[str, object]] = {
    "f-start": {
        "f-start_logit_lens_table": f_start_logit_lens_table,
        "f-start_logit_lens_plot": f_start_logit_lens_plot,
        "f-start_patchscope_plot": f_start_patchscope_plot,
    },
    "an-start": {
        "an-start_logit_lens_table": an_start_logit_lens_table,
        "an-start_logit_lens_plot": an_start_logit_lens_plot,
        "an-start_patchscope_plot": an_start_patchscope_plot,
    },
}

__all__ = [
    "all_analyses",
    "f_start_logit_lens_table",
    "f_start_logit_lens_plot",
    "f_start_patchscope_plot",
    "an_start_logit_lens_table",
    "an_start_logit_lens_plot",
    "an_start_patchscope_plot",
]
