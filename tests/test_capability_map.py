"""PACo's role sends the user to PAC's pages by their names: each must be a page of PAC's menu."""

import re
from pathlib import Path

import pytest

# The assistant is PAC's optional agent extra: without PACo, nothing to check.
prompts = pytest.importorskip("paco.prompts")

APP = Path(__file__).parents[1] / "frontend" / "src" / "App.tsx"


def test_every_page_the_assistant_names_is_in_pacs_menu() -> None:
    labels = set(re.findall(r'\blabel: "([^"]+)"', APP.read_text()))

    assert set(prompts.PAC_PAGES) <= labels, set(prompts.PAC_PAGES) - labels
