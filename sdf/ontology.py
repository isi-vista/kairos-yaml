"""Loads ontology for use elsewhere."""

import json
from pathlib import Path
from typing import Any, Mapping, Optional

ONTOLOGY: Optional[Mapping[str, Any]] = None


def get_ontology() -> Mapping[str, Any]:
    """Loads the ontology from the JSON file.

    Returns:
        Ontology.
    """
    global ONTOLOGY  # pylint: disable=global-statement

    if ONTOLOGY is None:
        with Path("ontology.json").open() as file:
            ONTOLOGY = json.load(file)

    return ONTOLOGY


ontology = get_ontology()
