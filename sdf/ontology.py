"""Loads ontology for use elsewhere."""

import json
from pathlib import Path
from typing import Any, Mapping


def load_ontology() -> Mapping[str, Any]:
    """Loads the ontology from the JSON file.

    Returns:
        Ontology.
    """
    with Path("ontology.json").open() as file:
        ontology_json: Mapping[str, Any] = json.load(file)

    return ontology_json


ontology = load_ontology()
