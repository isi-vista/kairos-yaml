"""Loads ontology for use elsewhere."""

from pathlib import Path
from typing import Mapping, Sequence

import lazy_object_proxy
from pydantic import BaseModel, Extra

ONTOLOGY_PATH = Path("ontology.json")


class InternalBase(BaseModel):
    """Base class for schema objects."""

    class Config:
        """Model configuration."""

        extra = Extra.forbid


class Arg(InternalBase):
    """Predicate argument.

    Attributes:
        position: Position of argument.
        label: Argument name.
        constraints: Allowed entity types for argument.
    """

    position: str
    label: str
    constraints: Sequence[str]


class Predicate(InternalBase):
    """KAIROS predicate (event or relation).

    Attributes:
        id: Predicate ID.
        type: Predicate name.
        definition: Definition of predicate.
        template: Template of example predicate usage.
        args: Mapping from argument names to arguments.
    """

    id: str
    type: str
    definition: str
    template: str
    args: Mapping[str, Arg]


class Entity(InternalBase):
    """KAIROS entity.

    Attributes:
        id: Entity ID.
        type: Entity name.
        definition: Definition of entity.
    """

    id: str
    type: str
    definition: str


class Ontology(InternalBase):
    """KAIROS ontology.

    Attributes:
        source_file: Name of source spreadsheet.
        events: Mapping from event names to events.
        entities: Mapping from entity names to entities.
        relations: Mapping from relation names to relations.
    """

    source_file: str
    events: Mapping[str, Predicate]
    entities: Mapping[str, Entity]
    relations: Mapping[str, Predicate]


def load_ontology() -> Ontology:
    """Loads the ontology from the JSON file.

    Returns:
        Ontology.
    """
    return Ontology.parse_file(ONTOLOGY_PATH)


# Lazy loading is needed to prevent trying to load the ontology
# when convert_ontology.py has not been run yet
ontology: Ontology = lazy_object_proxy.Proxy(load_ontology)
