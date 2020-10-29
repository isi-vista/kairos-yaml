"""Loads ontology for use elsewhere."""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import lazy_object_proxy
from pydantic import BaseModel, Extra, PrivateAttr

ONTOLOGY_PATH = Path(__file__).parent / "ontology.json"


class InternalBase(BaseModel):
    """Base class for schema objects."""

    class Config:
        """Model configuration."""

        allow_mutation = False
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
        full_type: Predicate name.
        type: Predicate type.
        subtype: Predicate subtype.
        subsubtype: Predicate subsubtype.
        definition: Definition of predicate.
        template: Template of example predicate usage.
        args: Mapping from argument names to arguments.
    """

    id: str
    full_type: str
    type: str
    subtype: str
    subsubtype: str
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

    # Private fields
    _event_types: Mapping[str, Mapping[str, Sequence[str]]] = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        """Instantiate ontology."""
        # TODO: Fix typing errors
        super().__init__(**data)  # type: ignore
        self._event_types = self._generate_event_tree()  # type: ignore

    def _generate_event_tree(self) -> Mapping[str, Mapping[str, Sequence[str]]]:
        """Generates tree of split event primitives for use by other methods.

        Returns:
            Tree of event types, subtypes, and subsubtypes.
        """
        tree: Dict[str, Dict[str, List[str]]] = {}
        event_triples = [(e.type, e.subtype, e.subsubtype) for e in self.events.values()]
        for event_type, subtype, subsubtype in event_triples:
            if event_type not in tree:
                tree[event_type] = {}
            if subtype not in tree[event_type]:
                tree[event_type][subtype] = []
            if subsubtype not in tree[event_type][subtype]:
                tree[event_type][subtype].append(subsubtype)
        return tree

    def get_event_subcats(self, event_type: str, subtype: Optional[str] = None) -> Sequence[str]:
        """Get subcategories for event types and subtypes.

        If only the type is given, allowed subtypes are returned. If the subtype is also given,
        allowed subsubtypes are returned.

        Args:
            event_type: Event type.
            subtype: Event subtype.

        Returns:
            List of subcategories if any exist, empty list otherwise.
        """
        if subtype is None:
            subtypes = self._event_types.get(event_type, None)
            if subtypes is None:
                return []
            else:
                return sorted(subtypes)

        subsubtypes = self._event_types[event_type].get(subtype, None)
        if subsubtypes is None:
            return []
        else:
            return sorted(subsubtypes)

    def get_default_event(self, partial_primitive: str) -> Optional[str]:
        """Converts primitive into a full, three-part primitive.

        Default subtypes and subsubtypes are selected if possible. If there is only a single
        subcategory, then that subcategory is selected. If "Unspecified" is a subcategory,
        "Unspecified" is selected. If there is no subcategory selectable by those heuristics,
        no default exists and None is returned instead. If the primitive is provided complete and
        in the ontology, then it is returned unchanged.

        Args:
            partial_primitive: Primitive for completion.

        Returns:
            Complete primitive if a default exists, None otherwise.
        """
        if partial_primitive in self.events:
            return partial_primitive

        primitive_segments = partial_primitive.split(".")

        if len(primitive_segments) == 1:
            subtypes = self.get_event_subcats(primitive_segments[0])
            if len(subtypes) == 1:
                primitive_segments.append(list(subtypes)[0])
            elif "Unspecified" in subtypes:
                primitive_segments.append("Unspecified")
            else:
                return None

        if len(primitive_segments) == 2:
            subsubtypes = self.get_event_subcats(primitive_segments[0], primitive_segments[1])
            if len(subsubtypes) == 1:
                primitive_segments.append(list(subsubtypes)[0])
            elif "Unspecified" in subsubtypes:
                primitive_segments.append("Unspecified")
            else:
                return None

        primitive = ".".join(primitive_segments)
        if primitive not in self.events:
            return None
        return primitive

    def get_event_by_id(self, event_index: int) -> Optional[str]:
        """Get primitive by its ID number.

        Args:
            event_index: ID of event.

        Returns:
            Primitive if ID is valid, None otherwise.
        """
        event_list = list(self.events)
        if 0 < event_index <= len(event_list):
            return event_list[event_index - 1]
        else:
            return None


def load_ontology() -> Ontology:
    """Loads the ontology from the JSON file.

    Returns:
        Ontology.
    """
    return Ontology.parse_file(ONTOLOGY_PATH)


# Lazy loading is needed to prevent trying to load the ontology
# when convert_ontology.py has not been run yet
ontology: Ontology = lazy_object_proxy.Proxy(load_ontology)
