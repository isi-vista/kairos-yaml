"""Specification of SDF schema format v1.1."""

# from datetime import datetime, timedelta
from typing import Any, Optional, Sequence, Tuple, TypeVar, Union

from pydantic import BaseModel, Extra, Field

T = TypeVar("T")
SingleOrSeq = Union[T, Sequence[T]]

# dateTime = datetime
DateTime = str
# duration = timedelta
Duration = str


class InternalBase(BaseModel):
    """Base class for schema objects."""

    comment: Optional[SingleOrSeq[str]]
    privateData: Optional[Any]

    class Config:
        """Model configuration."""

        extra = Extra.forbid


class Provenance(InternalBase):
    """Provenance for TA2."""

    boundingBox: Optional[Tuple[int, int, int, int]]
    childID: Optional[str]
    endTime: Optional[float]
    keyframes: Optional[Sequence[int]]
    length: Optional[int]
    mediaType: Optional[str]
    offset: Optional[int]
    parentIDs: Optional[SingleOrSeq[str]]
    provenance: Optional[SingleOrSeq[str]]
    startTime: Optional[float]


class Child(InternalBase):
    """Child of an event."""

    child: str
    importance: Optional[float]
    optional: Optional[bool]
    outlinks: Optional[SingleOrSeq[str]]


class Entity(InternalBase):
    """Entity usable in multiple contexts."""

    id: str = Field(alias="@id")
    confidence: Optional[float]
    modality: Optional[SingleOrSeq[str]]
    name: str
    provenance: Optional[SingleOrSeq[str]]
    qlabel: Optional[str]
    qnode: str
    reference: Optional[str]


class Participant(InternalBase):
    """Entity that participates in an event."""

    id: str = Field(alias="@id")
    aka: Optional[SingleOrSeq[str]]
    entity: str
    reference: Optional[str]
    roleName: str


class Relation(InternalBase):
    """Relation between entities and/or events."""

    id: str = Field(alias="@id")
    confidence: Optional[float]
    modality: Optional[SingleOrSeq[str]]
    name: Optional[str]
    provenance: Optional[SingleOrSeq[str]]
    reference: Optional[str]
    relationObject: str
    relationPredicate: str
    relationProvenance: SingleOrSeq[str]
    relationSubject: str
    TA1explanation: Optional[str]


class Temporal(InternalBase):
    """Temporal metadata for an event."""

    absoluteTime: Optional[DateTime]
    confidence: Optional[float]
    duration: Optional[Duration]
    earliestEndTime: Optional[DateTime]
    earliestStartTime: Optional[DateTime]
    latestEndTime: Optional[DateTime]
    latestStartTime: Optional[DateTime]
    provenance: Optional[SingleOrSeq[str]]


class Event(InternalBase):
    """Event, which can be an EC, schema, or subschema."""

    id: str = Field(alias="@id")
    aka: Optional[SingleOrSeq[str]]
    children: Optional[Sequence[Child]]
    confidence: Optional[float]
    description: Optional[str]
    goal: Optional[str]
    maxDuration: Optional[Duration]
    minDuration: Optional[Duration]
    modality: Optional[SingleOrSeq[str]]
    name: str
    outlink_gate: Optional[str]
    participants: Optional[Sequence[Participant]]
    provenance: Optional[SingleOrSeq[str]]
    qlabel: Optional[str]
    qnode: Optional[str]
    reference: Optional[str]
    relations: Optional[Sequence[Relation]]
    repeatable: Optional[bool]
    TA1explanation: Optional[str]
    template: Optional[str]
    temporal: Optional[SingleOrSeq[Temporal]]


class Library(InternalBase):
    """Entire SDF document."""

    context: Any = Field(alias="@context")
    id: str = Field(alias="@id")
    ceID: Optional[str]
    entities: Sequence[Entity]
    events: Sequence[Event]
    provenanceData: Optional[Sequence[Provenance]]
    relations: Optional[Sequence[Relation]]
    sdfVersion: str
    ta2: Optional[bool]
    task2: Optional[bool]
    version: str
