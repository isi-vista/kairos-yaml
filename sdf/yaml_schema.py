"""Specification of YAML schema format."""

from typing import Any, MutableMapping, Optional, Sequence, Union

from pydantic import BaseModel, Extra


class InternalBase(BaseModel):
    """Base class for schema objects.

    Attributes:
        private_data: Additional data that does not fit in any existing field.
    """

    private_data: Optional[MutableMapping[str, Any]]

    class Config:
        """Model configuration."""

        extra = Extra.forbid


class Slot(InternalBase):
    """Argument of schema or primitive.

    Attributes:
        role: Semantic role.
        refvar: Reference variable for entity.
        constraints: Course-grained entity types.
        reference: External fine-grained entity type.
        comment: Additional comments.
    """

    role: str
    refvar: Optional[str]
    constraints: Optional[Sequence[str]]
    reference: Optional[str]
    comment: Optional[str]


class Step(InternalBase):
    """Event step.

    Attributes:
        id: Event ID.
        primitive: Event type.
        reference: External fine-grained entity type.
        required: Whether step is required.
        comment: Additional comments.
        slots: Event arguments.
    """

    id: str
    primitive: str
    reference: Optional[str]
    required: Optional[bool]
    comment: Optional[str]
    slots: Sequence[Slot]


class Order(InternalBase):
    """Temporal order relation.

    Attributes:
        comment: Additional comments.
    """

    comment: Optional[str]


class Before(Order):
    """Precedence relation.

    Attributes:
        before: Earlier event.
        after: Later event.
    """

    before: str
    after: str


class Container(Order):
    """Container relation.

    Attributes:
        container: Containing event.
        contained: Contained event.
    """

    container: str
    contained: str


class Overlaps(Order):
    """Overlap relation.

    Attributes:
        overlaps: Overlapping relations.
    """

    overlaps: Sequence[str]


class Schema(InternalBase):
    """Schema for a complex event.

    Attributes:
        schema_id: Schema ID.
        schema_name: Human-readable label.
        schema_dscpt: Schema description.
        schema_version: Version of schema.
        slots: Schema arguments.
        steps: Event steps.
        order: Order of events.
        comment: Additional comments.
    """

    schema_id: str
    schema_name: str
    schema_dscpt: str
    schema_version: str
    slots: Sequence[Slot]
    steps: Sequence[Step]
    order: Sequence[Union[Before, Container, Overlaps]]
    comment: Optional[str]
