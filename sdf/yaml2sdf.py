"""Converts CMU YAML into KAIROS SDF JSON-LD."""

import argparse
from collections import defaultdict
from copy import deepcopy
import itertools
import json
import logging
from pathlib import Path
import re
import typing
from typing import (
    Any,
    List,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pydantic import parse_obj_as
import requests
import yaml

from sdf.ontology import ontology
from sdf.sdf_schema import Child, Entity, Event, Library, Participant, SingleOrSeq
from sdf.yaml_schema import Before, Schema, Slot, Step

VALIDATOR_ENDPOINTS = {
    "remote": "http://validation.kairos.nextcentury.com/json-ld/ksf/validate",
    "local": "http://localhost:8008/json-ld/ksf/validate",
}


def replace_whitespace(name: str) -> str:
    """Replaces whitespace with hyphens.

    Args:
        name: String to process.

    Returns:
        String with whitespace replaced.
    """
    return re.sub(r"\s+", "-", name)


def get_step_type(step: Step) -> str:
    """Gets type of step.

    Args:
        step: Step data.

    Returns:
        Step type.
    """
    primitive = ontology.get_default_event(step.primitive)

    if primitive not in ontology.events:
        logging.warning("Primitive '%s' in step '%s' not in ontology", step.primitive, step.id)

    return f"kairos:Primitives/Events/{primitive}"


def get_slot_role(slot: Slot, step_type: Optional[str], step_id: str) -> str:
    """Gets slot role.

    Args:
        slot: Slot data.
        step_type: Type of step.
        step_id: ID of step.

    Returns:
        Slot role.
    """
    if step_type is not None:
        event_type = ontology.events.get(step_type, None)
        if event_type is not None and slot.role not in event_type.args:
            logging.warning(
                "Role '%s' is not valid for event '%s'", slot.role, event_type.full_type
            )

    return f"{step_id}/Slots/{slot.role}"


def get_slot_name(slot: Slot, slot_shared: bool) -> str:
    """Gets slot name.

    Args:
        slot: Slot data.
        slot_shared: Whether slot is shared.

    Returns:
        Slot name.
    """
    name = slot.role
    uppercase_indices = [index for index, char in enumerate(name) if char.isupper()]
    if len(uppercase_indices) > 1:
        name = name[: uppercase_indices[1]]
    name = name.lower()
    if slot_shared and slot.refvar is not None:
        name += "-" + slot.refvar
    return replace_whitespace(name)


def get_slot_id(
    slot: Slot, schema_slot_counter: typing.Counter[str], parent_id: str, slot_shared: bool
) -> str:
    """Gets slot ID.

    Args:
        slot: Slot data.
        schema_slot_counter: Slot counter.
        parent_id: Parent object ID.
        slot_shared: Whether slot is shared.

    Returns:
        Slot ID.
    """
    slot_name = get_slot_name(slot, slot_shared)
    slot_id = chr(schema_slot_counter[slot_name] + 97)
    schema_slot_counter[slot_name] += 1
    return f"{parent_id}/Slots/{slot_name}-{slot_id}"


def get_slot_constraints(constraints: Sequence[str]) -> Sequence[str]:
    """Gets slot constraints.

    Args:
        constraints: Constraints.

    Returns:
        Slot constraints.
    """
    for entity in constraints:
        if entity not in ontology.entities and entity != "EVENT":
            logging.warning("Entity '%s' not in ontology", entity)

    return [f"kairos:Primitives/Entities/{entity}" for entity in constraints]


def get_step_id(step: Step, schema_id: str) -> str:
    """Gets step ID.

    Args:
        step: Step data.
        schema_id: Schema ID.

    Returns:
        Step ID.
    """
    return f"{schema_id}/Steps/{replace_whitespace(step.id)}"


def create_orders(yaml_data: Schema, step_map: Mapping[str, str]) -> Sequence[Mapping[str, Any]]:
    """Gets orders.

    Args:
        yaml_data: Data from YAML file.
        step_map: Mapping of steps from YAML IDs to SDF IDs.

    Returns:
        Orders in SDF format.
    """
    step_ids = set(step.id for step in yaml_data.steps)
    order_tuples: List[Tuple[str, ...]] = []
    for order in yaml_data.order:
        if isinstance(order, Before):
            order_tuples.append((order.before, order.after))
        else:
            raise NotImplementedError
    order_ids = set(itertools.chain.from_iterable(order_tuples))
    missing_order_ids = order_ids - step_ids
    if missing_order_ids:
        for missing_id in sorted(missing_order_ids):
            logging.error("The ID '%s' in `order` is not in `steps`", missing_id)
        exit(1)

    orders = []
    for order in yaml_data.order:
        if isinstance(order, Before):
            before_id = step_map[order.before]
            after_id = step_map[order.after]
            cur_order: MutableMapping[str, Union[str, Sequence[str]]] = {
                "before": before_id,
                "after": after_id,
            }
        else:
            raise NotImplementedError
        if order.comment is not None:
            cur_order["comment"] = order.comment
        orders.append(cur_order)
    return orders


def convert_yaml_to_sdf(
    yaml_data: Schema, performer_prefix: str
) -> Tuple[Sequence[Event], Sequence[Entity]]:
    """Converts YAML to SDF.

    Args:
        yaml_data: Data from YAML file.
        performer_prefix: Performer prefix for context.

    Returns:
        Schema in SDF format.
    """
    yaml_data = deepcopy(yaml_data)

    schema_id = f"{performer_prefix}:Schemas/{yaml_data.schema_id}"

    # Remove any steps without a primitive
    # ignored_steps = set(step.id for step in yaml_data.steps if step.primitive == "NotInOntology")
    # yaml_data.steps = [step for step in yaml_data.steps if step.primitive != "NotInOntology"]

    # Get comments
    comments = [x.id.replace("-", " ") for x in yaml_data.steps]
    comments = ["Steps:"] + [f"{idx}. {text}" for idx, text in enumerate(comments, start=1)]
    schema_comment = comments
    if yaml_data.comment is not None:
        schema_comment.append(yaml_data.comment)

    # Get steps
    events = []
    children = []

    # For order
    step_map: MutableMapping[str, str] = {}

    refvars = defaultdict(list)

    for idx, step in enumerate(yaml_data.steps):
        cur_step_id = get_step_id(step, schema_id)
        cur_step_comment: SingleOrSeq[str] = comments[idx + 1]
        event = Event(
            **{"@id": cur_step_id},  # type: ignore[arg-type]
            description="N/a",  # TODO: Remove if confirmed not required
            name=step.id,
            participants=None,
            qlabel=None,  # TODO: Fill with KGTK query
            qnode=step.reference,
            TA1explanation=None,  # TODO: Fill once extractable from YAML
        )
        events.append(event)
        child = Child(
            child=cur_step_id, optional=True if step.required is False else None, outlinks=[]
        )
        children.append(child)
        if step.comment is not None:
            cur_step_comment = [cur_step_comment, step.comment]  # type: ignore[list-item]

        step_map[step.id] = cur_step_id

        participants = []
        for i, slot in enumerate(step.slots):
            if slot.refvar is None:
                raise RuntimeError(f"{slot} misses refvar")
            refvars[replace_whitespace(slot.refvar)].append(slot.reference)
            primitive = ontology.get_default_event(step.primitive)
            if primitive:
                role = f"A{ontology.events[primitive].args[slot.role].position[-1]}"
            else:
                role = "A?"
            participants.append(
                Participant(
                    **{"@id": cur_step_id + f"/Participant/{i}"},
                    entity=slot.refvar if slot.refvar else "",  # TODO: Make refvar required in YAML
                    roleName=role,
                )
            )

        event.participants = participants

    participants = []
    for i, slot in enumerate(yaml_data.slots):
        if slot.refvar is None:
            raise RuntimeError(f"{slot} misses refvar")
        refvars[replace_whitespace(slot.refvar)].append(slot.reference)

        participants.append(
            Participant(
                **{"@id": schema_id + f"/Participant/{i}"},
                entity=slot.refvar if slot.refvar else "",  # TODO: Make refvar required in YAML
                roleName="A?",
            )
        )

    entities = []
    for refvar, qnodes in refvars.items():
        # TODO: Check for consistency across usages, and make qnode required
        qnode = (
            f"wiki:{qnodes[0]}" if qnodes and qnodes[0] is not None else "Q355120"
        )  # Qnode for entity
        entity = Entity(
            **{"@id": schema_id + f"/{refvar}"},  # type: ignore[arg-type]  # TODO: Figure out how to use aliases properly
            name=refvar,
            qlabel=None,  # TODO: Fill with KGTK query
            qnode=qnode,
        )
        entities.append(entity)

    # Cleaning "-a" suffix for slots with counter == 1.
    # for cur_step in steps:
    #     for cur_slot in cur_step["participants"]:
    #         if schema_slot_counter[cur_slot["name"]] == 1:
    #             cur_slot["@id"] = cur_slot["@id"].strip("-a")

    schema_order = create_orders(yaml_data, step_map)

    child_dict = {c.child: c for c in children}
    for order in schema_order:
        child = child_dict[order["before"]]
        child.outlinks = list(child.outlinks if child.outlinks else []) + [order["after"]]

    event = Event(
        **{"@id": schema_id},  # type: ignore[arg-type]
        children=children,
        description="N/a",  # TODO: Remove if confirmed not required
        name=schema_id,
        participants=None,
        qlabel=None,  # TODO: Fill with KGTK query
        qnode=None,
        TA1explanation=None,  # TODO: Fill once extractable from YAML
    )
    events.append(event)

    return events, entities


def merge_schemas(
    events: Sequence[Event],
    entities: Sequence[Entity],
    latest_version: str,
    performer_prefix: str,
    performer_uri: str,
    library_id: str,
) -> Mapping[str, Any]:
    """Merge multiple schemas.

    Args:
        schema_list: List of SDF schemas.
        performer_prefix: Performer prefix for context.
        performer_uri: Performer URI for context.
        library_id: ID of schema collection.

    Returns:
        Data in JSON output format.
    """
    sdf = Library(
        **{  # type: ignore[arg-type]
            "@context": [
                "https://kairos-sdf.s3.amazonaws.com/context/kairos-v1.1.jsonld",
                {performer_prefix: performer_uri},
            ],
            "@id": f"{performer_prefix}:Submissions/TA1/{library_id}",
        },
        comment=(
            "This file was generated using a very rudimentary implementation of SDF v1.1, "
            "so it does not look good and likely contains errors."
        ),
        entities=entities,
        events=events,
        sdfVersion="1.1",
        version=f"{performer_prefix}_{latest_version}",
    )

    return sdf.dict(by_alias=True, exclude_none=True)


def validate_schemas(json_data: Mapping[str, Any], validator_endpoint: str) -> None:
    """Validates generated schema against the program validator.

    The program validator is not always available, so the request will time out if no response is
    received within 10 seconds.

    Args:
        json_data: Data in JSON output format.
        validator_endpoint: URL for accessing validator.
    """
    try:
        req = requests.post(
            validator_endpoint,
            json=json_data,
            headers={"Accept": "application/json", "Content-Type": "application/ld+json"},
            timeout=10,
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        logging.warning("Program validator is unavailable, so schema might not validate")
    else:
        response = req.json()
        validator_messages = response["errorsList"] + response["warningsList"]
        if validator_messages:
            print("Messages from program validator:")
            for message in validator_messages:
                print(f"\t{message}")


def convert_all_yaml_to_sdf(
    yaml_schemas: Sequence[Mapping[str, Any]],
    performer_prefix: str,
    performer_uri: str,
    library_id: str,
    validator: Optional[str],
) -> Mapping[str, Any]:
    """Convert YAML schema library into SDF schema library.

    Args:
        yaml_schemas: YAML schemas.
        performer_prefix: Performer prefix for context.
        performer_uri: Performer URI for context.
        library_id: ID of schema collection.
        validator: Validator to use.

    Returns:
        Data in JSON output format.
    """
    all_events: MutableSequence[Event] = []
    all_entities: MutableSequence[Entity] = []

    parsed_yaml = parse_obj_as(List[Schema], yaml_schemas)
    if [p.dict(exclude_none=True) for p in parsed_yaml] != yaml_schemas:
        raise RuntimeError(
            "The parsed and raw schemas do not match. The schema might have misordered fields, "
            "or there is a bug in this script."
        )
    for yaml_schema in parsed_yaml:
        events, entities = convert_yaml_to_sdf(yaml_schema, performer_prefix)
        all_events.extend(events)
        all_entities.extend(entities)

    latest_version = max(schema.schema_version for schema in parsed_yaml)
    json_data = merge_schemas(
        all_events, all_entities, latest_version, performer_prefix, performer_uri, library_id
    )

    if validator:
        validate_schemas(json_data, VALIDATOR_ENDPOINTS[validator])
    else:
        logging.info("Skipping validation")

    return json_data


def convert_files(
    yaml_files: Sequence[Path],
    json_file: Path,
    performer_prefix: str,
    performer_uri: str,
    validator: Optional[str],
) -> None:
    """Converts YAML files into a single JSON file.

    Args:
        yaml_files: List of YAML file paths.
        json_file: JSON file path.
        performer_prefix: Performer prefix for context.
        performer_uri: Performer URI for context.
        validator: Validator to use.
    """
    input_schemas = []
    for yaml_file in yaml_files:
        with yaml_file.open() as file:
            yaml_data = yaml.safe_load(file)
        input_schemas.extend(yaml_data)

    output_library = convert_all_yaml_to_sdf(
        input_schemas, performer_prefix, performer_uri, json_file.stem, validator
    )

    with json_file.open("w") as file:
        json.dump(output_library, file, ensure_ascii=True, indent=4)
        file.write("\n")


def main() -> None:
    """Converts YAML schema into JSON SDF."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-files", nargs="+", type=Path, required=True, help="Paths to input YAML schemas."
    )
    p.add_argument("--output-file", type=Path, required=True, help="Path to output JSON schema.")
    p.add_argument("--performer-prefix", required=True, help="Performer prefix for context.")
    p.add_argument("--performer-uri", required=True, help="Performer URI for context.")
    p.add_argument("--validator", choices=list(VALIDATOR_ENDPOINTS), help="Validator to use.")

    args = p.parse_args()

    convert_files(
        args.input_files,
        args.output_file,
        args.performer_prefix,
        args.performer_uri,
        args.validator,
    )


if __name__ == "__main__":
    main()
