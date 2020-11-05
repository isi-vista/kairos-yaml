"""Converts CMU YAML into KAIROS SDF JSON-LD."""

import argparse
from collections import Counter, defaultdict
import itertools
import json
import logging
from pathlib import Path
import random
import typing
from typing import Any, List, Mapping, MutableMapping, Sequence, Tuple, Union

from pydantic import parse_obj_as
import requests
from typing_extensions import TypedDict
import yaml

from sdf.ontology import ontology
from sdf.yaml_schema import Before, Container, Overlaps, Schema, Slot, Step


class StepMapItem(TypedDict):
    """Typing information for step_map.

    Attributes:
        id: Step ID.
        step_idx: Index of step.
    """
    id: str
    step_idx: int


def get_step_type(step: Step) -> str:
    """Gets type of step.

    Args:
        step: Step data.

    Returns:
        Step type.
    """
    primitive = ontology.get_default_event(step.primitive)

    if primitive not in ontology.events:
        logging.warning(f"Primitive '{step.primitive}' in step '{step.id}' not in ontology")

    return f"kairos:Primitives/Events/{primitive}"


def get_slot_role(slot: Slot, step_type: str) -> str:
    """Gets slot role.

    Args:
        slot: Slot data.
        step_type: Type of step.

    Returns:
        Slot role.
    """
    event_type = ontology.events.get(step_type.split("/")[-1], None)
    if event_type is not None and slot.role not in event_type.args:
        logging.warning(f"Role '{slot.role}' is not valid for event '{event_type.full_type}'")

    return f"{step_type}/Slots/{slot.role}"


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
        name = name[:uppercase_indices[1]]
    name = name.lower()
    if slot_shared and slot.refvar is not None:
        name += "-" + slot.refvar
    return name


def get_slot_id(slot: Slot, schema_slot_counter: typing.Counter[str],
                parent_id: str, slot_shared: bool) -> str:
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
            logging.warning(f"Entity '{entity}' not in ontology")

    return [f"kairos:Primitives/Entities/{entity}" for entity in constraints]


def create_slot(slot: Slot, schema_slot_counter: typing.Counter[str], schema_id: str, step_type: str,
                slot_shared: bool, entity_map: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Gets slot.

    Args:
        slot: Slot data.
        schema_slot_counter: Slot counter.
        schema_id: Schema ID.
        step_type: Type of step.
        slot_shared: Whether slot is shared.
        entity_map: Mapping from mentions to entities.

    Returns:
        Slot.
    """
    cur_slot: MutableMapping[str, Any] = {
        "name": get_slot_name(slot, slot_shared),
        "@id": get_slot_id(slot, schema_slot_counter, schema_id, slot_shared),
        "role": get_slot_role(slot, step_type),
    }

    if slot.constraints is None:
        primitive = step_type.split("/")[-1]
        slot.constraints = ontology.events[primitive].args[slot.role].constraints
    constraints = get_slot_constraints(slot.constraints)
    cur_slot["entityTypes"] = constraints
    if slot.reference is not None:
        cur_slot["reference"] = slot.reference

    # Get entity ID for relations
    if slot.refvar is not None:
        entity_map[cur_slot["@id"]] = slot.refvar
        cur_slot["refvar"] = slot.refvar
    else:
        logging.warning(f"{slot} misses refvar")
        entity_map[cur_slot["@id"]] = str(random.random())

    if slot.comment is not None:
        cur_slot["comment"] = slot.comment

    return cur_slot


def get_step_id(step: Step, schema_id: str) -> str:
    """Gets step ID.

    Args:
        step: Step data.
        schema_id: Schema ID.

    Returns:
        Step ID.
    """
    return f"{schema_id}/Steps/{step.id}"


def create_orders(
        yaml_data: Schema, schema_id: str, step_map: Mapping[str, StepMapItem]
) -> Sequence[Mapping[str, Any]]:
    """Gets orders.

    Args:
        yaml_data: Data from YAML file.
        schema_id: Schema ID.
        step_map: Mapping of steps containing some order information.

    Returns:
        Orders in SDF format.
    """
    step_ids = set(step.id for step in yaml_data.steps)
    order_tuples: List[Tuple[str, ...]] = []
    for order in yaml_data.order:
        if isinstance(order, Before):
            order_tuples.append((order.before, order.after))
        elif isinstance(order, Container):
            order_tuples.append((order.container, order.contained))
        elif isinstance(order, Overlaps):
            order_tuples.append(tuple(order.overlaps))
        else:
            raise NotImplementedError
    order_ids = set(itertools.chain.from_iterable(order_tuples))
    missing_order_ids = order_ids - step_ids
    if missing_order_ids:
        for missing_id in missing_order_ids:
            logging.error(f"The ID '{missing_id}' in `order` is not in `steps`")
        exit(1)

    base_order_id = f'{schema_id}/Order/'
    orders = []
    for order in yaml_data.order:
        if isinstance(order, Before):
            before_id = step_map[order.before]['id']
            after_id = step_map[order.after]['id']
            cur_order: MutableMapping[str, Union[str, Sequence[str]]] = {
                "@id": f"{base_order_id}precede-{order.before}-{order.after}",
                "comment": f"{order.before} precedes {order.after}",
                "before": before_id,
                "after": after_id
            }
        elif isinstance(order, Container):
            container_id = step_map[order.container]['id']
            contained_id = step_map[order.contained]['id']
            cur_order = {
                "@id": f"{base_order_id}contain-{order.container}-{order.contained}",
                "comment": f"{order.container} contains {order.contained}",
                "container": container_id,
                "contained": contained_id
            }
        elif isinstance(order, Overlaps):
            overlaps_id = [step_map[overlap]['id'] for overlap in order.overlaps]
            cur_order = {
                "@id": f"{base_order_id}overlap-{'-'.join(order.overlaps)}",
                "comment": f"{', '.join(order.overlaps)} overlaps",
                "overlaps": overlaps_id,
            }
        else:
            raise NotImplementedError
        # isinstance() check will always be True but is needed for mypy
        if order.comment is not None and isinstance(cur_order["comment"], str):
            cur_order["comment"] = [cur_order["comment"], order.comment]
        orders.append(cur_order)
    return orders


def convert_yaml_to_sdf(yaml_data: Schema, performer_prefix: str) -> Mapping[str, Any]:
    """Converts YAML to SDF.

    Args:
        yaml_data: Data from YAML file.
        performer_prefix: Performer prefix for context.

    Returns:
        Schema in SDF format.
    """
    schema: MutableMapping[str, Any] = {
        "@id": f"{performer_prefix}:Schemas/{yaml_data.schema_id}",
        "comment": [],
        "super": "kairos:Event",
        "name": yaml_data.schema_name,
        "description": yaml_data.schema_dscpt,
        "version": yaml_data.schema_version,
        "steps": [],
        "order": [],
        "entityRelations": []
    }

    # Get comments
    comments = [x.id.replace("-", " ") for x in yaml_data.steps]
    comments = ["Steps:"] + [f"{idx}. {text}" for idx, text in enumerate(comments, start=1)]
    schema["comment"] = comments
    if yaml_data.comment is not None:
        schema["comment"].append(yaml_data.comment)

    # Get steps
    steps = []

    # For sameAs relation
    entity_map: MutableMapping[str, Any] = {}

    # For order
    step_map: MutableMapping[str, StepMapItem] = {}

    # For naming slot ID
    schema_slot_counter: typing.Counter[str] = Counter()

    for idx, step in enumerate(yaml_data.steps):
        cur_step: MutableMapping[str, Any] = {
            "@id": get_step_id(step, schema["@id"]),
            "name": step.id,
            "@type": get_step_type(step),
            "comment": comments[idx + 1],
        }
        if step.comment is not None:
            cur_step["comment"] = [cur_step["comment"], step.comment]

        step_map[step.id] = {"id": cur_step["@id"], "step_idx": idx + 1}

        slots = []
        for slot in step.slots:
            slot_shared = sum([slot.role == sl.role for sl in step.slots]) > 1

            slots.append(
                create_slot(slot, schema_slot_counter, cur_step["@id"], cur_step["@type"], slot_shared, entity_map))

        cur_step["participants"] = slots
        steps.append(cur_step)

    slots = []
    for slot in yaml_data.slots:
        slot_shared = sum([slot.role == sl.role for sl in yaml_data.slots]) > 1

        parsed_slot = create_slot(slot, schema_slot_counter, schema["@id"], schema["@id"], slot_shared, entity_map)
        parsed_slot["roleName"] = parsed_slot["role"]
        del parsed_slot["role"]

        slots.append(parsed_slot)
    schema["slots"] = slots

    # Cleaning "-a" suffix for slots with counter == 1.
    for cur_step in steps:
        for cur_slot in cur_step["participants"]:
            if schema_slot_counter[cur_slot["name"]] == 1:
                temp = entity_map[cur_slot["@id"]]
                del entity_map[cur_slot["@id"]]

                cur_slot["@id"] = cur_slot["@id"].strip("-a")

                entity_map[cur_slot["@id"]] = temp

    schema["steps"] = steps

    schema["order"] = create_orders(yaml_data, schema["@id"], step_map)

    # Get entity relations
    entity_map = {x: y for x, y in entity_map.items() if y is not None}
    # Get same as relation
    reverse_entity_map = defaultdict(list)
    for k, v in entity_map.items():
        reverse_entity_map[v].append(k)
    entity_relations: Sequence[Any] = []
    # for v in reverse_entity_map.values():
    #     cur_entity_relation = {
    #         "relationSubject": v[0],
    #         "relations": [{
    #             "relationPredicate": "kairos:Relations/sameAs",
    #             "relationObject": x
    #         } for x in v[1:]]
    #     }
    #     if cur_entity_relation["relations"]:
    #         entity_relations.append(cur_entity_relation)
    schema["entityRelations"] = entity_relations

    return schema


def merge_schemas(schema_list: Sequence[Mapping[str, Any]], performer_prefix: str,
                  performer_uri: str, library_id: str) -> Mapping[str, Any]:
    """Merge multiple schemas.

    Args:
        schema_list: List of SDF schemas.
        performer_prefix: Performer prefix for context.
        performer_uri: Performer URI for context.
        library_id: ID of schema collection.

    Returns:
        Data in JSON output format.
    """
    sdf = {
        "@context": [
            "https://kairos-sdf.s3.amazonaws.com/context/kairos-v0.93.jsonld",
            {performer_prefix: performer_uri}
        ],
        "sdfVersion": "0.93",
        "@id": f"{performer_prefix}:Submissions/TA1/{library_id}",
        "schemas": schema_list,
    }

    return sdf


def validate_schemas(json_data: Mapping[str, Any]) -> None:
    """Validates generated schema against the program validator.

    The program validator is not always avoilable, so the request will time out if no response is
    received within 10 seconds.

    Args:
        json_data: Data in JSON output format.
    """
    try:
        req = requests.post("http://validation.kairos.nextcentury.com/json-ld/ksf/validate",
                            json=json_data,
                            headers={
                                "Accept": "application/json",
                                "Content-Type": "application/ld+json"
                            },
                            timeout=10)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        logging.warning("Program validator is unavailable, so schema might not validate")
    else:
        response = req.json()
        validator_messages = response['errorsList'] + response['warningsList']
        if validator_messages:
            print('Messages from program validator:')
            for message in validator_messages:
                print(f'\t{message}')


def convert_all_yaml_to_sdf(yaml_schemas: Sequence[Mapping[str, Any]], performer_prefix: str,
                            performer_uri: str, library_id: str) -> Mapping[str, Any]:
    """Convert YAML schema library into SDF schema library.

    Args:
        yaml_schemas: YAML schemas.
        performer_prefix: Performer prefix for context.
        performer_uri: Performer URI for context.
        library_id: ID of schema collection.

    Returns:
        Data in JSON output format.
    """
    sdf_schemas = []

    parsed_yaml = parse_obj_as(List[Schema], yaml_schemas)
    if [p.dict(exclude_none=True) for p in parsed_yaml] != yaml_schemas:
        raise RuntimeError(
            "The parsed and raw schemas do not match. The schema might have misordered fields, or there is a bug in this script.")
    for yaml_schema in parsed_yaml:
        out_json = convert_yaml_to_sdf(yaml_schema, performer_prefix)
        sdf_schemas.append(out_json)

    json_data = merge_schemas(sdf_schemas, performer_prefix, performer_uri, library_id)

    validate_schemas(json_data)

    return json_data


def convert_files(yaml_files: Sequence[Path], json_file: Path, performer_prefix: str,
                  performer_uri: str) -> None:
    """Converts YAML files into a single JSON file.

    Args:
        yaml_files: List of YAML file paths.
        json_file: JSON file path.
        performer_prefix: Performer prefix for context.
        performer_uri: Performer URI for context.
    """
    input_schemas = []
    for yaml_file in yaml_files:
        with yaml_file.open() as file:
            yaml_data = yaml.safe_load(file)
        input_schemas.extend(yaml_data)

    output_library = convert_all_yaml_to_sdf(input_schemas, performer_prefix, performer_uri, json_file.stem)

    with json_file.open("w") as file:
        json.dump(output_library, file, ensure_ascii=True, indent=4)


def main() -> None:
    """Converts YAML schema into JSON SDF."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-files", nargs="+", type=Path, required=True,
                   help="Paths to input YAML schemas.")
    p.add_argument("--output-file", type=Path, required=True,
                   help="Path to output JSON schema.")
    p.add_argument("--performer-prefix", required=True,
                   help="Performer prefix for context.")
    p.add_argument("--performer-uri", required=True,
                   help="Performer URI for context.")

    args = p.parse_args()

    convert_files(args.input_files, args.output_file, args.performer_prefix, args.performer_uri)


if __name__ == "__main__":
    main()
