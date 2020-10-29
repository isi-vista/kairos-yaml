"""Converts KAIROS ontology spreadsheet into usable JSON format."""

import argparse
import enum
import json
from pathlib import Path
import re
from typing import Any, Mapping, MutableMapping, Union

import pandas as pd

from sdf.ontology import Arg, Entity, Ontology, Predicate


@enum.unique
class Sheets(enum.Enum):
    """Enum for sheets and their names."""

    EVENTS = "events"
    ENTITIES = "entities"
    RELATIONS = "relations"


def convert_sheet(sheet: pd.DataFrame, sheet_type: Sheets) -> Mapping[str, Any]:
    """Converts spreadsheet events, entities, and relations into a usable format.

    Args:
        sheet: Contents of "events", "entities", or "relations" sheet.
        sheet_type: Type of sheet to convert.

    Returns:
        Usable representation of events, entities, or relations.
    """
    arg_label_matches = (re.match(r"arg(\d+) label", col) for col in sheet.columns)
    arg_label_ints = (int(match.group(1)) for match in arg_label_matches if match is not None)
    max_arg_label_col = max(arg_label_ints, default=0) + 1

    items: MutableMapping[str, Union[Entity, Predicate]] = {}
    for row in sheet.iterrows():
        row = row[1]
        if sheet_type == Sheets.ENTITIES:
            item_type = str(row["Type"])
            items[item_type] = Entity(
                id=row["AnnotIndexID"],
                type=item_type,
                definition=row["Definition"],
            )
        else:
            item_type = ".".join([row["Type"], row["Subtype"], row["Sub-subtype"]])
            items[item_type] = Predicate(
                id=row["AnnotIndexID"],
                full_type=item_type,
                type=row["Type"],
                subtype=row["Subtype"],
                subsubtype=row["Sub-subtype"],
                definition=row["Definition"],
                template=row["Template"],
                args={
                    row[f"arg{i} label"]: Arg(
                        position=f"arg{i}",
                        label=row[f"arg{i} label"],
                        constraints=row[f"arg{i} type constraints"].upper().split(", "),
                    )
                    for i in range(1, max_arg_label_col)
                    if isinstance(row[f"arg{i} label"], str)
                },
            )
    return items


def main() -> None:
    """Converts spreadsheet to formatted JSON."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in-file",
        type=Path,
        required=True,
        help="Path to input KAIROS ontology Excel spreadsheet.",
    )
    p.add_argument("--out-file", type=Path, required=True, help="Path to output JSON.")
    args = p.parse_args()

    source_file_name = args.in_file.name
    events_sheet = pd.read_excel(args.in_file, sheet_name=Sheets.EVENTS.value)
    events = convert_sheet(events_sheet, Sheets.EVENTS)
    entities_sheet = pd.read_excel(args.in_file, sheet_name=Sheets.ENTITIES.value)
    entities = convert_sheet(entities_sheet, Sheets.ENTITIES)
    relations_sheet = pd.read_excel(args.in_file, sheet_name=Sheets.RELATIONS.value)
    relations = convert_sheet(relations_sheet, Sheets.RELATIONS)
    ontology = Ontology(
        source_file=source_file_name,
        events=events,
        entities=entities,
        relations=relations,
    )
    output = ontology.dict()

    with open(args.out_file, "w") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
