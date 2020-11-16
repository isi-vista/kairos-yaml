import argparse
import json

import yaml


def search_and_tag(sc_obj, target_slot, val):
    if "slots" in sc_obj:
        for slt in sc_obj["slots"]:
            if slt["id"] == target_slot:
                slt["refVar"] = val
                return

    for stp in sc_obj["steps"]:
        for slt in stp["slots"]:
            if slt["id"] == target_slot:
                slt["refVar"] = val
                return


def parse_arguments():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Convert SDF (0.93) to YAML and print to stdout, for easier read. Might omit"
                    "some fields in original json..."
    )
    parser.add_argument("sdf_path",
                        type=str,
                        help="path to SDF json file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    with open(args.sdf_path) as sdf_f:
        data = json.load(sdf_f)

    yaml_obj = {}

    if "ta2" in data:
        yaml_obj["ta2"] = data["ta2"]

    if "schemas" in data:
        yaml_obj["schemas"] = []

        for sch in data["schemas"]:
            sc_obj = {
                "id": sch["@id"],
                "super": sch["super"],
                "name": sch["name"],
                "description": sch["description"],
                "steps": []
            }
            if "comment" in sch:
                sc_obj["comment"] = sch["comment"]

            if "slots" in sch:
                sc_obj["slots"] = []
                for slt in sch["slots"]:
                    sl_obj = {
                        "id": slt["@id"],
                        "roleName": slt["roleName"].split("/")[-1]
                    }

                    opt_fields = ["name", "super", "entityTypes", "reference", "provenance", "aka"]
                    for field in opt_fields:
                        if field in slt:
                            sl_obj[field] = slt[field]

                    sc_obj["slots"].append(sl_obj)

            for stp in sch["steps"]:
                st_obj = {
                    "id": stp["@id"],
                    "primitive": stp["@type"].split("/")[-1],
                    "slots": []
                }

                opt_fields = [
                    "name", "description", "reference", "provenance", "comment", "aka",
                    "startTime", "endTime", "absoluteTime", "minDuration", "maxDuration"
                ]
                for field in opt_fields:
                    if field in stp:
                        st_obj[field] = stp[field]

                for slt in stp["participants"]:
                    sl_obj = {
                        "id": slt["@id"],
                        "name": slt["name"],
                        "role": slt["role"].split("/")[-1]
                    }
                    if "values" in slt:
                        if slt["values"] and len(slt["values"]) > 0:
                            sl_obj["values"] = slt["values"]

                    opt_fields = ["entityTypes", "reference", "provenance", "aka"]
                    for field in opt_fields:
                        if field in slt:
                            sl_obj[field] = slt[field]

                    st_obj["slots"].append(sl_obj)

                sc_obj["steps"].append(st_obj)

            if "order" in sch:
                sc_obj["order"] = []
                for ord in sch["order"]:
                    od_obj = {}

                    opt_fields = ["before", "after", "overlaps", "container", "contained", "flags"]
                    for field in opt_fields:
                        if field in ord:
                            if field != "flags":
                                if isinstance(ord[field], list):
                                    od_obj[field] = [id.split("/")[-1] for id in ord[field]]
                                else:
                                    od_obj[field] = ord[field].split("/")[-1]
                            else:
                                od_obj[field] = ord[field]

                    sc_obj["order"].append(od_obj)

            # Accounting for sameAs relations only
            refVar_counter = 0
            for rels in sch["entityRelations"]:
                has_corefs = False
                for r in rels["relations"]:
                    if r["relationPredicate"] == "kairos:Relations/sameAs":
                        has_corefs = True
                    if r["relationPredicate"] == "kairos:primitives/Relations/SameAs":
                        has_corefs = True

                if not has_corefs:
                    continue

                search_and_tag(sc_obj, rels["relationSubject"], str(refVar_counter))
                for r in rels["relations"]:
                    confVal = f" (conf.: {r['confidence']})" if "confidence" in r else ""
                    if r["relationPredicate"] == "kairos:Relations/sameAs" or r[
                        "relationPredicate"] == "kairos:primitives/Relations/SameAs":
                        search_and_tag(sc_obj, r["relationObject"], str(refVar_counter) + confVal)

                refVar_counter += 1

            # Prune or shorten ids for easier read
            sc_obj["id"] = sc_obj["id"].split("/")[-1]
            if "slots" in sc_obj:
                for slt in sc_obj["slots"]:
                    del slt["id"]
            for stp in sc_obj["steps"]:
                stp["id"] = stp["id"].split("/")[-1]
                for slt in stp["slots"]:
                    del slt["id"]

            yaml_obj["schemas"].append(sc_obj)

    if "primitives" in data:
        yaml_obj["primitives"] = []

        for prm in data["primitives"]:
            pm_obj = {
                "id": prm["@id"].split("/")[-1],
                "super": prm["super"],
                "name": prm["name"],
                "description": prm["description"],
                "slots": []
            }
            opt_fields = [
                "comment", "aka", "minDuration", "maxDuration"
            ]
            for field in opt_fields:
                if field in prm:
                    pm_obj[field] = prm[field]

            for slt in prm["slots"]:
                sl_obj = {
                    "id": slt["@id"].split("/")[-1],
                    "roleName": slt["roleName"]
                }

                opt_fields = ["entityTypes", "reference", "provenance", "aka"]
                for field in opt_fields:
                    if field in slt:
                        sl_obj[field] = slt[field]

                pm_obj["slots"].append(sl_obj)

            yaml_obj["primitives"].append(pm_obj)

    print(yaml.dump(yaml_obj, default_flow_style=False, sort_keys=False))
