# KAIROS YAML Converter

## Overview

The KAIROS YAML converter transforms files between an internal YAML format and the JSON DARPA KAIROS data format.

## Requirements

The following software is required on the platform:

- Python 3.7.8
- Make

For running all checks, the following software is also needed:

- GNU Coreutils
- [Prettier](https://prettier.io/)

## Installation

Make sure all requirements are installed on the system before beginning setup.

To set up the program environment, run `make install`. Creating a virtualenv beforehand is likely desired.

## Usage

To convert schemas from YAML to JSON, run a command like the following:

```bash
python -m sdf.yaml2sdf --input-files example.yaml --output-file example.json
```

If a new version of the ontology is released, run the following beforehand, substituting in the path to the newest ontology file:

```bash
python convert_ontology.py --in-file KAIROS_Annotation_Tagset_Phase_1_V3.0.xlsx --out-file ontology.json
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

## License

This code is derived from [dmort27/kairos-yaml](https://github.com/dmort27/kairos-yaml), which was created by David R. Mortensen of CMU and is under the MIT license.

For complete license information, see [`LICENSE`](LICENSE).
