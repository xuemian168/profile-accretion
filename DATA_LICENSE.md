# Data in this repository

**Code** (`experiments/`, `scripts/`) is released under the MIT License (see `LICENSE`).

**Source data.** `data/raw/` contains record strings derived from two public sources. They remain subject to the terms of those sources; please cite them when reusing.

- **FOPPA v1.1.3** (French Open Public Procurement Award notices). L. Potin, V. Labatut, P.-H. Morand, C. Largeron, "FOPPA: an open database of French public procurement award notices from 2010–2020", *Scientific Data* 10:303 (2023), doi:10.1038/s41597-023-02213-z. Database on Zenodo: v1.1.3, used in this study, doi:10.5281/zenodo.10879932 (concept DOI for all versions 10.5281/zenodo.7433154), licensed CC BY 4.0. FOPPA is itself built from TED.
- **TED CSV subset** (contract award notices 2014–2017), Tenders Electronic Daily, supplement to the Official Journal of the European Union, DG Internal Market, Industry, Entrepreneurship and SMEs, European Commission, https://data.europa.eu/data/datasets/ted-csv. TED notices may be reused under the Commission's reuse policy (Commission Decision 2011/833/EU), provided the source is acknowledged and the original meaning is not distorted.

Only the record strings needed to reproduce the prompts are included (supplier name, city/postcode or street address, and the registry identifier used as ground truth); the full source databases are not redistributed.

**Model outputs and derived tables** (`data/raw/**/*.jsonl` scores and completions, `data/derived/`) were produced by the authors and are released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), subject to the source terms above for any record text they contain. Outputs of Claude models were obtained through the provider's API; open-weight model outputs were produced locally under each model's licence.
