# Data

The datasets are public and are not redistributed with this code. Download them
and place them as follows.

## Elliptic (Bitcoin, transaction level)

- Source: Elliptic, "Bitcoin transaction data set" (Kaggle)
  https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- Expected file: `data/elliptic/processed/elliptic.pt`
- The `.pt` file is a dictionary with `x` (features), `y` (labels: 0 = illicit,
  1 = licit, 2 = unknown), `time_step`, and `edge_index`.
- Note: 72 of the 166 features are one-hop aggregated features; the leakage
  controls reported in the manuscript use the 94 local features only.

## BERT4ETH (Ethereum, account level)

- Source: BERT4ETH official repository
  https://github.com/git-disl/BERT4ETH
- Expected files: `data/bert4eth/raw/*.csv` for the Phishing / ENS / Tornado /
  Normal transaction sets.
- Build the graph used here with:

  ```bash
  python exp/preprocess_bert4eth.py     # -> data/bert4eth/processed/{nodes,edges}.npz
  python exp/prep_protocol.py           # -> results/eth_protocol.npz
  ```

## Notes

- The repository carries no explicit LICENSE for the BERT4ETH data; cite the
  original paper and follow the repository's terms when reusing them.
- Derived artefacts (the account-level graph and the temporal split) are produced
  by the two commands above and can be regenerated from the raw CSV files.
