# Asset Provenance and Terms Summary

This note summarizes the main third-party assets used by the paper and the upstream terms we were able to verify before submission.
It is intentionally conservative: the supplement redistributes only paper sources, code, configs, and derived paper-facing artifacts, not raw benchmark data, pretrained weights, or trained checkpoints.

## Code dependencies

- `WILDS`
  - Role: dataset wrappers and official evaluation utilities for `camelyon17` and `civilcomments`
  - Source: https://github.com/p-lambda/wilds
  - License shown by the official repository: MIT

- `torchvision`
  - Role: ResNet backbone implementation used in the vision stacks
  - Source: https://github.com/pytorch/vision
  - License shown by the official repository: BSD-3-Clause

## Main supervised benchmark assets

- `Camelyon17` via the WILDS patch-based release
  - Sources:
    - https://wilds.stanford.edu/datasets/
    - https://camelyon17.grand-challenge.org/Download/
  - Public-term note:
    - the current WILDS dataset page states CC0
    - the CAMELYON17 challenge website contains mixed public license notices across its pages
  - Submission posture: the supplement does not redistribute raw pathology images or extracted patches

- `CelebA`
  - Source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  - Official-page note: distributed for research use under the original CelebA terms
  - Submission posture: the supplement does not redistribute raw images

- `CivilComments` through WILDS
  - Sources:
    - https://wilds.stanford.edu/datasets/
    - https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification
  - Public-term note: acquisition remains governed by the original upstream terms
  - Submission posture: the supplement does not redistribute raw text shards

- `ACSIncome` / `Folktables`
  - Sources:
    - https://github.com/socialfoundations/folktables
    - https://www.census.gov/programs-surveys/acs/microdata.html
  - Public-term note: the benchmark is derived from public ACS microdata and distributed through the Folktables tooling/paper
  - Submission posture: the supplement does not redistribute raw census extracts

## Supplement policy

- The supplement contains:
  - paper source
  - code and configs
  - derived CSV / JSON artifacts referenced by the paper

- The supplement does not contain:
  - raw images from `Camelyon17` or `CelebA`
  - raw `CivilComments` or ACS / Folktables data
  - benchmark artifact repositories beyond the paper-facing derived summaries
