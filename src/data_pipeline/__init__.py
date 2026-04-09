# Data pipeline: dataset loading, pre-processing, and PyTorch wrappers
#
# Key modules:
#   registry    — public dataset catalog + download info
#   sample      — BiomechanicalSample: unified data format
#   transforms  — normalization, windowing, filtering
#   torch_datasets — DynamicsDataset, FlightPhaseDataset, PoseLiftingDataset
#   loaders/    — per-dataset readers (addbiomechanics, biocv, opencap, athletepose3d)
