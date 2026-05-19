# Experiment 004: Dynamics PINN Public-Data Validation

Aggregate metrics only. No private athlete data.

## Scope

This evaluates the pretrained inverse-dynamics PINN against local public
force-plate biomechanics datasets. It validates the dynamics model, not
the phone-video pose/calibration pipeline.

- Checkpoint: `experiments\results\pretrain_dynamics\best_model.pth`
- Important caveat: the original checkpoint was not trained with a
  formal held-out split. Treat this as a post-hoc benchmark until the
  model is retrained with the subject split used here.

## Metrics

| Dataset | Windows | Model vRMSE (BW) | Mean vRMSE (BW) | BW vRMSE (BW) | Model peak MAE (BW) | Model vCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cmj_grf_zenodo | 2394 | 0.125 | 0.408 | 0.420 | 0.063 | 0.952 |
| cod_ik_id_zenodo | 36 | 0.654 | 0.689 | 0.858 | 0.281 | 0.533 |

## Interpretation

- The `mean_profile_baseline` is the average GRF waveform from the
  training side of the subject split.
- The `bodyweight_baseline` predicts quiet standing force
  (`Fy = 1 BW`) for every frame.
- A useful pretrained model should beat these baselines on vertical
  RMSE and peak-force error, especially on jump-like datasets.
