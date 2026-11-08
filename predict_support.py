"""Replay frozen support inference using archived or user-supplied teacher anchors."""
import argparse
from pathlib import Path
import pandas as pd
from longwall_surrogate.inference import ROOT, support_prediction

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['development', 'archived'], default='archived')
    p.add_argument('--input', type=Path, help='Custom CSV with 8 features and teacher_anchor_MPa; five-student ensemble.')
    p.add_argument('--variant', default='full_CMCRNet', choices=['independent_multitask','plus_censored_loss','plus_coupled_ranking','full_CMCRNet'])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output path; existing files are not overwritten.')
    folds = None
    if a.input:
        frame = pd.read_csv(a.input)
        if 'teacher_anchor_MPa' not in frame:
            p.error('Custom inputs require teacher_anchor_MPa from the separately obtained teacher.')
        anchors = frame['teacher_anchor_MPa'].to_numpy()
    else:
        stem = 'development_labels.csv' if a.dataset == 'development' else 'archived_labels.csv'
        cache = 'single_stage_stage7_oof_predictions.csv' if a.dataset == 'development' else 'single_stage_stage8_predictions.csv'
        frame = pd.read_csv(ROOT/'data/support'/stem)
        teacher = pd.read_csv(ROOT/'data/support'/cache).set_index('sample_id').loc[frame['sample_id']]
        anchors = teacher['two_stage_predicted_pressure_MPa'].to_numpy()
        if a.dataset == 'development':
            folds = teacher['matched_outer_fold'].to_numpy()
    output = support_prediction(frame, anchors, folds, a.variant)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(a.output, index=False)
    print(f'Saved {len(output)} support predictions to {a.output}')

if __name__ == '__main__':
    main()
