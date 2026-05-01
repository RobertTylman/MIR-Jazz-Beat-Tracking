import json
path = 'evaluation/graphs-madmom.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)
nb['cells'][0]['source'] = [
    '# Madmom on Jazz Trio Database — Evaluation Report\n',
    '\n',
    'End-to-end analysis of the **madmom** model on the JTD dataset (1,294 tracks).\n',
    'Each track has been scored with the standard `mir_eval` beat-tracking metrics\n',
    '(F-measure, Cemgil, Goto, P-score, CMLc/t, AMLc/t, Information Gain) for both\n',
    '**beats** and **downbeats**.\n'
]
with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
