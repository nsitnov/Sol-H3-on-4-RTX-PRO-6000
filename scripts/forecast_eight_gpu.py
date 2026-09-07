"""Print explicitly hypothetical eight-GPU latencies from measured stage medians."""
import json
from common import ROOT

data = json.loads((ROOT / 'benchmarks/2026-09-07/measurements.json').read_text())
print('UNMEASURED SCENARIOS, not confidence intervals or promised speedups')
for group in data['groups']:
    total = group['full_request_median_seconds']
    denoise = group['stage_medians_seconds']['denoise']
    video = group['stage_medians_seconds']['video_decode']
    residual = total - denoise - video
    short = group['case'] == 't2v_dialogue'
    scenarios = [('conservative', 1.3, 1.2, 0.5 if short else 1.0),
                 ('balanced', 1.6, 1.5, 0.25 if short else 0.5),
                 ('idealized', 2.0, 2.0, 0.0)]
    print(group['case'], f'measured_4gpu={total:.2f}s')
    for name, sd, sv, extra in scenarios:
        predicted = residual + denoise / sd + video / sv + extra
        print(f'  {name}: {predicted:.2f}s, modeled ratio={total / predicted:.2f}x')

