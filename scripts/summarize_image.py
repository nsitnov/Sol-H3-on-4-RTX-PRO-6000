"""Validate complete result metadata and report medians, excluding warmups."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt', type=Path)
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in sorted(args.attempt.glob('*/*/result.json'))]
    if len(records) != 8:
        raise SystemExit(f'Expected 8 successful records, found {len(records)}')
    groups = []
    for case in sorted({r['case']['id'] for r in records}):
        group = [r for r in records if r['case']['id'] == case]
        assert len(group) == 4 and sum(r['warmup'] for r in group) == 1
        for r in group:
            assert r['status'] == 'complete' and r['full_av_decode_passed']
            assert r['steps'] == 4 and r['seed'] == 42
            assert r['world_size']==4 and r['encoder_tp']['strategy']=='layer_pipeline'
            assert set(r['stage_seconds'])=={'text_encode','image_encode','denoise','video_decode','audio_decode'}
            assert r['full_request_seconds']>=r['pipeline_seconds']>0
            assert len(r['rank_audit']) == r['world_size']
            assert {a['rank'] for a in r['rank_audit']} == set(range(r['world_size']))
            for audit in r['rank_audit']:
                assert audit['forwards'] == 4
                attention = audit['attention']
                sparse=144 if r['task']=='i2v' else 200
                assert attention['sparse_calls'] == sparse * attention['requests']
                assert attention['dense_calls'] == (200-sparse) * attention['requests']
                assert attention['backend'] == 'sol_bsa'
                assert attention['qkv_wire_dtype'] == 'int8_qkv' and attention['output_wire_dtype'] == 'fp8'
                assert set(attention['declined']).issubset(({'warmup_step', 'dense_layer'} if r['task']=='i2v' else set()))
            video = next(s for s in r['ffprobe']['streams'] if s['codec_type'] == 'video')
            audio = next(s for s in r['ffprobe']['streams'] if s['codec_type'] == 'audio')
            assert (video['width'], video['height']) == (1344, 768)
            assert video['codec_name'] == 'h264' and video['avg_frame_rate'] == '24/1'
            assert int(video['nb_read_frames']) == {5: 124, 10: 243}[r['case']['seconds']]
            assert audio['codec_name'] == 'aac' and audio['channels'] == 2 and int(audio['sample_rate']) == 32000
            repeat = 'warmup' if r['warmup'] else f'repeat{r["repeat"]}'
            mp4 = args.attempt / case / repeat / 'output.mp4'
            assert hashlib.sha256(mp4.read_bytes()).hexdigest() == r['sha256']
        measured = [r for r in group if not r['warmup']]
        groups.append({'case': case, 'measured_repeats': len(measured),
                       'full_request_median_seconds': statistics.median(r['full_request_seconds'] for r in measured),
                       'pipeline_median_seconds': statistics.median(r['pipeline_seconds'] for r in measured),
                       'stage_medians_seconds':{k:statistics.median(r['stage_seconds'][k] for r in measured) for k in measured[0]['stage_seconds']},
                       'quality': 'pending_human_review'})
    result = {'groups': groups, 'successful_renders': len(records)}
    (args.attempt / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

