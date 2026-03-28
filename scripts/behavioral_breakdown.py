"""Extract behavioral time breakdown from FLOG training_log.bin files.
Parses FRAME_TRAINING (type=2) frames for the 'behavior' field.
Analyzes multiple ablation run directories.

Usage: py -3.11 scripts/behavioral_breakdown.py
"""
import struct, json, os, sys
from collections import Counter

try:
    import msgpack
except ImportError:
    print("ERROR: pip install msgpack")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: pip install numpy")
    sys.exit(1)


def parse_flog_behaviors(flog_path):
    """Parse a FLOG file and return behavior counts from FRAME_TRAINING frames."""
    behaviors = []
    steps = []
    
    with open(flog_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'FLOG':
            print(f"  ERROR: Not a FLOG file: {flog_path}")
            return None, None
        
        version = struct.unpack('<H', f.read(2))[0]
        phase = struct.unpack('<B', f.read(1))[0]
        meta_len = struct.unpack('<I', f.read(4))[0]
        meta = json.loads(f.read(meta_len))
        
        while True:
            header = f.read(13)
            if len(header) < 13:
                break
            ts, frame_type, data_len = struct.unpack('<dBI', header)
            payload = f.read(data_len)
            if len(payload) < data_len:
                break
            
            # Only parse FRAME_TRAINING (type=2)
            if frame_type == 2:
                try:
                    data = msgpack.unpackb(payload, raw=False)
                    if isinstance(data, dict):
                        beh = data.get('behavior', None)
                        step = data.get('step', 0)
                        if beh:
                            # Clean behavior string (might contain freq/amp info)
                            beh_clean = beh.split('(')[0].strip().split(' ')[0].strip()
                            behaviors.append(beh_clean)
                            steps.append(step)
                except Exception as e:
                    pass
            else:
                # Skip non-training frames
                pass
    
    return behaviors, meta


def analyze_behaviors(behaviors):
    """Calculate behavior percentages."""
    if not behaviors:
        return {}
    counts = Counter(behaviors)
    total = len(behaviors)
    result = {}
    for beh, count in sorted(counts.items(), key=lambda x: -x[1]):
        result[beh] = {
            'count': count,
            'percentage': round(100.0 * count / total, 1)
        }
    return result


def main():
    base_dir = r"D:\claude\DATEN"
    
    if not os.path.exists(base_dir):
        print(f"ERROR: {base_dir} not found")
        sys.exit(1)
    
    # Collect all ablation run directories
    conditions = {
        'A1_cpg_flat': 'abl_cpg_flat',
        'A2_cpg_hilly': 'abl_cpg_hilly', 
        'B1_snn_flat': 'abl_cpg_snn_flat',
        'B2_snn_hilly': 'abl_cpg_snn_hilly',
        'C1_full_flat': 'abl_full_flat',
        'C2_full_hilly': 'abl_full_hilly',
    }
    
    all_results = {}
    
    for cond_name, dir_prefix in conditions.items():
        cond_results = []
        
        # Find all directories matching this condition
        for entry in sorted(os.listdir(base_dir)):
            if entry.startswith(dir_prefix + '_') and os.path.isdir(os.path.join(base_dir, entry)):
                flog_path = os.path.join(base_dir, entry, 'training_log.bin')
                if os.path.exists(flog_path):
                    behaviors, meta = parse_flog_behaviors(flog_path)
                    if behaviors:
                        analysis = analyze_behaviors(behaviors)
                        cond_results.append({
                            'dir': entry,
                            'n_samples': len(behaviors),
                            'behaviors': analysis
                        })
                    else:
                        print(f"  WARNING: No behavior data in {entry}")
        
        if cond_results:
            # Aggregate across seeds
            all_behaviors = set()
            for r in cond_results:
                all_behaviors.update(r['behaviors'].keys())
            
            aggregated = {}
            for beh in sorted(all_behaviors):
                pcts = [r['behaviors'].get(beh, {}).get('percentage', 0.0) for r in cond_results]
                if pcts:
                    aggregated[beh] = {
                        'mean_pct': round(float(np.mean(pcts)), 1),
                        'std_pct': round(float(np.std(pcts)), 1),
                        'n_seeds': len(pcts)
                    }
            
            all_results[cond_name] = {
                'n_seeds': len(cond_results),
                'behaviors': aggregated,
                'per_seed': cond_results
            }
        else:
            print(f"  {cond_name}: No run directories found with prefix '{dir_prefix}_'")
    
    # Print results
    print()
    print("=" * 80)
    print("BEHAVIORAL TIME BREAKDOWN — Go2 Ablation (10 Seeds)")
    print("=" * 80)
    
    for cond_name in ['A1_cpg_flat', 'A2_cpg_hilly', 'B1_snn_flat', 'B2_snn_hilly', 'C1_full_flat', 'C2_full_hilly']:
        if cond_name not in all_results:
            print(f"\n{cond_name}: NO DATA")
            continue
        
        r = all_results[cond_name]
        print(f"\n{cond_name} ({r['n_seeds']} seeds):")
        print(f"  {'Behavior':<20s} {'Mean %':>8s} {'± Std':>8s}")
        print(f"  {'-'*20} {'-'*8} {'-'*8}")
        for beh, stats in sorted(r['behaviors'].items(), key=lambda x: -x[1]['mean_pct']):
            print(f"  {beh:<20s} {stats['mean_pct']:>7.1f}% {stats['std_pct']:>7.1f}%")
    
    # Save JSON
    output_path = r"D:\claude\mhflocke\output\behavioral_breakdown.json"
    json_results = {}
    for k, v in all_results.items():
        json_results[k] = {
            'n_seeds': v['n_seeds'],
            'behaviors': v['behaviors']
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
