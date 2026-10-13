#!/usr/bin/env python3
"""Build the paper's reduced EMA input directly from the parent-study OSF raw exports.

Input: data/raw/fsqpv-osfstorage-archive.zip, or the extracted data/raw/Raw_data/ CSVs.
Output: data/derived/ema_prompt_reduced.csv.

No private data or stored crosswalks are needed. The output uses pseudonymous IDs
and dates shifted to a common January 2020 origin while preserving time gaps.
"""
from __future__ import annotations
import io
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'derived' / 'ema_prompt_reduced.csv'
ARCHIVE = RAW / 'fsqpv-osfstorage-archive.zip'


def load_source() -> pd.DataFrame:
    batches = []
    if ARCHIVE.exists():
        with zipfile.ZipFile(ARCHIVE) as z:
            names = sorted(n for n in z.namelist() if '/Raw_data/' in '/' + n and n.endswith('.csv'))
            if len(names) != 10:
                raise ValueError(f'Expected 10 raw batch CSVs in OSF archive, found {len(names)}')
            for name in names:
                label = Path(name).name
                data = pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)
                data['Group'] = label.split(' Condition ')[0].replace('Medium', 'Med')
                batches.append(data)
    else:
        candidates = sorted((RAW / 'Raw_data').glob('*.csv')) or sorted(RAW.glob('*.csv'))
        if len(candidates) != 10:
            raise FileNotFoundError('Download https://osf.io/fsqpv/ and place its ZIP at '
                                    f'{ARCHIVE} or extract all 10 batch files under data/raw/Raw_data/.')
        for path in candidates:
            data = pd.read_csv(path, low_memory=False)
            data['Group'] = path.name.split(' Condition ')[0].replace('Medium', 'Med')
            batches.append(data)
    return pd.concat(batches, ignore_index=True)


def prepare() -> pd.DataFrame:
    d = load_source()
    if len(d) != 16753:
        raise ValueError(f'Unexpected OSF raw row count ({len(d)}; expected 16,753). Verify OSF archive version.')
    degree = [c for c in d if '_VAS]' in c and 'Craving Degree' in c and 'Metadata' not in c]
    amount = [c for c in d if '_SAQ]' in c and 'Snacking Amount' in c and 'Metadata' not in c]
    if len(degree) != 10 or len(amount) != 10:
        raise ValueError('OSF questionnaire columns changed: expected 10 craving and 10 snack-amount items.')

    # Original raw timestamps include a numerical offset. Interpret them as
    # instants, then recover local European wall time for the 2024 study.
    d['scheduled_datetime'] = (pd.to_datetime(d['Session Scheduled Time'], utc=True)
        .dt.tz_convert('Europe/Amsterdam').dt.tz_localize(None))
    d['Craved_YN'] = d['[1_SAQ] 1. Craving']
    d['Snacked_YN'] = d['[16_SAQ] 2. Intake']
    d['CD'] = d[degree].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    d['SA'] = (d[amount].replace({'more than 5': 6})
               .apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1))
    # Explicit 'No' means a zero. A 'Yes' without a follow-up response remains
    # missing, not zero. This differs from some values in the OSF processed Excel.
    d.loc[d.Craved_YN.eq('No'), 'CD'] = 0
    d.loc[d.Snacked_YN.eq('No'), 'SA'] = 0
    d['Self_Control'] = pd.to_numeric(d['[30_VAS] 3. Self-Control'], errors='coerce')
    d['Stress'] = pd.to_numeric(d['[31_VAS] 4. Stress'], errors='coerce')
    d['answered_prompt'] = (d.Craved_YN.notna() | d.Snacked_YN.notna()).astype(int)
    d['original_id'] = d['Participant ID'].astype(int)

    # The source analysis removes 21 near-duplicate notifications (<=6 minutes)
    # while keeping the more complete entry in each pair, breaking ties by time.
    d = d.sort_values(['original_id', 'scheduled_datetime'], kind='stable').reset_index(drop=True)
    gaps = d.groupby('original_id').scheduled_datetime.diff().dt.total_seconds() / 60
    d['duplicate_group'] = (gaps.isna() | gaps.gt(6)).groupby(d.original_id).cumsum()
    score = ['CD', 'SA', 'Self_Control', 'Stress', 'Craved_YN', 'Snacked_YN']
    d['completeness'] = d[score].notna().sum(axis=1)
    d = (d.sort_values(['original_id', 'duplicate_group', 'completeness', 'scheduled_datetime'],
                       ascending=[True, True, False, True], kind='stable')
          .drop_duplicates(['original_id', 'duplicate_group'])
          .sort_values(['original_id', 'scheduled_datetime'], kind='stable').reset_index(drop=True))
    if len(d) != 16732 or d.original_id.nunique() != 136:
        raise ValueError('Deduplication does not reproduce the 16,732 rows / 136 people in the manuscript.')

    # Deterministic per-person pseudonyms; no original IDs or dates are exported.
    d['ID'] = 'P' + d.original_id.rank(method='dense').astype(int).astype(str).str.zfill(3)
    first = d.groupby('original_id').scheduled_datetime.transform('first')
    d['shifted'] = pd.Timestamp('2020-01-01') + (d.scheduled_datetime - first.dt.normalize())
    d['Prompt'] = d.groupby('original_id').cumcount() + 1
    # The source pipeline used 24-hour periods from the participant's first
    # scheduled notification, rather than calendar-day boundaries.
    d['StudyDay'] = np.floor((d.scheduled_datetime - first).dt.total_seconds() / 86400).astype(int) + 1
    d['expected_prompts_per_day'] = d.Group.map({'Low': 3, 'Med': 6, 'High': 9})

    # Preserve the exact clock-hour definition used by the original R matcher.
    # Historical matching converted an already-local naive scheduled timestamp
    # as though it were UTC; correcting that convention here would change the
    # published matched controls. Do not substitute the hour of 'shifted'.
    historical = d.scheduled_datetime.dt.tz_localize('UTC').dt.tz_convert('Europe/Amsterdam')
    d['original_matching_hour'] = historical.dt.hour + historical.dt.minute / 60 + historical.dt.second / 3600

    cols = ['ID', 'Group', 'scheduled_datetime', 'original_matching_hour', 'Prompt',
            'expected_prompts_per_day', 'StudyDay', 'CD', 'Craved_YN', 'Snacked_YN',
            'SA', 'Self_Control', 'Stress', 'answered_prompt']
    result = d[cols].copy()
    result['scheduled_datetime'] = d.shifted.dt.strftime('%Y-%m-%d %H:%M:%S')
    sizes = result.groupby('Group').size().to_dict()
    if sizes != {'High': 8107, 'Low': 3083, 'Med': 5542}:
        raise ValueError(f'Unexpected group counts: {sizes}')
    return result


def main() -> None:
    out = prepare()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f'Prepared {len(out):,} prompts from {out.ID.nunique()} participants.')
    print('Removed 21 near-duplicate records; saved', OUT.relative_to(ROOT))
    import json
    report_dir = ROOT / 'results' / '00_prepare_data'
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {'source':'OSF raw CSV batches','retained_prompts':len(out),
              'participants':int(out.ID.nunique()),'deduplicated_records':21,
              'derived_file':str(OUT.relative_to(ROOT))}
    (report_dir / 'preparation_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
