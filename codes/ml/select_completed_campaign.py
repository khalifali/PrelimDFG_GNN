"""Apply the explicit user-approved exclusion to the completed 16-us campaign."""
import argparse,json,shutil
from pathlib import Path

def select(root):
    root=Path(root)
    case='fractal_dp2_N0100_Df2.6_kf0.8_rep04'
    bonded=root/'bonded'
    summary=json.loads((bonded/'run_summary.json').read_text())
    campaign=json.loads((bonded/'campaign.json').read_text())
    if summary['expected']!=180 or summary['completed']!=180 or campaign['count']!=180:
        raise ValueError('Expected completed 180-case source campaign')
    if [x['case'] for x in summary['failures']] != [case]:
        raise ValueError('Failure set differs from the approved single exclusion')
    for name in campaign['cases']:
        a=json.loads((bonded/name/'assessment.json').read_text())
        if name!=case and not a['passed']:
            raise ValueError(f'Unapproved failed case: {name}')
    rejected=json.loads((bonded/case/'assessment.json').read_text())
    report=dict(source_run_id=35288931479,original_cases=180,included_cases=179,
                core_cases=134,extension_cases=45,excluded_case=case,
                reason='User approved exclusion of the single failed DEM assessment before ML.',
                failed_assessment=rejected)
    # Keep the complete failed case and the source accounting for audit.
    (root/'excluded').mkdir(exist_ok=True)
    shutil.move(str(bonded/case),str(root/'excluded'/case))
    shutil.copy2(bonded/'campaign.json',bonded/'source_campaign.json')
    campaign['cases']=[x for x in campaign['cases'] if x!=case]
    campaign['count']=179
    campaign['selection_report']='../selection_report.json'
    (bonded/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n')
    (root/'selection_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Selected 179 passing cases; excluded case preserved with original assessment.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');select(p.parse_args().root)
