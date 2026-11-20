"""Package the current source and bounded real demo data, without old ZIP contents."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    target=ROOT/'results/college_submission_ready_20261119.zip'
    files=set()
    for folder in ('src','app','assets','tests','scripts','docs','THIRD_PARTY_LICENSES','.github','.vscode','.streamlit','references/joint_solver_before_optimization'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and '.before_' not in path.name:
                files.add(path)
    for pattern in ('*.md','requirements*.txt','pyproject.toml','LICENSE','CITATION.cff','app.py'):
        files.update(ROOT.glob(pattern))
    for folder in ('data/real_control_variant_under200mb','data/real_division_clips','data/cshaper_demo',
                   'results/submission_demo_20261119','results/submission_biohub_20261119',
                   'results/submission_verification_20261119','results/project_audit_20261119',
                   'results/large_pair_review_20261119'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.npy','.json','.csv','.png','.md','.txt'):
                files.add(path)
    for rel in ('data/real_control_variant_large/source/Emb8_CellTracing.csv',
                'data/real_control_variant_large/source/MT2_CellTracing.csv',
                'models/cellpose_nuclei/nucleitorch_0','models/cellpose_nuclei/size_nucleitorch_0.npy',
                'models/cellpose_nuclei/PROVENANCE.json','models/full_annotation_linker/final_all_annotations.pt',
                'results/kaggle_ce_cshaper_v2/detector.pt'):
        files.add(ROOT/rel)
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    info=dict(purpose='Current source, real bounded demo clips, official pretrained nuclei weights and recorded local results',
              excluded='Large 30/100-frame microscopy, raw archives, environments, prior ZIPs, git internals and backup source files',
              includes_verified_genetic_diagnostic=False,sha256=hashes)
    temporary=target.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in sorted(files):z.write(path,str(path.relative_to(ROOT)).replace('\\','/'))
        z.writestr('RELEASE_MANIFEST.json',json.dumps(info,indent=2))
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
    temporary.replace(target)
    checksum=hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.zip.sha256').write_text(checksum+'  '+target.name+'\n')
    print(json.dumps(dict(file=str(target),bytes=target.stat().st_size,files=len(hashes)+1,sha256=checksum)),flush=True)


if __name__=='__main__':main()
