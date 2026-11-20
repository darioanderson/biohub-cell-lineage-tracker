"""Freeze eight additional samples and bundle the private Kaggle evaluation."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/'notebooks/kaggle_joint_expansion';folder.mkdir(exist_ok=True)
    inventory=json.loads((root/'results/kaggle_division_bearing_diagnostics_v1/division_diagnostics/annotation_inventory.json').read_text())
    excluded=set()
    for run in ['kaggle_division_diagnostics_v1','kaggle_division_bearing_diagnostics_v1']:
        excluded.update(json.loads((root/'results'/run/'division_diagnostics/selection.json').read_text())['samples'])
    selected=[]
    for embryo in ['44b6','6bba']:
        for positive in [False,True]:
            names=sorted(n for n,c in inventory.items() if n.startswith(embryo+'_') and n not in excluded and (c>0)==positive)[:2]
            if len(names)!=2:raise ValueError('Insufficient samples')
            selected.extend({'name':n,'stratum':'annotated_division' if positive else 'no_annotated_division','annotated_divisions':inventory[n]} for n in names)
    scripts=['evaluate_joint_expansion.py','evaluate_detected_linker.py','evaluate_trackastra.py','division_candidate_audit.py']
    files=sorted((root/'src').rglob('*.py'))+[root/'scripts'/n for n in scripts]
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest={'samples':selected,'excluded_design_samples':sorted(excluded),
              'selection':'First two names in each embryo/annotation stratum; before candidate outcomes',
              'parameters':{'parent_gate':12.,'sister_gate':24.,'midpoint_gate':9.,'minimum_score':.05,'margin':.5,'persistence':2},
              'source_hashes':hashes,'limitations':['Previously used embryos and benchmark; no independent biological validation',
                  'No annotated divisions does not mean no biological divisions','Sparse unknowns remain unknown; no tuning or adoption from eight-volume test']}
    (folder/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2))
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,p.relative_to(root).as_posix())
        archive.writestr('frozen_manifest.json',json.dumps(manifest,indent=2))
    nb=json.loads((root/'notebooks/kaggle_division_bearing_diagnostics/biohub-division-bearing-diagnostics.ipynb').read_text())
    lines=[]
    for line in nb['cells'][1]['source']:
        if line.startswith('payload = '):line='payload = '+repr(base64.b64encode(buffer.getvalue()).decode())+'\n'
        if line.startswith('command = '):line="command = [sys.executable,str(project/'scripts/evaluate_joint_expansion.py'),str(train_dir),str(model_dir),'--manifest',str(project/'frozen_manifest.json'),'--output-dir','/kaggle/working/joint_expansion']\n"
        lines.append(line.replace('DIVISION DIAGNOSTIC','FROZEN JOINT EXPANSION'))
    nb['cells'][1]['source']=lines
    nb['cells'][0]['source']=['# Frozen joint expansion\n','Eight additional annotation-stratified volumes; fixed settings and source hashes. Exploratory validation only.\n']
    (folder/'run.ipynb').write_text(json.dumps(nb))
    meta=json.loads((root/'notebooks/kaggle_division_bearing_diagnostics/kernel-metadata.json').read_text())
    meta.update(id='dushi2/biohub-joint-expansion',title='Biohub Joint Expansion',code_file='run.ipynb')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(selected,indent=2))


if __name__=='__main__':main()
