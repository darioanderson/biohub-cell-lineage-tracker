"""Download only selected Biohub Zarr chunks. Never invokes download-all.

Requires authorized Kaggle API access and accepted competition rules.
Stage 1 defaults to 8 frames per sample, two embryo-disjoint samples, 512 MiB.
Missing Zarr chunks MUST NOT be interpreted as actual zero-valued observations.
"""
import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[2]
COMPETITION = 'biohub-cell-tracking-during-development'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def select(files, frames, budget):
    """Select complete annotation folders and image metadata/chunks from observed file names."""
    groups = {}
    for item in files:
        match = re.fullmatch(r'train/([^/]+)\.(zarr|geff)/(.*)', item['name'])
        if not match:
            continue
        name, kind, relative = match.groups()
        include = kind == 'geff' or relative.endswith('zarr.json')
        chunk = re.fullmatch(r'0/c/(\d+)/0/0/0', relative)
        if chunk and int(chunk[1]) in frames:
            include = True
        if include:
            groups.setdefault(name, []).append(item)
    selected, samples, embryos = [], [], set()
    for name, items in sorted(groups.items()):
        embryo = name.split('_')[0]
        names = {f['name'] for f in items}
        required = {f'train/{name}.zarr/0/zarr.json', f'train/{name}.geff/zarr.json'}
        required |= {f'train/{name}.zarr/0/c/{t}/0/0/0' for t in frames}
        if embryo in embryos or not required <= names:
            continue
        size = sum(f['bytes'] for f in selected+items)
        if size > budget:
            continue
        selected.extend(items)
        samples.append({'sample': name, 'embryo': embryo,
                        'role': 'train' if not samples else 'validation', 'frames': frames})
        embryos.add(embryo)
        if len(samples) == 2:
            break
    if len(samples) != 2:
        raise ValueError('Cannot find two complete embryo-disjoint subsets within the cap; no data downloaded')
    return {'competition': COMPETITION, 'samples': samples, 'files': selected,
            'planned_bytes': sum(f['bytes'] for f in selected), 'budget_bytes': budget,
            'partial_volumes': True, 'warning': 'Read only frames listed in this manifest. Unavailable chunks are not zero images.'}


def fetch(api, item, root):
    relative = PurePosixPath(item['name'])
    if relative.is_absolute() or '..' in relative.parts or '\\' in item['name']:
        raise ValueError('Unsafe remote file path')
    dest = root.joinpath(*relative.parts)
    receipt = dest.with_name(dest.name+'.download.json')
    if dest.exists() and receipt.exists():
        prior = json.loads(receipt.read_text())
        if hashlib.sha256(dest.read_bytes()).hexdigest() == prior['sha256']:
            return prior
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=PROJECT/'work') as stage:
        api.competition_download_file(COMPETITION, item['name'], path=stage, quiet=True)
        downloaded = [p for p in Path(stage).rglob('*') if p.is_file()]
        if len(downloaded) != 1:
            raise RuntimeError(f'Unexpected download layout for {item["name"]}')
        src = downloaded[0]
        if zipfile.is_zipfile(src):
            with zipfile.ZipFile(src) as archive:
                entries = [x for x in archive.infolist() if not x.is_dir()]
                if len(entries) != 1 or entries[0].file_size != item['bytes']:
                    raise ValueError('Unexpected archive content or size')
                with archive.open(entries[0]) as source, dest.with_suffix(dest.suffix+'.part').open('wb') as target:
                    shutil.copyfileobj(source, target)
            dest.with_suffix(dest.suffix+'.part').replace(dest)
        else:
            if src.stat().st_size != item['bytes']:
                raise ValueError('Download size differs from catalog; refusing unverified content')
            shutil.copyfile(src, dest)
    result = {'name': item['name'], 'bytes': dest.stat().st_size,
              'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    save(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--max-mib', type=int, default=512)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.frames <= 100 or args.max_mib < 1:
        raise ValueError('Invalid frame count or size cap')
    root = PROJECT/'data'/'biohub-stage1'
    root.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle import api
    except OSError:
        save(root/'status.json', {'state': 'authentication_required', 'downloaded_bytes': 0,
                                 'message': 'Kaggle API credentials are not configured. No competition data downloaded.'})
        raise SystemExit('Kaggle API authentication is required; no data downloaded.')
    catalog = root/'catalog.json'
    if catalog.exists():
        state = json.loads(catalog.read_text())
    else:
        state = {'files': [], 'next_page_token': None, 'complete': False}
    while not state['complete']:
        result = api.competition_list_files(COMPETITION, page_token=state['next_page_token'], page_size=200)
        state['files'].extend({'name': f.name, 'bytes': f.total_bytes} for f in result.files)
        state['next_page_token'] = result.next_page_token
        state['complete'] = not bool(result.next_page_token)
        save(catalog, state)
        print(f'Catalogued {len(state["files"])} file names; image data not yet downloaded.', flush=True)
    plan = select(state['files'], list(range(args.frames)), args.max_mib*1024**2)
    save(root/'subset-manifest.json', plan)
    print(f'Planned {plan["planned_bytes"]/1024**2:.1f} MiB in {len(plan["files"])} files')
    if args.download:
        receipts = []
        for item in plan['files']:
            receipts.append(fetch(api, item, root))
            save(root/'status.json', {'state': 'downloading', 'complete_files': len(receipts),
                                     'total_files': len(plan['files'])})
        save(root/'status.json', {'state': 'download_complete', 'receipts': receipts,
                                 'downloaded_bytes': sum(x['bytes'] for x in receipts)})


if __name__ == '__main__':
    main()
