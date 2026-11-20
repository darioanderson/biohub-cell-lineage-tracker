"""Export one real frame and an unreviewed background record, without invented labels."""
import argparse,json
from pathlib import Path
import tifffile
from biohub.data import read_zarr_meta,load_volume
from biohub.background_review import review_template

def main():
    p=argparse.ArgumentParser()
    p.add_argument('volume',type=Path);p.add_argument('--frame',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();shape,dtype=read_zarr_meta(a.volume)
    if not 0<=a.frame<shape[0]:raise ValueError('Frame outside volume')
    sample=a.volume.name.removesuffix('.zarr');a.output.mkdir(parents=True,exist_ok=True)
    stem=a.output/f'{sample}_{a.frame}'
    if stem.with_suffix('.json').exists() or stem.with_suffix('.tif').exists():
        raise FileExistsError('Review package already exists; refusing to overwrite')
    frame=load_volume(a.volume,a.frame,shape,dtype)
    tifffile.imwrite(stem.with_suffix('.tif'),frame,metadata={'axes':'ZYX'})
    stem.with_suffix('.json').write_text(json.dumps(review_template(frame,sample,a.frame),indent=2))
    print('UNREVIEWED:',stem.with_suffix('.json'))

if __name__=='__main__':main()
