"""Build a standalone visual comparison using model predictions and local images.

No database, model weights, GPU, or inference dependencies are needed locally.
"""
import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FindMyFit model comparison</title>
<style>
*{box-sizing:border-box}body{font:15px system-ui,sans-serif;margin:0;background:#f4f3ef;color:#202824}main{max-width:1450px;margin:auto;padding:24px}h1{margin:0 0 8px}p{line-height:1.5}.muted{color:#59655e}header,article,.anchor{background:white;border:1px solid #dfe3dd;border-radius:12px;padding:18px;margin-bottom:18px}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:16px 0}select,button{font:inherit;padding:9px;border:1px solid #b5c0b7;border-radius:6px;background:white}select{max-width:100%}button{cursor:pointer}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}.items{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}.outfit{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.card{min-width:0;margin:0;border:1px solid #e0e5df;border-radius:6px;padding:6px}.card img{width:100%;height:150px;object-fit:contain;background:white}.card figcaption{font-size:11px;overflow-wrap:anywhere}.known{border:2px solid #398651}.badge{color:#276739;font-weight:600}.missing{height:150px;background:#f3e6df;display:flex;align-items:center;justify-content:center;padding:8px;font-size:12px}.anchor{display:flex;gap:18px;align-items:center}.anchor .card{width:150px;flex-shrink:0}.partners{display:flex;gap:8px;flex-wrap:wrap}.partners .card{width:100px}.partners .card img{height:100px}h3{margin-bottom:10px}.notice{color:#8b3b24}details{margin-top:12px}@media(max-width:850px){.columns{grid-template-columns:1fr}.card img{height:115px}.anchor{align-items:flex-start}}
</style>
<main><header><h1>FindMyFit model comparison</h1><p>Compare recommendations for the same held-out top. Start with FashionSigLIP and CLIP using shared ranking, then explore the other methods.</p><p class="muted">The outfit preview combines each model’s first trousers and first shoes recommendation. These are independently ranked pieces; the model has not scored the complete outfit. A green border marks a recorded partner, not the only valid match.</p><p id="status" class="muted"></p><div class="controls"><button id="previous">Previous top</button><label>Top <select id="query"></select></label><button id="next">Next top</button></div></header><section id="anchor"></section><div class="columns"><article><label>Left model <select id="left"></select></label><div id="leftResults"></div></article><article><label>Right model <select id="right"></select></label><div id="rightResults"></div></article></div><p class="muted">Scores are similarity values within each model’s space, not compatibility percentages. Score magnitudes are not calibrated across models.</p></main>
<script>
const data=__DATA__;
const byId=id=>document.getElementById(id);
const escapeHtml=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const encoderLabels={clip:'CLIP',fashionsiglip:'FashionSigLIP'};
const methodLabels={cosine:'raw cosine',shared_contrastive:'shared contrastive',shared_projector:'shared ranking',two_tower:'two towers · ranking'};
function card(id,caption='',known=false){const item=data.items[id],src=data.thumbnails[id];return `<figure class="card ${known?'known':''}">${src?`<img loading="lazy" src="${src}" alt="${escapeHtml(item.category+' '+id)}">`:`<div class="missing">Missing local image</div>`}<figcaption>${escapeHtml(caption)}<br>${escapeHtml(id)}${known?'<br><span class="badge">Recorded partner</span>':''}</figcaption></figure>`;}
function results(side,query){const model=byId(side).value,p=query.predictions[model];let html='<h3>Outfit preview</h3><div class="outfit">'+card(query.anchor,'Anchor top');for(const c of ['pants','shoes']){const item=p[c][0].item;html+=card(item,c==='pants'?'Top-ranked trousers':'Top-ranked shoes',query.partners[c].includes(item));}html+='</div>';for(const c of ['pants','shoes']){html+=`<h3>${c==='pants'?'Trousers':'Shoes'} · top 5</h3><div class="items">`;p[c].forEach((r,i)=>html+=card(r.item,`#${i+1} · score ${r.score.toFixed(3)}`,query.partners[c].includes(r.item)));html+='</div>';}byId(side+'Results').innerHTML=html;}
function render(){const q=data.queries[Number(byId('query').value)];byId('anchor').innerHTML='<div class="anchor">'+card(q.anchor,'Anchor top')+'<div><h2>Selected top</h2><p>'+escapeHtml(data.items[q.anchor].image_key)+'</p><details><summary>Recorded outfit partners</summary><div class="partners">'+['pants','shoes'].flatMap(c=>q.partners[c].map(id=>card(id,c,true))).join('')+'</div></details></div></div>';results('left',q);results('right',q);byId('previous').disabled=Number(byId('query').value)===0;byId('next').disabled=Number(byId('query').value)===data.queries.length-1;}
for(const side of ['left','right']){for(const m of data.models){const option=document.createElement('option');option.value=m.id;option.textContent=encoderLabels[m.encoder]+' · '+methodLabels[m.method];byId(side).append(option);}byId(side).value=side==='left'?'fashionsiglip/shared_projector':'clip/shared_projector';byId(side).onchange=render;}
data.queries.forEach((q,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${i+1} / ${data.queries.length} · ${q.anchor}`;byId('query').append(option);});byId('query').onchange=render;byId('previous').onclick=()=>{byId('query').value=Number(byId('query').value)-1;render();};byId('next').onclick=()=>{byId('query').value=Number(byId('query').value)+1;render();};byId('status').textContent=`${data.queries.length} reproducibly sampled tops · training seed ${data.training_seed} · ${data.pool_sizes.pants} trousers and ${data.pool_sizes.shoes} shoes in held-out pools · ${data.missing_images} missing images`;if(data.missing_images)byId('status').className='notice';render();
</script></html>'''


def build_preview(predictions, images_dir, output, queries=12, allow_missing=False):
    data = json.loads(Path(predictions).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported prediction schema')
    data['queries'] = data['queries'][:queries]
    if not data['queries']:
        raise ValueError('No queries available')
    used = set()
    for query in data['queries']:
        used.add(query['anchor'])
        for partners in query['partners'].values(): used.update(partners)
        for predictions_by_category in query['predictions'].values():
            for rows in predictions_by_category.values(): used.update(row['item'] for row in rows)
    root = Path(images_dir).expanduser().resolve()
    thumbnails = {}
    missing = []
    for item in sorted(used):
        key = data['items'][item]['image_key']
        path = (root/key).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Image key escapes image directory: {key}')
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((240,240))
                buffer = io.BytesIO(); image.save(buffer,format='JPEG',quality=80)
            thumbnails[item] = 'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
        except (OSError,ValueError):
            missing.append(key)
            thumbnails[item] = None
    if len(missing) == len(used) and not allow_missing:
        raise ValueError(f'No images could be loaded from {root}. Expected paths such as {missing[:3]}')
    data['items'] = {item:data['items'][item] for item in sorted(used)}
    data['thumbnails'] = thumbnails
    data['missing_images'] = len(missing)
    payload = json.dumps(data,ensure_ascii=True).replace('<',r'\u003c')
    html = TEMPLATE.replace('__DATA__',payload)
    output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(html,encoding='utf-8')
    print(f'Saved {output.resolve()} ({len(data["queries"])} tops, {len(used)-len(missing)} images loaded, {len(missing)} missing).')
    if missing: print('Example missing image keys:',', '.join(missing[:5]))
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,default=ROOT/'examples/model_comparison/predictions.json')
    parser.add_argument('--images-dir',type=Path,required=True,help='Directory containing category folders such as top, pants, and shoes')
    parser.add_argument('--output',type=Path,default=ROOT/'data/model-comparison.html')
    parser.add_argument('--queries',type=int,default=12,help='Number of preselected tops to include (up to 50)')
    parser.add_argument('--allow-missing',action='store_true',help='Allow a placeholder-only diagnostic preview')
    args=parser.parse_args()
    if args.queries<1: parser.error('--queries must be positive')
    try: build_preview(args.predictions,args.images_dir,args.output,args.queries,args.allow_missing)
    except ValueError as exc: parser.error(str(exc))

if __name__=='__main__':main()
