from pathlib import Path
import json, re, shutil, hashlib, os
E=Path('/home/cai_tianshun/Project/HOI/experiments/structured_hoi_20260923')
inventory=json.loads((E/'docx_delivery_qa/source_inventory.json').read_text())
root=E/'portable_report'
materials=root/'report_materials'
originals=materials/'sources_original'
materials.mkdir(parents=True,exist_ok=True);originals.mkdir(exist_ok=True)
source=E/'REPORT.md'
assert hashlib.sha256(source.read_bytes()).hexdigest()==inventory['sha256']
path_map={str(source):'report_materials/REPORT_portable.md'}
labels={str(source):'主报告可移植 Markdown 来源'}
for i,row in enumerate(inventory['links'],1):
    src=Path(row['target'])
    path_map[str(src)]=f'report_materials/R{i:02d}_{src.name}'
    labels[str(src)]=row['label']
idx=0
for group in inventory['supporting_markdown_image_dependencies']:
    for row in group['inline_images']:
        src=Path(row['target'])
        if str(src) not in path_map:
            idx+=1
            path_map[str(src)]=f'report_materials/I{idx:02d}_{src.name}'
            labels[str(src)]=row['label']
links_pattern=re.compile(r'(!?)\[([^\]]*)\]\(([^\n]+?)\)')
unsupported=[]
rewritten=[]
original_entries=[]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def copy_exact(src,dst):
    if dst.exists() and dst.read_bytes()!=src.read_bytes():
        raise RuntimeError(f'Refuse to overwrite different file: {dst}')
    shutil.copy2(src,dst)
    assert sha(src)==sha(dst)

def portable_markdown(src,dst):
    text=src.read_text()
    def change(match):
        bang,label,target=match.groups()
        if target.startswith(('https://','http://','mailto:','#')):
            return match.group(0)
        p=Path(target)
        if not p.is_absolute():p=(src.parent/p).resolve()
        k=str(p)
        if k in path_map:
            rel=os.path.relpath(root/path_map[k],dst.parent)
            rewritten.append({'source_document':str(src),'source_target':k,'portable_document':str(dst.relative_to(root)),'portable_target':rel})
            return f'{bang}[{label}]({rel})'
        unsupported.append({'source_document':str(src),'source_target':k,'label':label,'source_exists':p.exists()})
        return f'{label}（原主机路径：`{target}`；未随包携带）'
    dst.write_text(links_pattern.sub(change,text),encoding='utf-8')

for absolute,relative in path_map.items():
    src=Path(absolute);dst=root/relative
    assert src.is_file(),src
    if src.suffix.lower()=='.md':
        orig=originals/('REPORT.md' if src==source else dst.name)
        copy_exact(src,orig)
        original_entries.append({'source':str(src),'archive':str(orig.relative_to(root)),'sha256':sha(src),'bytes':src.stat().st_size})
        portable_markdown(src,dst)
    else:copy_exact(src,dst)
(root/'path_map.json').write_text(json.dumps(path_map,ensure_ascii=False,indent=2)+'\n')
manifest={'scope':'Reading companion only; not a complete experiment reproduction archive','source_report_sha256':sha(source),'path_map_file':'path_map.json','assets':[{'source':a,'portable_path':r,'label':labels[a],'source_sha256':sha(Path(a)),'portable_sha256':sha(root/r),'bytes':(root/r).stat().st_size,'markdown_links_rewritten':Path(a).suffix.lower()=='.md'} for a,r in path_map.items()],'exact_original_markdown':original_entries,'rewritten_local_links':rewritten,'omitted_nested_local_links':unsupported,'counts':{'main_report_originals':1,'support_markdown_originals':9,'first_level_explicit_assets':16,'nested_inline_images':idx,'unique_portable_assets':len(path_map),'exact_original_markdown':len(original_entries),'rewritten_local_links':len(rewritten),'omitted_nested_link_occurrences':len(unsupported)},'notes':['Existing remote URLs are preserved solely as citations; no remote content was downloaded.','Raw original Markdown intentionally preserves original paths and equations for provenance; use portable copies for relative image references.','Images and videos are copied byte-for-byte. DOCX embedding/visual validation is completed separately by the main document task.']}
(materials/'asset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
readme='''# 结构化人—物—场景报告：可移植阅读包\n\n本包用于在另一台主机阅读 2026-09-23 的首轮 S0/S1 报告，不是训练环境或完整实验复现归档。\n\n## 阅读入口\n\n- `REPORT.docx`：Word 主阅读文档。主报告正文、表格及报告关键图片将由文档生成流程内嵌；复制该文件即可阅读正文。\n- `report_materials/REPORT_portable.md`：完整 Markdown 来源的可移植副本。主报告中的全部显式链接均已指向本包相对路径。\n- 两条原报告直接链接的视频、支撑 Markdown、JSON 和图片位于 `report_materials`。若需要视频与支撑材料，请转移整个文件夹或完整 ZIP；仅复制 DOCX 不等于同时携带动态视频。\n- `path_map.json`：原主机绝对路径 → 本包相对路径的对照。\n- `report_materials/asset_manifest.json`：文件大小、SHA-256、原始文档身份、链接改写与未随包携带的嵌套链接清单。\n\n## 内容与边界\n\n保留主报告直接引用的全部 14 个链接目标、正文 2 张图片，以及支撑 Markdown 正文嵌入的另 9 张图片。所有图片和两条 MP4 视频均按原字节复制。支撑文档中指向已携带文件的链接改为相对路径；指向未携带代码、检查点、逐帧几何、数据、其他 JSON/NPZ/CSV 等的嵌套链接则保留标签和原路径，并明确标记“未随包携带”。这些路径是来源记录，不能在另一台主机直接访问。\n\n本包没有递归复制训练数据、模型权重、检查点、执行环境或所有中间结果。因此可以独立阅读正文和本包所列支撑材料，不保证在新主机直接重新运行实验。原文中提到但没有直接链接的其他 S1/查询视频、逐帧 `geometry`、`gaussian_identity.npz`、`object_pose.npz` 也不在当前复制范围。\n\n已有网页 URL 仅作为原文引用保留；本次未抓取网页。执行指导原件仍含原始 LaTeX，属于原始来源材料，其 Markdown 数学显示依赖阅读工具；主报告本身没有公式块。\n\n## 原始文档保全\n\n`report_materials/sources_original/` 保存主报告及 9 个直接引用 Markdown 的逐字节原件，SHA-256 已核验。这里的原件故意保留原路径用于追踪；实际阅读图片请使用上一层改为相对路径的版本。可移植副本只改写本地链接，正文叙述、数值与结论不变。\n\n## 文件索引\n\n'''
for absolute,relative in path_map.items():
    readme+=f'- [{labels[absolute]}]({relative})（{(root/relative).stat().st_size:,} 字节）\n'
(root/'README.md').write_text(readme,encoding='utf-8')
# Validate every rewritten relative link points to a packaged file, and every media byte is original.
checks=[]
for absolute,relative in path_map.items():
    dst=root/relative
    if dst.suffix.lower()=='.md':
        for m in links_pattern.finditer(dst.read_text()):
            target=m.group(3)
            if not target.startswith(('https://','http://','mailto:','#')):
                checks.append({'document':relative,'target':target,'exists':(dst.parent/target).is_file()})
assert checks and all(x['exists'] for x in checks)
qa={'status':'passed','portable_asset_count':len(path_map),'exact_original_markdown_count':len(original_entries),'all_relative_links_exist':True,'relative_link_checks':checks,'total_bytes_without_docx':sum(x.stat().st_size for x in root.rglob('*') if x.is_file())}
(E/'docx_delivery_qa/portable_materials_checks.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'passed','root':str(root),'path_map':str(root/'path_map.json'),'counts':manifest['counts'],'total_bytes_without_docx':qa['total_bytes_without_docx']},ensure_ascii=False,indent=2))
