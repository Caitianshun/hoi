"""Private V10 report preparation, embedded DOCX and hash-verified feedback ZIP.

--prepare only assembles measured evidence. Before --docx, a researcher must
record complete-vector and fixed-figure review. Author with bundled python-docx,
then render and inspect every page before setting document_final_QA=passed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import zipfile

RUN = Path(__file__).resolve().parents[1]; ROOT = RUN.parents[2]
KEYS = ["psnr_db", "ssim", "lpips_spatial_mean"]
ARMS = ["U", "Q", "Q0", "PARENT", "C", "P", "PQ"]
REGIONS = [("full", "全图"), ("foreground", "前景"), ("background", "背景")]
PAIRS = ["P-C", "PQ-P", "PQ-C"]


def read(path): return json.loads(Path(path).read_text())


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+"."+str(os.getpid())+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(tmp,path)


def identity(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(2**20),b""):h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def jsonl(path):
    path=Path(path);return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def config():return read(RUN/"configs/v10.json")


def prepare():
    assert read(RUN/"protocol/independent_verification.json")["status"]=="passed"
    jobs=jsonl(RUN/"protocol/GPU_jobs.jsonl");cpu=jsonl(RUN/"protocol/CPU_jobs.jsonl")
    external=read(RUN/"protocol/external_GPU_costs.json") if (RUN/"protocol/external_GPU_costs.json").exists() else dict(GPU_seconds=0,scope="None recorded externally")
    attempts=jsonl(RUN/"protocol/formal_attempts.jsonl");branches=[];models=[];confirmed=0;recoveries=[]
    for scene,cfg in config()["scenes"].items():
        models.append(dict(scene=scene,role="parent",schema=cfg["parent_schema"],historical_updates=cfg["parent_completed_updates"],**identity(ROOT/cfg["parent_checkpoint"])))
        for arm in config()["arms"]:
            directory=RUN/"scenes"/scene/"runs"/arm;receipt=read(directory/"run.json")
            assert receipt["status"]=="completed" and receipt["completed_updates"]==20000
            rows=jsonl(directory/"training_metrics.jsonl");confirmed+=len(rows)
            processes=[j for j in jobs if Path(j["command"][2]).name=="train_scene.py" and "--scene" in j["command"] and j["command"][j["command"].index("--scene")+1]==scene and j["command"][j["command"].index("--arm")+1]==arm]
            assert processes,(scene,arm,"Missing GPU cost ledger")
            whole_peak=len(processes)==1 and processes[0]["returncode"]==0
            recovery_path=directory/"controller_recovery_receipt.json"
            if recovery_path.exists():recoveries.append(read(recovery_path))
            branches.append(dict(scene=scene,arm=arm,completed_updates=receipt["completed_updates"],
                confirmed_logged_Adam_updates=len(rows),formal_attempts=sum(a["scene"]==scene and a["arm"]==arm for a in attempts),
                final_points=receipt["final_points"],base_points=receipt["base_points"],support_points=receipt["support_points"],
                peak_support_points=receipt["peak_support_points"],seconds=sum(j["wall_seconds"] for j in processes),
                engine_final_process_seconds=receipt["seconds"],process_attempts=processes,
                peak_allocated_bytes=receipt["peak_allocated_bytes"] if whole_peak else None,
                peak_reserved_bytes=receipt["peak_reserved_bytes"] if whole_peak else None,
                observed_final_process_peak_allocated_bytes=receipt["peak_allocated_bytes"],
                peak_memory_scope="whole job" if whole_peak else "Whole-job peak unavailable after failure; saved final-process peak is only a lower bound",
                checkpoint=receipt["checkpoint"],actual_first_LR=receipt.get("actual_first_LR"),actual_last_LR=receipt.get("actual_last_LR")))
            models.append(dict(scene=scene,role=arm,completed_updates=20000,**receipt["checkpoint"]))
    actual_attempts=len(attempts);effective_updates=sum(b["completed_updates"] for b in branches)
    assert effective_updates==config()["budgets"]["formal_updates"]
    assert actual_attempts<=config()["budgets"]["formal_attempts"] and len(recoveries)<=6
    unknown=actual_attempts-confirmed;assert unknown>=0
    GPU_seconds=sum(j["wall_seconds"] for j in jobs)+external["GPU_seconds"]
    assert GPU_seconds<=config()["budgets"]["GPU_seconds"]
    costs=dict(GPU_task_seconds=GPU_seconds,GPU_task_hours=GPU_seconds/3600,
        CPU_job_seconds=sum(j["wall_seconds"] for j in cpu),external_GPU_costs=external,GPU_jobs=jobs,CPU_jobs=cpu,
        branches=branches,formal_effective_updates=effective_updates,formal_attempts=actual_attempts,
        confirmed_logged_Adam_updates_including_replay=confirmed,unlogged_attempt_outcomes=unknown,
        replay_attempts=sum(r["replay_attempts"] for r in recoveries),recoveries=recoveries,
        counting="GPU process wall time includes loading, saving, failures, integrated acceptance and final rendering; resource-lock waits and report layout are CPU time. A begun attempt without a completed-update record has unknown Adam outcome.")
    save(RUN/"costs.json",costs);save(RUN/"model_index.json",dict(models=models,
        self_contained_retraining_package=False,large_assets_retained_on_training_host=True))
    conditions={}
    for scene,cfg in config()["scenes"].items():
        directory=RUN/"scenes"/scene/"protocol"
        conditions[scene]=dict(parent=next(m for m in models if m["scene"]==scene and m["role"]=="parent"),
            training_frames=cfg["training_frames"],development_frames=cfg["retained_frames"],
            V9_time_mapping=f"(frame_id-1)/{cfg['time_denominator']}",native_HOSNeRF_time_mapping="frame_id/(N-1); never substituted for V9 time",
            camera_identity="Locked published parent manifest; full-sequence camera-preprocessing provenance unconfirmed",
            canonical_skeleton="Published scene prior; source preprocessing is not asserted train-only",
            extra_inputs=dict(C="Original parent representation; no pose-driven support",P="Training poses, roots, canonical joints, training RGB/mask for seed colors",PQ="Same P inputs; training foreground mask also weights Qr"),
            query=config()["pose_query"],identity_assets=[identity(p) for p in [directory/"input_identity.json",directory/"pose_cache.pt",directory/"support_seed.pt",directory/"RGB_schedule.json"] if p.exists()])
    save(RUN/"input_conditions.json",conditions)
    ev=read(RUN/"evaluation_summary.json");pareto={}
    for scene,value in ev["scenes"].items():
        vectors={a:[value["summaries"][a]["retained"]["full"][k]*(1 if k!="lpips_spatial_mean" else -1) for k in KEYS] for a in ["PARENT","C","P","PQ"]}
        pareto[scene]={a:[b for b in vectors if a!=b and all(x>=y for x,y in zip(vectors[b],vectors[a])) and any(x>y for x,y in zip(vectors[b],vectors[a]))] for a in vectors}
    save(RUN/"protocol/report_inputs.json",dict(status="ready_for_metric_vector_and_visual_review",
        pareto_dominated_by=pareto,metrics=identity(RUN/"evaluation_summary.json"),costs=identity(RUN/"costs.json"),
        terminal_freeze=identity(RUN/"protocol/terminal_freeze.json"),selection_not_automated=True,
        pending=["Review all 80 fixed full/foreground/train panels", "State each scene's complete-vector tradeoffs and retained model", "Build embedded DOCX; render and inspect every page", "Verify private feedback ZIP"],
        pose_query=config()["pose_query"],human_object_metrics="NA; no reliable separate masks",HOSNeRF="Paused by explicit user request; no complete native baseline was produced by this task"))


def helpers():
    path=ROOT/"experiments/foreground_stage_calibration_20260927/run01/code/build_report_docx.py"
    spec=importlib.util.spec_from_file_location("v10_document_helpers",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def docx():
    ev=read(RUN/"evaluation_summary.json");cost=read(RUN/"costs.json")
    review=read(RUN/"protocol/research_decision_review.json");visual=read(RUN/"protocol/visual_review.json")
    verification=read(RUN/"protocol/independent_verification.json");accept=read(RUN/"protocol/module_acceptance.json")
    assert review["status"]==visual["status"]=="completed" and verification["status"]=="passed"
    h=helpers();num=h.num
    class Report(h.Report):
        def __init__(self):
            h.v3.Report.__init__(self,RUN,{},cost)
            self.doc.core_properties.title="姿态驱动补充高斯与区域重建完整结果"
            self.doc.core_properties.subject="V10 全图质量向量、条件与资源取舍"
            for paragraph in self.doc.sections[0].footer.paragraphs:
                for run in paragraph.runs:run.text=run.text.replace("V3 基线校准","V10 完整对照")
        def table(self, rows, widths, size=8.6, left_columns=(0,)):
            table=super().table(rows,widths,size,left_columns)
            # Keep long numeric tables and their explanatory text on one page.
            # Text size stays fixed; only generous cell padding is adjusted.
            from docx.oxml.ns import qn
            for row in table.rows:
                for cell in row.cells:
                    margins=cell._tc.get_or_add_tcPr().find(qn("w:tcMar"))
                    for side in ("top","bottom"):
                        margins.find(qn("w:"+side)).set(qn("w:w"),"40")
            return table
    r=Report();r.p("姿态驱动补充高斯与区域重建完整结果","Title");r.p("V10  Backpack 与 Tennis  逐场景优化","Subtitle")
    for paragraph in review["summary_paragraphs"]:r.p(paragraph)
    r.p("两场景各C、P、PQ新增20000次Adam，共120000次有效更新。全部六终态先冻结，再统一评价；未经续训父模型另列。主目标为全图PSNR、SSIM与LPIPS质量向量，接受有意义的取舍，不设前景或几何硬门槛。")
    table=[["场景","设置","全图 PSNR ↑","全图 SSIM ↑","LPIPS ↓"]]
    for scene,value in ev["scenes"].items():
        for arm in ["PARENT","C","P","PQ"]:table.append([scene,arm,*[num(value["summaries"][arm]["retained"]["full"][k],6) for k in KEYS]])
    r.table(table,[1.12,.8,1.69,1.69,1.69],10)
    r.p("U、Q、Q0为历史评价结果，保留其来源哈希与训练预算差异。Backpack父模型选择V9 Q0，Tennis选择V8 Q，是已经开发过的场景级选择，不是独立盲测。C→P同时增加训练姿态先验、初始化与支撑容量。")
    r.page("问题 表示与训练信息")
    r.p("峰值信噪比PSNR把像素均方误差换算为dB，越高越好；结构相似性SSIM比较局部亮度、对比度和结构，越高越好。学习感知图像块相似度LPIPS比较预训练网络特征差异，越低越好。本轮同时报告三项，避免用一种指标掩盖另一种质量变化。")
    r.p("运动主体仅占画面的一小部分，但其像素误差可能占据相当比例。已有全场自由形变保留对背景、背包、球拍和衣物的解释能力；新增P用关节骨架承担人体粗运动，让小网络主要拟合剩余形变。该机制的作用须由完整配对结果判断。")
    r.p("例如Tennis中人物从球场走向长凳。自由形变基座直接由位置和时间预测点的运动，某些留出帧中人体变成模糊团块。P先把新增点放在人体骨段附近，再随肩、肘、髋等关节移动。训练RGB与联合渲染的差异反向更新点的外观、位置和小网络，使其补足观测人体；姿态与四骨权重保持固定。背包与球拍仍由原自由基座解释，本轮没有给它们新增独立物体节点。")
    r.p("规范域指人体参考姿态下的坐标。bank指一组具有独立生命周期的高斯点；高斯协方差描述点在三维中的椭球形状和方向。线性混合蒙皮LBS以固定权重对四个骨变换求加权和，使一个点可同时随相邻骨段移动。多层感知机MLP是小型全连接网络，本轮用规范位置和时间预测粗运动之外的自由残差。raster指高斯投影和像素合成过程。")
    r.p("P从24关节骨段体积中建立20000个弱形状种子，规范位置、尺度、旋转、透明度和颜色可学习；固定四骨非负权重给出加权骨变换。小MLP预测规范位置、尺度和旋转残差。中心和协方差均转换到父模型世界坐标，两bank一次联合深度排序与raster，没有按mask贴图。")
    r.p("原base点拓扑固定，所有属性和旧形变网络仍可学。新bank独立增密，完成500–8000次更新期间每100次after-Adam处理clone、split和prune，最多60000点。其屏幕梯度先按每视图取范数，再以可见次数归一化；它不是完整AbsGS。")
    r.page("优化目标与查询条件")
    r.p("三臂采用同一场景父状态、fresh Adam、预生成RGB批次顺序和base学习率；P/PQ使用完全相同的新bank初态。C/P目标Q为0.8 L1加0.2乘以1−SSIM11。PQ用0.9 Q_full加0.1 Q_FG；先在完整RGB计算SSIM图，再按原训练前景mask求区域平均，空mask退回全图。")
    r.p("P/PQ优化及轨迹查询只使用训练帧姿态；查询时局部关节旋转和根旋转以四元数SLERP插值，局部与根平移线性插值，根正尺度在log空间插值。区间外取最近训练姿态，因此00000为边界近似。推理不读查询mask，不读查询人体根变换绕过留出。")
    r.p("相机和世界规范锁定父manifest。发布规范骨架属于场景先验，其全序列预处理来源与相机估计信息边界尚未完全证明train-only。模型时间为(f−1)/282或(f−1)/299，经frame_id与原生HOSNeRF时间区分；不能将不同查询条件的模型混成无标注排行榜。")
    conditions=read(RUN/"input_conditions.json")
    table=[["场景","训练帧","开发帧","父状态","父实际Adam"]]
    for scene,value in conditions.items():
        table.append([scene,value["training_frames"],value["development_frames"],value["parent"]["schema"],value["parent"]["historical_updates"]])
    r.table(table,[1.39,1.2,1.2,1.6,1.6],10)
    r.p("Backpack父状态为V9 Q0，Tennis为V8 Q。Tennis路径名义014000对应13999次实际Adam，旧文件标记不可直接恢复；本轮仅导入权重，并为三臂建立全新Adam、RNG和固定采样位置，未继续旧优化器或伪造第14000次更新。父状态与六个新终态的路径、大小、完整SHA见model_index.json。")
    for scene,value in ev["scenes"].items():
        for split,label in [("retained","开发16帧"),("train","完整训练集"),("retained_without_00000","去00000补充15帧")]:
            r.page(scene+" "+label+" 质量向量")
            table=[["区域","设置","PSNR dB ↑","SSIM ↑","LPIPS ↓"]]
            for region,name in REGIONS:
                for arm in ARMS:table.append([name,arm,*[num(value["summaries"][arm][split][region][k],6) for k in KEYS]])
            r.table(table,[.8,.8,1.79,1.79,1.81],9.7)
            r.p("前景沿用发布合并mask。没有可靠独立人、物mask，相应指标为NA；图像质量不直接证明几何、接触、真实深度或物体独立运动。")
            if split=="retained":
                r.page(scene+" 开发结果解释")
                for paragraph in review["scene_interpretation"][scene]:r.p(paragraph)
                table=[["设置","训练全图 PSNR","开发全图 PSNR","训练减开发 dB"]]
                for arm in ["C","P","PQ"]:
                    train=value["summaries"][arm]["train"]["full"]["psnr_db"]
                    dev=value["summaries"][arm]["retained"]["full"]["psnr_db"]
                    table.append([arm,num(train,6),num(dev,6),num(train-dev,6)])
                r.table(table,[.8,2.05,2.05,2.09],10)
                r.p("这一差距只描述不同训练与开发图像集合的拟合差异，包含相机视角与时刻变化。不能将它单独解释成纯时间泛化误差或过拟合的因果证明。")
        r.page(scene+" 三组核心配对差")
        table=[["比较","区域","指标","均值差","中位差","胜 平 负"]]
        for pair in PAIRS:
            for region,name in REGIONS:
                for key,label in zip(KEYS,["PSNR","SSIM","LPIPS"]):
                    d=value["paired"][pair]["retained"][region][key]
                    table.append([pair,name,label,num(d["mean"],6,True),num(d["median"],6,True),f"{d['win']} {d['tie']} {d['loss']}"])
        r.table(table,[.8,.65,.65,1.5,1.5,1.89],9)
        r.p("LPIPS负差有利，胜平负统一按有利方向。全部训练、15帧补充、相对父模型与历史U/Q/Q0配对，以及四个连续时间块保留于CSV/JSON；连续视频帧不作独立场景显著性承诺。")
    r.page("等场景汇总与选择")
    table=[["设置","全图 PSNR","全图 SSIM","全图 LPIPS"]]
    for arm in ARMS:table.append([arm,*[num(ev["equal_scene_means"][arm]["retained"]["full"][k],6) for k in KEYS]])
    r.table(table,[1.2,1.93,1.93,1.93],10)
    r.p("先计算每场景逐帧宏平均，再将两个场景等权平均。允许场景级配置，但不按单帧选最好分支，不按不同指标拼接不存在的模型。")
    for paragraph in review["selection_paragraphs"]:r.p(paragraph)
    r.page("工程验收 恢复与实际预算")
    r.p("一次集成验收实际记录为："+str(accept.get("diagnostic_Adam_updates",accept.get("actual_Adam_updates","NA")))+"次Adam、"+str(accept.get("extra_no_update_backwards","NA"))+"次额外不更新反向。其坐标、数据、联合渲染、梯度、拓扑与恢复范围以module_acceptance.json为准。")
    for paragraph in review.get("engineering_incidents",[]):r.p(paragraph)
    r.p(f"正式有效更新{cost['formal_effective_updates']}，实际尝试{cost['formal_attempts']}，含重放的日志确认Adam{cost['confirmed_logged_Adam_updates_including_replay']}；另{cost['unlogged_attempt_outcomes']}次已开始尝试缺少完成记录，Adam执行情况未知。状态恢复{len(cost['recoveries'])}次，重放尝试{cost['replay_attempts']}；恢复成功不代表根因已修复。")
    r.p(f"GPU任务进程累计{num(cost['GPU_task_hours'],6)}小时，含加载、保存、验收、失败与最终渲染；资源等待和文档排版另计。仅使用物理GPU1 RTX3090，同硬件完成六臂。本轮没有重启用户已暂停的HOSNeRF，完整原生基线仍缺失。")
    table=[["场景","设置","base点数","support终态","support峰值","进程秒","峰值 GiB"]]
    for b in cost["branches"]:table.append([b["scene"],b["arm"],b["base_points"],b["support_points"],b["peak_support_points"],num(b["seconds"],2),"NA" if b["peak_allocated_bytes"] is None else num(b["peak_allocated_bytes"]/(1<<30),3)])
    r.table(table,[1.05,.5,1.1,1.1,1.1,1.14,1],8.8)
    r.p("分支秒数包含失败与恢复前后进程。缺失崩溃前峰值时全程峰值为NA，恢复尾段观测仅为下界。完整状态包含两bank、骨变换与规范尺度、四骨权重、MLP、Adam、LR、SH、密度缓冲、RNG和样本位置。")
    r.page("共同评价与核验范围")
    r.p(f"独立核验{verification['rows']}指标行、{verification['paired_rows']}配对行，从浮点渲染重算{verification['PSNR_recomputed']}个新PSNR，最大差{verification['max_PSNR_difference']:.3g}。SSIM/LPIPS只独立复核聚合与配对，未声称第二实现重新评价。")
    r.p("评价RGB先clip至[0,1]，PSNR为逐帧dB宏平均，pooled与raw另列。SSIM7使用非Gaussian窗口与sample covariance，与训练SSIM11区分；LPIPS采用AlexNet0.1、spatial=True、normalize=True，区域从完整feature map聚合，不以放大裁剪替代。")
    r.table([["核验项","记录数","实际范围"],
        ["指标行",verification["rows"],"包含历史与新结果"],
        ["配对差行",verification["paired_rows"],"逐帧对应与汇总"],
        ["新PSNR重算",verification["PSNR_recomputed"],"浮点RGB与原GT"],
        ["正式完成Adam",cost["confirmed_logged_Adam_updates_including_replay"],"与开始attempt和采样日程一致"],
        ["固定比较图",visual["fixed_figure_comparisons_reviewed"],"两场景400个方法分栏"]],
        [2.1,1.29,3.6],10)
    r.p(f"CPU评价与独立数值核验进程共{num(cost['CPU_job_seconds'],2)}秒，文档排版与目检另计。所有新分数由同一评价器计算，开发图均在六终态冻结后读取；工程验收没有提前按短训质量挑选配置。历史U/Q/Q0按原冻结分数和身份索引复用，没有冒称重新训练或统一输入。")
    r.p("反馈包中的metrics_per_frame.csv、paired_differences.csv、evaluation_summary.json与costs.json保留可追溯数值。protocol/final_numeric_audit.json记录独立逐行、日程及哈希复核；protocol/final_protocol_audit.json记录额外输入、拓扑与终态来源边界。原始完整浮点数组与模型留在训练机。")
    for scene in ev["scenes"]:
        figures=read(RUN/"scenes"/scene/"figure_manifest.json")["figures"]
        for split,kind,label,columns in [("retained","full","开发固定全图",1),("retained","foreground_crop","开发固定前景裁剪",2),("train","full","原固定训练图",1)]:
            ids=sorted({f["frame_id"] for f in figures if f["split"]==split and f["kind"]==kind})
            assert len(ids)==(16 if split=="retained" else 8)
            for start in range(0,len(ids),4):
                selected=ids[start:start+4];r.page(scene+" "+label+" "+str(start//4+1))
                r.p("每组从左至右为GT、父模型、C、P、PQ。沿用原固定帧与局部框，未按输出误差重新选样；指标在原分辨率计算，显示统一缩放。","Caption")
                items=[f for fid in selected for f in figures if f["split"]==split and f["frame_id"]==fid and f["kind"]==kind]
                r.image(r.sheet(scene+"_"+split+"_"+kind+"_"+str(start),items,columns,True),"固定帧 "+", ".join(selected),max_height=7.7)
    r.page("实际图像观察与贡献边界")
    for paragraph in visual["findings"]:r.p(paragraph)
    for paragraph in review["limitations"]:r.p(paragraph)
    for paragraph in review["next_step"]:r.p(paragraph)
    r.p("P检验初始化、人体粗运动、自由修正与额外容量的完整组合，未独立证明每个因素。LBS、区域加权与高斯支撑来自已有机制迁移，不宣称首次提出。全图收益也不自动证明球拍或背包已独立建模。")
    r.p("11月4日北京时间23:00前冻结核心证据；11月5日至15日保留连续11个完整自然日写作。剩余场景扩展须围绕有效完整配置和明确父模型生成方式另立协议。本轮未开展第二种子、额外权重扫描或新架构。")
    path=RUN/"output/V10_pose_support_results.docx";path.parent.mkdir(parents=True,exist_ok=True);r.doc.save(path)
    with zipfile.ZipFile(path) as z:
        media=[n for n in z.namelist() if n.startswith("word/media/")]
        external=[n for n in z.namelist() if n.endswith(".rels") and b'TargetMode="External"' in z.read(n)]
        assert media and not external
    save(RUN/"protocol/document_authored.json",dict(status="authored_pending_every_page_render_QA",document=identity(path),embedded_images=len(media),external_relationship_files=external))


def package():
    document=RUN/"output/V10_pose_support_results.docx";qa=read(RUN/"protocol/document_final_QA.json")
    assert qa["status"]=="passed" and qa["document"]["sha256"]==identity(document)["sha256"]
    assert read(RUN/"protocol/research_decision_review.json")["status"]=="completed"
    files={}
    def add(path,name=None):
        path=Path(path)
        if path.is_file():files[name or path.relative_to(RUN).as_posix()]=path
    for name in ["metrics_per_frame.csv","paired_differences.csv","evaluation_summary.json","costs.json","model_index.json","input_conditions.json","NEXT_DECISION.md","REPRODUCE.md"]:add(RUN/name)
    for folder in ["protocol","scenes","diagnostics","logs","code","configs"]:
        for path in (RUN/folder).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:continue
            if path.suffix in [".py",".json",".jsonl",".csv",".md",".log",".txt"]:add(path)
            elif path.suffix in [".jpg",".png"] and "figures" in path.parts:add(path)
            elif path.suffix==".npz" and "feedback_arrays" in path.parts:add(path)
    for name in ["pose_prior_adapter.py","pose_support_gaussians.py","region_reconstruction.py","projected_motion.py"]:add(ROOT/"hoi_modules"/name,"source/hoi_modules/"+name)
    dependencies={
        "baseline_protocol_calibration_20260927":["adapter_4dgs.py","evaluate_frozen.py","build_report_docx.py","summarize_existing.py"],
        "numerical_stability_calibration_20260927":["common.py","evaluate_and_report.py"],
        "foreground_stage_calibration_20260927":["build_report_docx.py"],
        "aux_ref_object_reconstruction_20260924":["evaluate_aux.py"]}
    for experiment,names in dependencies.items():
        for name in names:
            path=ROOT/"experiments"/experiment/"run01/code"/name
            add(path,"source/"+path.relative_to(ROOT).as_posix())
    for scene,cfg in config()["scenes"].items():
        for name in ["manifest.json","evaluation_manifest.json"]:add(ROOT/cfg["input_dir"]/name,"inputs/"+scene+"/"+name)
        add(ROOT/cfg["historical_dir"]/"protocol/fixed_examples.json","inputs/"+scene+"/fixed_examples.json")
    add(document,document.name)
    index={name:dict(bytes=path.stat().st_size,sha256=identity(path)["sha256"]) for name,path in sorted(files.items())}
    path=RUN/"output/V10_feedback.zip";tmp=path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,source in sorted(files.items()):z.write(source,name)
        z.writestr("feedback_index.json",json.dumps(dict(members=index,
            excluded=["Full model/Adam checkpoints","Original RGB/masks","Full float render caches","Third-party source and environments"],
            reproduction_scope="Auditable private evidence/code bundle, not a self-contained retraining package; paths/bytes/SHA identify large assets on training host"),ensure_ascii=False,indent=2))
    os.replace(tmp,path)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        assert set(z.namelist())==set(index)|{"feedback_index.json"}
        for name,record in index.items():
            data=z.read(name);assert len(data)==record["bytes"] and hashlib.sha256(data).hexdigest()==record["sha256"]
    save(RUN/"feedback_verification.json",dict(status="passed",zip=identity(path),members=len(index)+1,content_hashes_checked=len(index),CRC_verified=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true");group.add_argument("--docx",action="store_true");group.add_argument("--package",action="store_true")
    args=parser.parse_args();prepare() if args.prepare else docx() if args.docx else package()
