"""Create separate gate-only branch scripts without editing an active run."""
from pathlib import Path
import hashlib,json
CODE=Path(__file__).resolve().parent
src=CODE/'run_control.py';s=src.read_text()
edits=[
 ("    p.add_argument('--normalize-world', action='store_true')", "    p.add_argument('--normalize-world', action='store_true')\n    p.add_argument('--disable-dynamic-gate', action='store_true')"),
 ("    cfg.mode = 'behave_calibrated_pilot'", "    cfg.mode = 'behave_calibrated_pilot'\n    cfg.validation_disable_dynamic_gate = a.disable_dynamic_gate"),
 ("        return original_dynamic(self, *args, **kwargs)", "        model = original_dynamic(self, *args, **kwargs)\n        if a.disable_dynamic_gate:\n            model.dyn_o_flag.fill_(False)\n        return model"),
 ("        torch.save(state, output/'initial_dynamic.pth')", """        if a.resume_scaffold_dir:
            previous_initial = torch.load(a.resume_scaffold_dir/'initial_dynamic.pth', map_location='cpu', weights_only=False)
            tensor_differences = []
            for key in set(state)|set(previous_initial):
                old, new = previous_initial.get(key), state.get(key)
                if torch.is_tensor(old) and torch.is_tensor(new):
                    if not torch.equal(old.cpu(), new.detach().cpu()): tensor_differences.append(key)
                elif repr(old) != repr(new): tensor_differences.append(key)
            (output/'paired_initialization_check.json').write_text(json.dumps({'changed_state_keys': tensor_differences,
                       'expected_only_change': 'dyn_o_flag', 'source': str(a.resume_scaffold_dir)}, indent=2)+'\\n')
            assert tensor_differences == ['dyn_o_flag'] or set(tensor_differences)=={'dyn_o_flag'}, tensor_differences
        torch.save(state, output/'initial_dynamic.pth')"""),
 ("assert set(differences) <= {'gs_include_fg_in_static'}, differences", "assert set(differences) <= {'validation_disable_dynamic_gate'}, differences")]
for old,new in edits:
    assert s.count(old)==1,old[:70]
    s=s.replace(old,new,1)
dest=CODE/'run_gate_control.py'
if dest.exists():raise FileExistsError(dest)
dest.write_text(s)
pipeline=(CODE/'run_pipeline.py').read_text()
pipeline=pipeline.replace("CODE/'run_control.py'","CODE/'run_gate_control.py'")
pipeline=pipeline.replace("'--normalize-world']","'--normalize-world','--disable-dynamic-gate']")
(CODE/'run_gate_pipeline.py').write_text(pipeline)
(CODE/'gate_adapter_changes.json').write_text(json.dumps({'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
     'output_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'edits':edits,'control':'Only dyn_o flag differs from paired normalized branch; static initialization unchanged'},indent=2)+'\n')
