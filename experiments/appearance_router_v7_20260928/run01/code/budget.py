"""Count actual rasterizer invocations, not high-level probe labels."""
from common import *
import time
class CallBudget:
    def __init__(self,label):
        import diff_gaussian_rasterization as backend
        self.label=label;self.mode='diagnostic';self.path=RUN/'protocol/kernel_calls.jsonl'
        self.used={'render':0,'backward':0};cfg=read(RUN/'configs/v7.json')['budgets'];self.limits={'render':cfg['diagnostic_renders'],'backward':cfg['diagnostic_backwards']}
        if self.path.exists():
            for s in self.path.read_text().splitlines():
                x=json.loads(s)
                if x['mode']=='diagnostic':self.used[x['kind']]+=1
        self.f=self.path.open('a',buffering=1)
        for name,kind in [('rasterize_gaussians','render'),('rasterize_gaussians_backward','backward')]:
            fn=getattr(backend._C,name)
            def wrapper(*args,_fn=fn,_kind=kind):
                if self.mode=='diagnostic':
                    assert self.used[_kind]<self.limits[_kind],f'{_kind} diagnostic budget exhausted'
                    self.used[_kind]+=1
                self.f.write(json.dumps(dict(label=self.label,kind=_kind,mode=self.mode,time_unix=time.time()))+'\n')
                return _fn(*args)
            setattr(backend._C,name,wrapper)
