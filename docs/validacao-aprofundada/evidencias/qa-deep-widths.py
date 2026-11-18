from eficiencia.artefatos import carregar
import numpy as np
p=carregar()
for k,qs in p.quantis.items():
    res=np.abs(p.erros_oof[k])
    print(k,len(res),[round(qs[f]*100,3) for f in ('mj_km','mj_km_urbano','mj_km_estrada')],sum(res<=qs['mj_km']),flush=True)
