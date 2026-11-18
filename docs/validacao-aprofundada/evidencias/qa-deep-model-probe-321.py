import json, sys, pathlib, itertools, os
sys.path.insert(0,str(pathlib.Path.cwd()/'src'))
from eficiencia.prever import prever
from eficiencia.artefatos import carregar
import pandas as pd
import numpy as np
p=carregar()
base=json.loads(pathlib.Path('app/veiculo_padrao.json').read_text(encoding='utf8'))
rows=[]
for prop,fuel in itertools.product(['ice','hev','phev','bev'],['gasolina','flex','diesel','electric']):
    if (prop=='bev') != (fuel=='electric'): continue
    for bat,rng in [(12,45),(20,90),(35,140)]:
        rows.append(dict(base,propulsao=prop,combustivel=fuel,bateria_kwh=bat,autonomia_eletrica_km=rng))
df=pd.DataFrame(rows)
out=prever(df,p)
results=pd.concat([df[['propulsao','combustivel','bateria_kwh','autonomia_eletrica_km']],out],axis=1)
results.to_csv(pathlib.Path(os.environ['TEMP'])/'qa-deep-model-probe-321.csv',index=False)
print(results[['propulsao','combustivel','mj_km','kml_etanol','kml_etanol_min','kml_etanol_max','kml_gasolina','kml_diesel']].to_string(index=False))
print("QUANTILES",p.quantis['Plug-in'])
print("PHEV FLEX warnings",out.iloc[21].avisos)
print("Model target attrs",vars(p.modelo).keys())
for k,v in vars(p.modelo).items():
    if isinstance(v,dict): print(k,list(v.keys()))
df.index=[0]*len(df)
try:
    dupl=prever(df,p)
    print('REPEATED INDEX EQ',dupl.reset_index(drop=True).equals(out))
except Exception as e: print('REPEATED INDEX ERROR',repr(e))
