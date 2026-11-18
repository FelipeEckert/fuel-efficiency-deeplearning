import sys,pathlib,json,os
sys.path.insert(0,str(pathlib.Path.cwd()/'src'))
import numpy as np,pandas as pd
from eficiencia.artefatos import carregar
from eficiencia.prever import prever
from eficiencia.fisica import classe_inercia
p=carregar()
print('ranges',p.faixas.keys())
for n,m in p.modelo.estimadores.items():
    b=m.get_booster()
    print(n,'tree count',len(b.get_dump()),'split counts',b.get_score(importance_type='weight'))
base=json.loads(pathlib.Path('app/veiculo_padrao.json').read_text(encoding='utf8'))
rg=np.random.default_rng(1152)
rows=[]
for i in range(240):
    mass=rg.uniform(1600,2600)
    rows.append(dict(base,propulsao='phev',combustivel='flex',categoria='SUVG',bateria_kwh=rg.uniform(14,35),autonomia_eletrica_km=rg.uniform(45,95),massa_om_kg=mass,classe_inercia_kg=classe_inercia(mass),pbt_kg=mass+500,f0_n=rg.uniform(120,200),f2_n_kmh2=rg.uniform(.035,.06),potencia_kw=rg.uniform(125,275),cilindrada_cm3=rg.uniform(1500,2500)))
x=pd.DataFrame(rows);y=prever(x,p)
for n in ['mj_km','kml_gasolina','kml_etanol','kml_etanol_urbano','kml_etanol_estrada']:
    print(n,y[n].nunique(),y[n].min(),y[n].max())
print('no warnings',y.avisos.map(len).eq(0).sum(),y.avisos.iloc[0])
pd.concat([x,y],axis=1).to_csv(pathlib.Path(os.environ['TEMP'])/'qa-deep-model-support-321.csv',index=False)
