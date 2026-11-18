from pathlib import Path
from openpyxl import Workbook
import os
root = Path(os.environ['TEMP'])/'fuel-ui-20261118'
head = ['Name','Volume','Known MJ/km']
def put(name, rows, other=None):
    w=Workbook()
    w.active.title='First'
    for r in rows: w.active.append(r)
    if other:
        s=w.create_sheet('Second')
        for r in other: s.append(r)
        w.active=1
    w.save(root/f'qa-deep-xlsx-{name}.xlsx')
put('baseline',[head,['Baseline',100,2]])
put('names',[head,['0012',100,1],['0142',100,3]])
put('na-name',[head,['NA',100,1],['N/A',100,3],['Normal',100,2]])
put('na-mj',[head,['Invalid text',100,'NA'],['Baseline',100,2]])
put('mixed-text-mj',[head,['Brazil 1',100,'1,25'],['Brazil 2',300,'2,75']])
put('numeric-volume',[head,['Fraction',1.125,1],['Control',100,3]])
put('numeric-volume-positive',[head,['Fraction',0.125,1],['Control',100,3]])
put('sheets-valid',[head,['First sheet',100,1]],[head,['Second sheet',100,3]])
put('sheets-invalid',[['Instructions only']], [head,['Second sheet',100,3]])
put('missing-all',[['Unknown'],['data']])
(root/'qa-deep-xlsx-corrupt.xlsx').write_bytes(b'Synthetic QA, not a workbook')
fields=['propulsao','combustivel','categoria','categoria_proconve','transmissao','direcao','massa_om_kg','classe_inercia_kg','pbt_kg','f0_n','f2_n_kmh2','potencia_kw','cilindrada_cm3','n_marchas','bateria_kwh','autonomia_eletrica_km']
vehicle=['Combustão','flex','SUVC','PL','automatica','eletrica',1300,1474,1900,155,0.046,129,1600,6,0,0]
for tag,value in [('known-na','NA'),('known-empty',None),('known-invalid','oops')]:
    put(tag,[head+fields,['Synthetic estimate',100,value]+vehicle,['Known label',100,3]])
