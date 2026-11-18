from playwright.sync_api import sync_playwright,expect
from pathlib import Path
import json,os,re,time
T=Path(os.environ['TEMP'])/'fuel-ui-20261118'
goal=json.loads((T/'deep-exp.json').read_text())
base='http://127.0.0.1:8501/Portfolio'
with sync_playwright() as pw:
 b=pw.chromium.launch(channel='chrome',headless=True)
 expect.set_options(timeout=90000)
 ca=b.new_context();cb=b.new_context()
 a=ca.new_page();z=cb.new_page()
 for p in (a,z):
    p.goto(base,wait_until='domcontentloaded')
    expect(p.get_by_text('Download template',exact=True)).to_be_visible()
 def load(p,file,value,count):
    ts=time.monotonic()
    p.locator('input[type=file]').set_input_files(T/file)
    expect(p.locator('[data-testid=stMetricValue]')).to_have_text([value+' MJ/km'])
    expect(p.get_by_text(re.compile(count+' vehicles included'))).to_be_visible()
    print(file,p.locator('[data-testid=stMetricValue]').inner_text(),count,round(time.monotonic()-ts,2),flush=True)
 def after(p,action):
    with p.expect_download(timeout=30000) as down: action()
    d=down.value;d.save_as(T/'deep-persisted.xlsx')
 load(a,'deep25.xlsx',f'{goal["avg"]:.2f}','25 of 25')
 assert z.locator('[data-testid=stMetricValue]').count()==0
 load(z,'normal.xlsx','1.18','2 of 2')
 expect(a.locator('[data-testid=stMetricValue]')).to_have_text('1.04 MJ/km')
 load(a,'deep-one.xlsx',f'{goal["single"]:.2f}','1 of 1')
 print('single from scenario',goal['single'],flush=True)
 load(a,'deep25reverse.xlsx',f'{goal["avg"]:.2f}','25 of 25')
 z.locator('input[type=file]').set_input_files(T/'missing.xlsx')
 expect(z.get_by_text('required column missing: Volume',exact=True)).to_be_visible()
 expect(z.locator('[data-testid=stMetricValue]')).to_have_text('1.18 MJ/km')
 z.reload(wait_until='domcontentloaded')
 expect(z.get_by_text('Add vehicles in the table, with the form or from a spreadsheet.',exact=True)).to_be_visible()
 expect(a.locator('[data-testid=stMetricValue]')).to_have_text('1.04 MJ/km')
 after(a,lambda:a.get_by_text('Download portfolio',exact=True).click())
 print('ISOLATION PASS A still 25 after B bad upload/reload, exported A',flush=True)
 b.close()
