from pathlib import Path
from playwright.sync_api import sync_playwright, expect
import os, re
root=Path(os.environ['TEMP'])/'fuel-ui-20261118'
with sync_playwright() as pw:
    browser = pw.chromium.launch(channel='chrome', headless=True)
    ctx=browser.new_context(viewport={'width':1440,'height':900}, locale='pt-BR', accept_downloads=True)
    p=ctx.new_page()
    p.on('pageerror',lambda e:print('PAGEERROR',e))
    p.goto('http://127.0.0.1:8501',wait_until='domcontentloaded')
    p.get_by_role('link',name='Portfolio',exact=True).last.click()
    expect(p.get_by_text('Download template',exact=True)).to_be_visible(timeout=60000)
    m=p.locator('[data-testid=stMetricValue]')
    for tag, score, marker in [
        ('baseline','2',None),('names','2',None),('na-name','2',None),
        ('na-mj','2','row 1:'),('mixed-text-mj','2.38',None),
        ('numeric-volume',None,None),('numeric-volume-positive',None,None),
        ('sheets-valid','1',None),('sheets-invalid','1','required column missing: Name'),
        ('missing-all','1','required column missing: Name'),
        ('corrupt','1','could not read the spreadsheet'),
        ('baseline','2',None)]:
        file=root/f'qa-deep-xlsx-{tag}.xlsx'
        p.locator('input[type=file]').set_input_files(file)
        p.wait_for_timeout(2000)
        expect(p.get_by_role('button',name='Stop',exact=True)).to_have_count(0,timeout=45000)
        if marker:
            expect(p.get_by_text(re.compile(marker)).first).to_be_visible(timeout=12000)
        if score:
            try: expect(m).to_have_text(f'{score} MJ/km',timeout=4000)
            except AssertionError as err: print('SCORE_DIFF',score,m.all_inner_texts())
        results=[x for x in p.locator('body').inner_text().splitlines() if 'vehicles included' in x]
        alerts=p.locator('[data-testid=stException],[data-testid=stAlert]').all_inner_texts()
        print(tag,m.all_inner_texts(),results,alerts,flush=True)
        if tag in ('names','na-name','na-mj','mixed-text-mj','numeric-volume','numeric-volume-positive','sheets-invalid','corrupt'):
            with p.expect_download(timeout=45000) as dl:
                p.get_by_text('Download portfolio',exact=True).click()
            dl.value.save_as(root/f'qa-deep-xlsx-export-{tag}.xlsx')
        if tag in ('numeric-volume','sheets-invalid'):
            p.screenshot(path=root/f'qa-deep-xlsx-{tag}.png', full_page=True)
    browser.close()
