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
    p.wait_for_timeout(1800)
    states={}
    for tag in ('known-empty','known-na','known-invalid','export-names','export-na-name'):
        p.locator('input[type=file]').set_input_files(root/f'qa-deep-xlsx-{tag}.xlsx')
        expect(p.get_by_role('button',name=f'Remove qa-deep-xlsx-{tag}.xlsx',exact=True)).to_be_visible(timeout=60000)
        expect(m).to_have_count(1,timeout=60000)
        p.wait_for_timeout(1800)
        expect(p.get_by_role('button',name='Stop',exact=True)).to_have_count(0,timeout=45000)
        results=[x for x in p.locator('body').inner_text().splitlines() if 'vehicles included' in x or '90 % range' in x]
        alerts=p.locator('[data-testid=stException],[data-testid=stAlert]').all_inner_texts()
        states[tag]=(m.all_inner_texts(),results,alerts)
        print(tag,states[tag],flush=True)
        p.screenshot(path=root/f'qa-deep-xlsx-{tag}.png',full_page=True)
        with p.expect_download(timeout=45000) as dl:
            p.get_by_text('Download portfolio',exact=True).click()
        dl.value.save_as(root/f'qa-deep-xlsx-confirm-{tag}.xlsx')
    assert states['known-empty']==states['known-na']
    assert any('2 of 2' in x for x in states['known-na'][1])
    assert any('1 of 2' in x for x in states['known-invalid'][1])
    assert any('known MJ/km must be a number greater than zero' in x for x in states['known-invalid'][2])
    browser.close()
