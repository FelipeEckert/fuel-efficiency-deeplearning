from pathlib import Path
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from playwright.sync_api import sync_playwright, expect

root = Path(os.environ["TEMP"]) / "fuel-ui-20261118"
base = "http://127.0.0.1:8501"

def read_excel(path):
    ns = {"x":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as book:
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            strings = ["".join(si.itertext()) for si in ET.fromstring(book.read("xl/sharedStrings.xml"))]
        for name in book.namelist():
            if re.fullmatch(r"xl/worksheets/sheet\d+.xml", name):
                table = []
                for row in ET.fromstring(book.read(name)).findall(".//x:row", ns):
                    cells = {}
                    for cell in row:
                        value = cell.findtext("x:v", namespaces=ns)
                        if cell.get("t") == "s":
                            value = strings[int(value)]
                        elif cell.get("t") == "inlineStr" and cell.find("x:is", ns) is not None:
                            value = "".join(cell.find("x:is", ns).itertext())
                        elif cell.get("t") is None and value is not None:
                            value = float(value)
                        cells[re.sub(r"\d", "", cell.get("r"))] = value
                    table.append(cells)
                print(path.name, name, table, flush=True)
                yield table

with sync_playwright() as pw:
    browser = pw.chromium.launch(channel="chrome", headless=True)
    context = browser.new_context(viewport={"width":1440,"height":900}, locale="en-US", accept_downloads=True)
    page = context.new_page()
    page.set_default_timeout(60000)
    page.on("pageerror", lambda e: print("PAGEERROR", e, flush=True))

    def visible(text):
        expect(page.get_by_text(text, exact=True).first).to_be_visible(timeout=90000)

    def nav(label):
        page.get_by_role("link", name=label, exact=True).last.click()

    def settle():
        page.wait_for_timeout(1000)
        expect(page.get_by_role("button",name="Stop",exact=True)).to_have_count(0,timeout=60000)

    def click(label, index=0):
        page.get_by_role("button",name=label,exact=True).nth(index).click()
        settle()

    def snapshot(label):
        body = page.locator("main").inner_text()
        sections = body.split("The first saved scenario is the baseline for the comparison.")
        print(label, sections[0].splitlines()[:37], sections[-1].splitlines()[-30:] if len(sections)>1 else "NO SCENARIOS SECTION", flush=True)
        print("ALERTS", page.locator('[data-testid="stException"],[data-testid="stAlert"]').all_inner_texts(), flush=True)

    def name(text):
        entry = page.get_by_role("textbox",name="Scenario name",exact=True)
        entry.fill(text)
        entry.press("Enter")
        settle()

    def mass(value):
        entry = page.get_by_role("spinbutton",name="Curb mass (kg)",exact=True)
        if not entry.is_visible():
            page.get_by_text("Edit the vehicle specifications",exact=True).click()
        entry.fill(str(value))
        entry.press("Enter")
        settle()

    def download(suffix):
        with page.expect_download(timeout=60000) as event:
            page.get_by_role("button",name="Export scenarios (Excel)",exact=True).click()
        path = root / ("qa-deep-vehicle-" + suffix + ".xlsx")
        event.value.save_as(path)
        assert zipfile.is_zipfile(path)
        return list(read_excel(path))

    print("STARTING", flush=True)
    page.goto(base, wait_until="domcontentloaded")
    visible("Vehicle energy efficiency")
    nav("Vehicle")
    visible("MJ/km combined")
    name("  Urban test  ")
    click("Save scenario")
    visible("Urban test (baseline)")
    click("10 % lighter")
    name("Urban test")
    click("Save scenario")
    snapshot("DUPLICATE EXACT")
    name("urban test")
    click("Save scenario")
    snapshot("DUPLICATE CASE")
    first = download("duplicates")
    mass(1500)
    name("Unsaved draft")
    snapshot("BEFORE NAV")
    nav("Comparison")
    visible("Candidates")
    nav("Vehicle")
    visible("Urban test (baseline)")
    snapshot("AFTER NAV")
    print("AFTER NAV DRAFT",page.get_by_role("textbox",name="Scenario name",exact=True).input_value(), flush=True)
    after = download("navigation")
    assert first == after, "Saved workbook changed during navigation"

    mass(301)
    click("10 % lighter")
    snapshot("INVALID ADJUSTMENT")
    name("Rejected adjustment")
    click("Save scenario")
    visible("Rejected adjustment")
    click("Back to original")
    mass(1500)
    click("10 % lighter")
    name("Recovered valid")
    click("Save scenario")
    visible("Recovered valid")
    download("recovered")
    click("Remove")
    snapshot("AFTER BASELINE REMOVAL")
    download("new-baseline")

    page.reload(wait_until="domcontentloaded")
    visible("MJ/km combined")
    settle()
    snapshot("AFTER FULL RELOAD")
    print("AFTER FULL RELOAD SCENARIO ROWS", page.get_by_role("button",name="Remove",exact=True).count(), flush=True)
    browser.close()
