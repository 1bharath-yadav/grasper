import sys
import os
import re
from bs4 import BeautifulSoup


def analyze_html(file_path: str):
    temp_dir = os.path.dirname(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
    except Exception as e:
        print(f"Failed to read HTML file: {str(e)}")
        return
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
    new_soup = BeautifulSoup(
        '<html><head></head><body></body></html>', 'html.parser')
    new_soup.head.append(new_soup.new_tag(
        'title', string="Compressed Tables from " + title))
    h1 = new_soup.new_tag('h1')
    h1.string = "Extracted Tables from " + title
    new_soup.body.append(h1)
    tables = soup.find_all('table')
    unique_keys = set()
    for i, table in enumerate(tables):
        rows = table.find_all('tr')
        if len(rows) < 2:
            continue
        selectors = ['table']
        if 'id' in table.attrs:
            selectors.append(f'#{table.attrs["id"]}')
        if 'class' in table.attrs:
            selectors.append(f'.'.join(table.attrs['class']))
        header_row = table.find('thead').find(
            'tr') if table.find('thead') else rows[0]
        headers = [cell.get_text(strip=True)
                   for cell in header_row.find_all(['th', 'td'])]
        if not headers:
            continue
        key = tuple(sorted(headers))
        if key in unique_keys:
            continuenew_soup
        unique_keys.add(key)
        data_rows = rows[1:] if header_row == rows[0] else rows
        sample_row = [cell.get_text(strip=True) for cell in data_rows[0].find_all([
            'td', 'th'])] if data_rows else []
        h2 = new_soup.new_tag('h2')
        h2.string = f"Table {i} - Selector: {', '.join(selectors)}"
        new_soup.body.append(h2)
        new_table = new_soup.new_tag('table')
        tr = new_soup.new_tag('tr')
        for h in headers:
            th = new_soup.new_tag('th')
            th.string = h
            tr.append(th)
        new_table.append(tr)
        if sample_row:
            tr = new_soup.new_tag('tr')
            for s in sample_row:
                td = new_soup.new_tag('td')
                td.string = s
                tr.append(td)
            new_table.append(tr)
        new_soup.body.append(new_table)
    result = _minify_html(str(new_soup))
    original_len = len(html_content)
    minified_len = len(result)
    compression = 100 * (1 - minified_len /
                         original_len) if original_len > 0 else 0
    print(f"Compression percentage: {compression:.2f}%")
    return result


def _minify_html(html):
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    html = re.sub(r'\s+', ' ', html)
    html = re.sub(r'>\s+<', '><', html)
    return html.strip()
