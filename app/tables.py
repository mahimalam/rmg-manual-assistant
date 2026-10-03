"""Conservative normalization of native table cells and branch ownership."""

import re


AMBIGUOUS_TABLE = 'Merged/blank cells or headers require visual source review'


def native_table(table, section):
    grid = table.extract()
    boxes = [[tuple(cell) if cell else None for cell in row.cells] for row in table.rows]
    # Recover only cells whose actual rectangle spans a later row. A missing
    # cell without that geometric evidence remains unresolved.
    for i, row in enumerate(grid):
        visible = [box for box in boxes[i] if box]
        if not visible:
            continue
        y = min(box[1] for box in visible)
        for j, value in enumerate(row):
            if value is not None:
                continue
            for k in range(i - 1, -1, -1):
                box = boxes[k][j]
                if box and box[1] <= y + .5 and box[3] > y + .5:
                    grid[i][j], boxes[i][j] = grid[k][j], box
                    break
    first = grid[0]
    switch_table = (len(first) == 4 and re.search(r'\bDIP\s+switch', section, re.I) and
                    first[0] and re.fullmatch(r'\d{1,2}', first[0].strip()) and
                    (first[2] or '').strip().upper() in ('ON', 'OFF'))
    if switch_table:
        headers = ['DIP switch', 'Description', 'State', 'Operation']
        rows, cells = grid, boxes
    else:
        names = list(table.header.names)
        for j, name in enumerate(names):
            if name:
                continue
            column = next((row[j] for row in boxes if row[j]), None)
            if column is None:
                continue
            # A shared header belongs to both columns only when its PDF cell
            # rectangle covers the complete horizontal extent of this column.
            owners = {name for name, box in zip(table.header.names, table.header.cells)
                      if name and box and box[0] <= column[0] + .5 and box[2] >= column[2] - .5}
            if len(owners) == 1:
                names[j] = owners.pop()
        headers = [value or f'Column {i+1} (header unresolved)'
                   for i, value in enumerate(names)]
        start = 0 if table.header.external else 1
        rows, cells = grid[start:], boxes[start:]
    uncertainty = []
    if any(value is None for row in rows for value in row) or any('header unresolved' in h for h in headers):
        uncertainty.append(AMBIGUOUS_TABLE)
    # A numeric first "header" in a headerless control table is data. Outside
    # the recognized switch shape we keep it inspectable but do not guess.
    if not switch_table and headers and re.fullmatch(r'\d+', headers[0].strip()):
        uncertainty.append('Headerless table relationship unresolved')
    return headers, rows, cells, uncertainty
