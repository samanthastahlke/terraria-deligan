from bs4 import BeautifulSoup
import csv
import os

datafolder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
htmlpath = os.path.abspath(os.path.join(datafolder, 'terraria_items.html'))
csvpath = os.path.abspath(os.path.join(datafolder, 'terraria_items.csv'))

soup = BeautifulSoup(open(htmlpath), "html.parser")
table = soup.select_one("table.terraria")

html_header = table.select('thead tr')[0]
header_contents = html_header.find_all('th')
header = [attr.text.strip() for attr in header_contents]

html_rows = table.select('tbody tr')
rows = []

for row in html_rows:
    new_row = []
    row_contents = row.find_all('td')

    new_row.append(row_contents[0].text.strip())
    new_row.append(row_contents[1].text.strip())
    new_row.append(row.find('img').get('src').strip())

    rows.append(new_row)

with (open(csvpath, 'w+', newline='')) as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
