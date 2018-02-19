#a coding: utf-8
import os
import csv
import re
import datetime
import xlrd


def main(fileob,outputpath,filename):
    # ブックを読み込みます。
    book = xlrd.open_workbook(file_contents=fileob)

    dest_dir = os.path.join(outputpath, filename)
    os.makedirs(dest_dir, exist_ok=True)

    # シートでループ
    for sheet in book.sheets():
        sheet_name = sheet.name
        dest_path = os.path.join(dest_dir, sheet_name + '.csv')

        with open(dest_path, 'w', encoding='utf-8') as fp:
            writer = csv.writer(fp)

            for row in range(sheet.nrows):
                li = []
                for col in range(sheet.ncols):
                    cell = sheet.cell(row, col)

                    if cell.ctype == xlrd.XL_CELL_NUMBER:  # 数値
                        val = cell.value

                        if val.is_integer():
                            # 整数に'.0'が付与されていたのでintにcast
                            val = int(val)

                    elif cell.ctype == xlrd.XL_CELL_DATE:  # 日付
                        dt = get_dt_from_serial(cell.value)
                        val = dt.strftime('%Y-%m-%d %H:%M:%S')

                    else:  # その他
                        val = cell.value

                    li.append(val)
                writer.writerow(li)
    book.release_resources


def get_dt_from_serial(serial):
    """日付のシリアル値をdatetime型に変換します。"""
    base_date = datetime.datetime(1899, 12, 30)
    d, t = re.search(r'(\d+)(\.\d+)', str(serial)).groups()
    return base_date + datetime.timedelta(days=int(d)) \
        + datetime.timedelta(seconds=float(t) * 86400)

