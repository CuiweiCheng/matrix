import requests
from lxml import etree


def download_stock_price(stock_symbol):

    url = f'https://charting.nasdaq.com/ext/charts.dll?2-1-14-0-0-512-03NA000000\
{stock_symbol}\
-&SF:6|8|5-SH:8=20-BG=FFFFFF-BT=0-WD=605-HT=395--xtbl'

    r = requests.get(url)
    html = etree.HTML(r.text)

    # csv_xpath = '/html/body/center[1]/table/tbody/tr[1]/td[2]/font/a'  # no tbody
    result = html.xpath('/html/body/center[1]/table/tr[1]/td[2]/font/a/@href')
    # print(result)
    # type(result[0])
    a = result[0].encode('utf-8')
    b = str(a)
    # print(b)
    # print(b[2:-1])
    download_url = 'https://charting.nasdaq.com/ext/' + b[2:-4] + 'xcl-'
    print(download_url)
    r = requests.get(download_url)
    with open(f"{stock_symbol}.csv", "wb") as code:
        code.write(r.content)


download_stock_price('chh')