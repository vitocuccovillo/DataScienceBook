import urllib3
from bs4 import BeautifulSoup as bs, BeautifulSoup
import requests

def get_historical_data(name, number_of_days):
    data = []
    url = "https://finance.yahoo.com/quote/" + name + "/history/"
    #rows = bs(urllib3.urlopen(url).read()).findAll('table')[0].tbody.findAll('tr')

    r = requests.get(
        'https://query2.finance.yahoo.com/v10/finance/quoteSummary/AAPL?formatted=true&crumb=8ldhetOu7RJ&lang=en-US&region=US&modules=defaultKeyStatistics%2CfinancialData%2CcalendarEvents&corsDomain=finance.yahoo.com')
    data = r.json()

    financial_data = data['quoteSummary']['result'][0]['defaultKeyStatistics']
    enterprise_value_dict = financial_data['enterpriseValue']
    print(enterprise_value_dict)

    http = urllib3.PoolManager()
    response = http.request('GET', url)
    rows = BeautifulSoup(response.data.decode('utf-8')).findAll('table')[0].tbody.findAll('tr')

    for each_row in rows:
        divs = each_row.findAll('td')
        print(divs)
        for dd in divs:
            print(dd)
        #if divs[1].text != 'Dividend': #Ignore this row in the table
            #I'm only interested in 'Open' price; For other values, play with divs[1 - 5]
            #data.append({'Date': divs[0].span.text, 'Open': float(divs[1].text.replace(',',''))})

    return data[:number_of_days]

#Test
for i in get_historical_data('amzn', 5):
    print(i)