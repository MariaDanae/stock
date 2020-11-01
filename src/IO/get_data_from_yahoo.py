from datetime import datetime, timedelta

from yahoo_fin import stock_info as si


def get_last_stock_price(ticker, last=False):
    if last:
        now = datetime.now()
        start_date = now - timedelta(days=30)
        return si.get_data(ticker, start_date)
    return si.get_data(ticker)

def get_last_close_price(ticker):
    now = datetime.now()
    start_date = now - timedelta(days=1)
    latest_data = si.get_data(ticker, start_date)
    latest_data['date'] = latest_data.index
    return {'close': latest_data.iloc[0]['close'],
            'date': latest_data.iloc[0]['date']
            }