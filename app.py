from flask import Flask

from src.IO.get_data_from_yahoo import get_last_close_price
from src.business_logic.process_query import create_business_logic

app = Flask(__name__)

#Linked to GCP project called 'stock'

@app.route('/', methods=['GET'])
def hello():
    return f'Hello dear students, you should use an other route:!\nEX: get_stock_val/<ticker>\n'

# @app.route('/get_stock_val/<ticker>', methods=['GET'])
# def get_stock_value(ticker):
#     bl = create_business_logic()
#     prediction = bl.do_predictions_for(ticker)
#
#     return f'{prediction}\n'


@app.route('/get_stock_val_with_accuracy/<ticker>', methods=['GET'])
def get_stock_val_with_accuracy(ticker):
    bl = create_business_logic()
    prediction_and_score = bl.do_predictions_for(ticker)
    last_close_price = get_last_close_price(ticker)

    result = {"ticker": f'{ticker}',
              "last_close_price": f'{last_close_price.get("close")}',
              "prediction": f'{prediction_and_score.get("prediction")}',
              "balanced_accuracy": f'{prediction_and_score.get("balanced_accuracy")}'
              }
    return f'{result}'

@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    bl = create_business_logic()
    prediction_and_score = bl.do_predictions_for(ticker)

    return prediction_and_score.get("prediction")


if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
