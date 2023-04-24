from flask import Flask, render_template
from plotly.subplots import make_subplots
import requests
import os
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta # added import
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect
from sqlalchemy import func
import plotly.graph_objs as go # added import
from models import engine, OilPrices, Base
from datetime import datetime, timedelta
from lstm_model import lstm_predict_next_year_prices
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')

app = Flask(__name__)

# create the engine and the session
Session = sessionmaker(bind=engine)

# create the table if it doesn't exist
if not inspect(engine).has_table('oil_prices'):
    Base.metadata.create_all(bind=engine)
    print("Created table 'oil_prices'")

# define the route for the home page
@app.route('/')
def home():
    # create a session
    session = Session()

    # define the start and end dates for the historical data
    start_date = datetime(2003, 1, 1)
    end_date = datetime.now()

    # get the historical data from the database
    historical_data = session.query(OilPrices).filter(OilPrices.date >= start_date, OilPrices.date <= end_date).order_by(OilPrices.date.asc()).all()

    # if there is no historical data, create it
    json_data = []

    if not historical_data:
        url = f'https://data.nasdaq.com/api/v3/datasets/OPEC/ORB?start_date={start_date.strftime("%Y-%m-%d")}&end_date={end_date.strftime("%Y-%m-%d")}&order=asc&api_key={API_KEY}'
        response = requests.get(url)
        json_data = response.json()['dataset']['data']

        for data in json_data:
            date = datetime.strptime(data[0], '%Y-%m-%d')
            price = data[1]
            oil_price = OilPrices(date=date, price=price)
            session.add(oil_price)
            session.commit()

    else:
        # check if there is any missing data
        first_missing_date = historical_data[-1].date + timedelta(days=1)
        if first_missing_date <= end_date.date():
            url = f'https://data.nasdaq.com/api/v3/datasets/OPEC/ORB?start_date={first_missing_date.strftime("%Y-%m-%d")}&end_date={end_date.strftime("%Y-%m-%d")}&order=asc&api_key={API_KEY}'
            response = requests.get(url)
            missing_data = response.json()['dataset']['data']

            for data in missing_data:
                date = datetime.strptime(data[0], '%Y-%m-%d')
                price = data[1]
                oil_price = OilPrices(date=date, price=price)
                session.add(oil_price)
                session.commit()

                # add missing data to json_data
                json_data.append({'date': date.strftime('%Y-%m-%d'), 'price': price})

    if json_data:
        for data in json_data['dataset']['data']:
            date = datetime.strptime(data[0], '%Y-%m-%d')
            price = data[1]
            oil_price = OilPrices(date=date, price=price)
            session.add(oil_price)
            session.commit()

        # get the historical data from the database
        historical_data = session.query(OilPrices).filter(OilPrices.date >= start_date, OilPrices.date <= end_date).order_by(OilPrices.date.asc()).all()

    # extract the dates and prices from the historical data
    dates = [row.date for row in historical_data]
    prices = [row.price for row in historical_data]

    # generate a list of dates for the next 365 days
    next_year_dates = [dates[-1]]
    for i in range(364):
        next_date = next_year_dates[-1] + relativedelta(days=+1)
        next_year_dates.append(next_date)

    # get the predicted oil prices for the next year
    next_year_prices = lstm_predict_next_year_prices()
    next_year_prices = np.array(next_year_prices).flatten()  # flatten the 2D array to a 1D array

    # UPPER CHART

    # create traces for the historical and predicted prices
    historical_trace = go.Scatter(x=dates[-730:], y=prices[-730:], mode='lines', name='Historical Prices', line=dict(color='blue'))
    predicted_trace = go.Scatter(x=next_year_dates[:365], y=next_year_prices[:365], mode='lines', name='Predicted Prices', line=dict(color='red'))

    # get the highest, lowest and average points from predicted prices
    highest_point = max(next_year_prices[:365])
    lowest_point = min(next_year_prices[:365])
    average_point = sum(next_year_prices[:365])/len(next_year_prices[:365])

    # create traces for the highest, lowest and average points lines
    highest_trace = go.Scatter(x=[next_year_dates[0], next_year_dates[-1]], y=[highest_point, highest_point], mode='lines', name='Highest Point', line=dict(color='green'))
    lowest_trace = go.Scatter(x=[next_year_dates[0], next_year_dates[-1]], y=[lowest_point, lowest_point], mode='lines', name='Lowest Point', line=dict(color='orange'))
    average_trace = go.Scatter(x=[next_year_dates[0], next_year_dates[-1]], y=[average_point, average_point], mode='lines', name='Average Point', line=dict(color='purple'))

    # Create the plot layout for the first chart
    fig1 = make_subplots(rows=2, cols=1)
    fig1.add_trace(historical_trace, row=1, col=1)
    fig1.add_trace(predicted_trace, row=1, col=1)
    fig1.add_trace(highest_trace, row=1, col=1)
    fig1.add_trace(lowest_trace, row=1, col=1)
    fig1.add_trace(average_trace, row=1, col=1)

    fig1.update_layout(autosize=False, xaxis_title='Date', yaxis_title='Price (USD/bbl)', width=1200, height=1600, margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0, #top margin
    ), legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
    ))

    # LOWER CHART

    # create traces for the historical and predicted prices without lines
    historical_trace_nolines = go.Scatter(x=dates, y=prices, mode='lines', name='Historical Prices', line=dict(color='blue'))
    predicted_trace_nolines = go.Scatter(x=next_year_dates, y=next_year_prices, mode='lines', name='Predicted Prices', line=dict(color='red'))

    # Create the plot layout for the second chart
    fig2 = make_subplots(rows=1, cols=1)
    fig2.add_trace(historical_trace_nolines)
    fig2.add_trace(predicted_trace_nolines)

    fig2.update_layout(autosize=False, xaxis_title='Date', yaxis_title='Price (USD/bbl)', width=1200, height=600, margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0, #top margin
    ), legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
    ))

    session.close()

    # Convert the Plotly figures to an HTML string and pass them to the template
    plot_html1 = fig1.to_html(full_html=False, include_plotlyjs='cdn')
    plot_html2 = fig2.to_html(full_html=False, include_plotlyjs='cdn')
    return render_template('index.html', plot_html1=plot_html1, plot_html2=plot_html2)

if __name__ == '__main__':
    app.run(debug=True)