from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['DATA'] = 'DATA/'
regressor = joblib.load("DATA/Trail6.pkl")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/plot')
def plot():
    return send_file('DATA/fuel_prices.png', mimetype='image/png')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        csv_file = request.files['csv-file']
        time_period = request.form['time-period']
        time_value = request.form['time-value']

        # Save the uploaded file to the uploads folder
        csv_file.save(app.config['DATA'] + csv_file.filename)

        # Load the CSV file into a Pandas DataFrame
        df = pd.read_csv(app.config['DATA'] + csv_file.filename)

        # Convert date column to separate day, month, and year columns
        df["day"] = pd.to_datetime(df["Date"], format="%d-%m-%Y").dt.day
        df["month"] = pd.to_datetime(df["Date"], format="%d-%m-%Y").dt.month
        df["year"] = pd.to_datetime(df["Date"], format="%d-%m-%Y").dt.year

        # Get the latest date from the input file
        latest_date = df["Date"].max()

        # Set the time period and time interval
        if time_period == "Days":
            time_interval = int(time_value)
        elif time_period == "Weeks":
            time_interval = int(time_value) * 7
        elif time_period == "Months":
            time_interval = int(time_value) * 30
        elif time_period == "Years":
            time_interval = int(time_value) * 365
        else:
            return "Invalid time period"

        # Create a new date range starting from the latest date and going back by the specified time interval
        date_range = pd.date_range(
            end=latest_date, periods=time_interval, freq='D')

        # Create a new dataframe with the date range and matching columns
        df_new = pd.DataFrame({'Store': [0]*time_interval, 'Holiday_Flag': [0]*time_interval, 'Temperature': [0]*time_interval, 'CPI': [0]*time_interval, 'Unemployment': [0]*time_interval, 'Fuel_Price': [0]*time_interval, 'MarkDown1': [0]*time_interval,
                              'MarkDown2': [0]*time_interval, 'MarkDown3': [0]*time_interval, 'MarkDown4': [0]*time_interval, 'MarkDown5': [0]*time_interval, 'IsHoliday': [0]*time_interval, 'day': date_range.day, 'month': date_range.month, 'year': date_range.year})

        # Add missing columns to df_new with default values
        df_new['CPI'] = 0
        df_new['Holiday_Flag'] = 0
        df_new['Store'] = 1
        df_new['Temperature'] = 0
        df_new['Unemployment'] = 0

        # Drop the same columns that were dropped during training
        df_new.drop(["Fuel_Price", "IsHoliday", "MarkDown1", "MarkDown2",
                    "MarkDown3", "MarkDown4", "MarkDown5"], axis=1, inplace=True)

        # Make predictions using the trained model
        future_petrol_prices = regressor.predict(df_new)

        # GRAPH
        # Clear the legend
        plt.clf()

        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
        latest_date = df["Date"].max()
        future_dates = pd.date_range(
            latest_date, periods=time_interval, freq='D')
        x_values = np.arange(len(df), len(df) + len(future_dates))
        last_dates = df.groupby("Date").max().sort_values(
            "Date", ascending=False)[:55]["Fuel_Price"].tolist()
        x_last_dates = np.arange(len(df)-len(last_dates), len(df))
        start_date = latest_date - pd.Timedelta(days=time_interval)
        actual_petrol_prices = df[df["Date"] >= start_date]["Fuel_Price"]
        regressor.fit(
            np.array(actual_petrol_prices.index).reshape(-1, 1), actual_petrol_prices)
        future_petrol_prices = regressor.predict(
            np.array(x_values).reshape(-1, 1))
        plt.plot(x_last_dates, last_dates, label="Last 55 Max Dates Prices")
        plt.plot(x_values, future_petrol_prices,
                 linestyle='dotted', label="Predicted Prices")
        plt.title("Actual and Predicted Fuel Prices")
        plt.xlabel("Time Period")
        plt.ylabel("Fuel Price")

        # Add legend
        plt.legend()
        # Save the plot to a file
        plt.savefig('DATA/fuel_prices.png')

    # Render the result page with the predicted values
    return render_template("result.html", predicted_prices=future_petrol_prices)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
