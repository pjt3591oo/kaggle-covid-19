import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet

df = pd.read_csv('./dataset/Time.csv')

deleted_df = df.drop(columns=['time', 'test', 'negative'], inplace=False)
deleted_df.plot.line(subplots=True, x="date", rot=45)


forecasting_df = pd.DataFrame({'ds': deleted_df['date'], 'y': df['deceased']})
m = Prophet(yearly_seasonality=True, daily_seasonality=True)
m.fit(forecasting_df)

future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)
m.plot(forecast)
# plt.legend()
plt.show()