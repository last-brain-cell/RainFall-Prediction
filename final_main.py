import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Rainfall Prediction using RandomForest",
    layout="wide",
    page_icon="🌦️"
)
st.title("Rainfall Prediction using RandomForest")
month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10,'NOV': 11, 'DEC': 12}

subdiv = st.selectbox("Subdivision", ['ANDAMAN & NICOBAR ISLANDS', 'ARUNACHAL PRADESH', 'ASSAM & MEGHALAYA','NAGA MANI MIZO TRIPURA', 'SUB HIMALAYAN WEST BENGAL & SIKKIM','GANGETIC WEST BENGAL', 'ORISSA', 'JHARKHAND', 'BIHAR', 'EAST UTTAR PRADESH','WEST UTTAR PRADESH', 'UTTARAKHAND', 'HARYANA DELHI & CHANDIGARH', 'PUNJAB','HIMACHAL PRADESH', 'JAMMU & KASHMIR', 'WEST RAJASTHAN', 'EAST RAJASTHAN','WEST MADHYA PRADESH', 'EAST MADHYA PRADESH', 'GUJARAT REGION','SAURASHTRA & KUTCH', 'KONKAN & GOA', 'MADHYA MAHARASHTRA', 'MATATHWADA','VIDARBHA', 'CHHATTISGARH', 'COASTAL ANDHRA PRADESH', 'TELANGANA','RAYALSEEMA', 'TAMIL NADU', 'COASTAL KARNATAKA', 'NORTH INTERIOR KARNATAKA','SOUTH INTERIOR KARNATAKA', 'KERALA', 'LAKSHADWEEP'])

year = st.selectbox('Year', range(1990, 2051))
duration = st.radio(
    "Select Duration",
    ('Monthly', 'Annual'))

if duration == "Monthly":
    month = st.selectbox('Month', month_map.keys())
    m = month
    month = month_map[month]
else:
    month = 1
data_load_state = st.markdown('### Running...')
acc_load_state = st.text('Accuracy: Calculating...')

df = pd.read_csv("rainfall in india 1901-2015.csv")

df = df.fillna(0)
df["SUBDIVISION"].nunique()
print(df["SUBDIVISION"].unique())
group = df.groupby('SUBDIVISION')['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

data = group.get_group(subdiv)
# data.head()

df1 = data.melt(['YEAR']).reset_index()
df1 = df1[['YEAR', 'variable', 'value']].reset_index().sort_values(by=['YEAR', 'index'])
df1.columns = ['Index', 'Year', 'Month', 'Avg_Rainfall']
df1.drop(columns="Index", inplace=True)
# df1.head()

df1['Month'] = df1['Month'].map(month_map)
df1.groupby("Year").sum()
# df1.head(12)

x = np.asanyarray(df1[['Year', 'Month']]).astype('int')
y = np.asanyarray(df1['Avg_Rainfall']).astype('int')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

y_train_predict = lin_reg.predict(x_train)
y_test_predict = lin_reg.predict(x_test)

random_forest_model = RandomForestRegressor(max_depth=500, max_features='sqrt', min_samples_leaf=4, min_samples_split=10, n_estimators=800)
random_forest_model.fit(x_train, y_train)
# RandomForestRegressor(max_depth=100, max_features='sqrt', min_samples_leaf=4, min_samples_split=10, n_estimators=800)
y_train_predict = random_forest_model.predict(x_train)
y_test_predict = random_forest_model.predict(x_test)
print("-------Test Data--------")
print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))

print("\n-------Train Data--------")
print('MAE:', metrics.mean_absolute_error(y_train, y_train_predict))
print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))

print("-----------Training Accuracy------------")
print(round(random_forest_model.score(x_train, y_train), 3) * 100)
print("-----------Testing Accuracy------------")
print(round(random_forest_model.score(x_test, y_test), 3) * 100)
summ = [random_forest_model.predict([[year, month], ])[0]]
for x in range(1, 13):
    if x == month:
        continue
    summ.append(random_forest_model.predict([[year, x], ])[0])
if duration == "Annual":
    data_load_state.markdown(f"### {'Low' if sum(summ) < 900 else 'Moderate' if sum(summ) < 2700 else 'High'} rainfall of {round(sum(summ), 2)}mm is predicted")
else:
    data_load_state.markdown(f"### {'Low' if summ[0] < 900/12 else 'Moderate' if summ[0] < 2700/12 else 'High'} rainfall of {round(summ[0], 2)}mm is predicted")
acc_load_state.text(f"Accuracy: {round(random_forest_model.score(x_train, y_train), 3) * 100}%")


col1, col2 = st.columns(2)

with col1:
    st.markdown("Model Fit")
    st.line_chart(pd.DataFrame({"Test": y_test, "Prediction": y_test_predict}))

with col2:
    df2 = df[df["YEAR"] == year]
    if duration == 'Annual':
        st.markdown('Annual Rainfall v/s SubDivisions')
        df2 = df2[["SUBDIVISION", "ANNUAL"]]
        st.bar_chart(df2, x="SUBDIVISION", y="ANNUAL")
    else:
        st.markdown(f'{m} Rainfall v/s SubDivisions')
        df2 = df2[["SUBDIVISION", m]]
        st.bar_chart(df2, x="SUBDIVISION", y=m)

st.title("Detailed Data View")
st.dataframe(df)
