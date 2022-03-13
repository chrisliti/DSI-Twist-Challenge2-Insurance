from cProfile import label
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine
import statistics

## Configure page
st.set_page_config(page_title="Medical Expense Dashboard",page_icon=":bar_chart:",layout='wide')

## Set title
st.title('Medical Expenses Dashboard')

st.header("KPI's")

## Create engine
engine = create_engine('postgresql://postgres:Cml.9283.1010@localhost:5432/DSI')

## Query db
df = pd.read_sql_query('select * from "insurance"',con=engine)

## Convert todf

df = pd.DataFrame(df)

## Create sidebar for filtering regions
st.sidebar.header("Region Filter")

region = st.sidebar.multiselect("Select region of interest",
options = df['region'].unique(),
default = df['region'].unique())

## ML app links
st.sidebar.header("Predict Medical Expenses")
st.sidebar.write("Click on link to predict expenses")
st.sidebar.write("[Interactive Single Predictor](https://medical-expenses-prediction.herokuapp.com/)")
st.sidebar.write("[Batch Predictor](https://medical-expenses-batch.herokuapp.com/)")


## Filter df
filtered_df = df.query("region == @region")

## Top KPIs
total_customers = len(filtered_df)

total_expenses = np.round(filtered_df['charges'].sum(),2)

min_expense = np.round(filtered_df['charges'].min(),2)

max_expense = np.round(filtered_df['charges'].max(),2)

average_expense = np.round(filtered_df['charges'].mean(),2)

median_expense = np.round(filtered_df['charges'].median(),2)

female_customers = len(filtered_df[filtered_df['sex']=="female"])
female_percent = np.round((female_customers/total_customers)*100,2)

male_customers = len(filtered_df[filtered_df['sex']=="male"])
male_percent = np.round((male_customers/total_customers)*100,2)

child_mean = int(np.round(filtered_df['children'].mean()))
#iqr = np.round(np.subtract(*np.percentile(filtered_df['charges'], [75, 25])),2)


## Row 1
kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(label="Total Expenses",value=f"{total_expenses:,}")

kpi2.metric(label="Average Expense",value=f"{average_expense:,}")

kpi3.metric(label="Median Expense",value=f"{median_expense:,}")

## Row 2
kpi4, kpi5,kpi6= st.columns(3)

kpi4.metric(label="Max Expense",value=f"{max_expense:,}")

kpi5.metric(label="Min Expense",value=f"{min_expense:,}")

kpi6.metric(label="Total Customers",value=f"{total_customers:,}")

## Row 2
kpi7, kpi8,kpi9 = st.columns(3)

kpi7.metric(label="Female Customers",value=f"{female_percent:,}%")

kpi8.metric(label="Male Customers",value=f"{male_percent:,}%")

kpi9.metric(label="Average Child Count",value=f"{child_mean:,}")

## Charts
st.header("Charts")
## Distribution of expenses
fig_exp = px.histogram(filtered_df,x="charges",nbins = 20,title="<b>Medical Expenses Distribution</b>",color_discrete_sequence = ['yellowgreen'],template="plotly_white")

fig_exp.update_layout(plot_bgcolor="rgba(0,0,0,0)",yaxis=(dict(showgrid=False)))


## Expense by smoke status
expenses_by_smoker = (filtered_df.groupby(['smoker']).mean()[['charges']].sort_values(by="charges"))

fig_ebs = px.bar(expenses_by_smoker,x="charges",y=expenses_by_smoker.index,orientation="h",title="<b>Average Expenses by Smoking Status</b>",template="plotly_white",color_discrete_sequence=['cyan']*len(expenses_by_smoker))

fig_ebs.update_layout(plot_bgcolor="rgba(0,0,0,0)",xaxis=(dict(showgrid=False)))


bar_chart1, bar_chart2 = st.columns(2)


bar_chart1.plotly_chart(fig_exp)
bar_chart2.plotly_chart(fig_ebs)

## Expense by Children count status


#filtered_df.loc[:,"children"] = str(filtered_df['children'])
expenses_by_children = (filtered_df.groupby(['children']).mean()[['charges']].sort_values(by="charges"))

fig_ebc = px.bar(expenses_by_children,y="charges",x=expenses_by_children.index,title="<b>Average Expenses by Child Count</b>",template="plotly_white",color_discrete_sequence=['orange']*len(expenses_by_children))

fig_ebc.update_layout(plot_bgcolor="rgba(0,0,0,0)",yaxis=(dict(showgrid=False)))



## Histogram
fig_bmi = px.histogram(filtered_df,x="bmi",nbins = 20,title="<b>BMI Distribution</b>",color_discrete_sequence = ['lightpink'],template="plotly_white")

fig_bmi.update_layout(plot_bgcolor="rgba(0,0,0,0)",yaxis=(dict(showgrid=False)))

chart1, chart2 = st.columns(2)

chart1.plotly_chart(fig_ebc)
chart2.plotly_chart(fig_bmi)

pie1, pie2 = st.columns(2)

fig1 = px.pie(filtered_df, values='charges', names='sex', title='Total Expenses by Gender',color_discrete_sequence=px.colors.sequential.YlGn)
fig2 = px.pie(filtered_df, values='charges', names='children', title='Total Expenses by Child Count',color_discrete_sequence=px.colors.sequential.Reds)


pie1.plotly_chart(fig1)
pie2.plotly_chart(fig2)









