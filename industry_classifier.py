import yfinance as yf
import requests
import numpy as np
import pandas as pd
import pickle
import warnings
import seaborn as sns
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.datasets import make_classification
from sklearn.model_selection import (cross_val_score, train_test_split)
import operator
from xgboost import XGBClassifier
import streamlit as st 
import matplotlib.pyplot as plt

with open('/Users/jenniferhilibrand/Metis/Classification_ML/R3K_pickle.pickle', 'rb') as to_read: 
    R3K_data = pickle.load(to_read)
    
input_ticker = st.text_input("Input a ticker:", value = 'AAPL')



X_overall=R3K_data.loc[:,['Symbol','Market Cap', 'yf_dividend_rate', 'yf_price2book', 'yf_enterprise2rev', 'yf_enterprise2EBITDA', 'yf_trailing_eps', 'yf_profit_margin', 'yf_implied_vol_clean' ]]
Y_overall=R3K_data['yf_sector']
boolean_mask = Y_overall.isnull()
drop_me_index=Y_overall[boolean_mask].index
Y_overall.drop(drop_me_index,inplace=True)  
X_overall.drop(drop_me_index, inplace=True)
X_train_all, X_test, Y_train_all, Y_test = train_test_split(X_overall, Y_overall, test_size=0.2,random_state=44)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_all, Y_train_all, test_size=0.2,random_state=44)
X_test2=X_test.copy()
X_test=X_test.drop('Symbol', axis =1)
X_train2=X_train.copy()
X_train=X_train.drop('Symbol',axis =1)
X_val2=X_val.copy()
X_val=X_val.drop('Symbol',axis =1)
symbol_list = pd.concat([X_test2['Symbol'], X_val2['Symbol'], X_train2['Symbol']])

model = XGBClassifier(n_estimators = 100, learning_rate=0.4, objective="reg:squarederror", max_depth=3, subsample=1, gamma = 0, colsample_bytree=0.4)
model.fit(X_train, Y_train)
y_preds = model.predict(X_val)
accuracy = accuracy_score(Y_val, y_preds)


y_preds=model.predict(X_test)
output = pd.DataFrame(Y_test)
output['xgb1_preds']=y_preds
output['correct_pred_xgb1'] = (output['yf_sector']==output['xgb1_preds'])

y_preds_probs = model.predict_proba(X_test)
xgb_probs=[]
for i in y_preds_probs:
    xgb_probs.append(i)
output['xgb1_probs']=xgb_probs
output['symbol']=X_test2['Symbol']

class_array = model.classes_
xgb_dict_probs=[]
for i in xgb_probs:
    prob_dict=dict(zip(list(class_array),i))
    xgb_dict_probs.append(prob_dict)
output['xgb_prob_dict']=xgb_dict_probs

first_choice=[]
second_choice=[]
for i in xgb_dict_probs:
    x = sorted(((v,k) for k,v in i.items()))
    first_choice.append(x[-1][1])
    second_choice.append(x[-2][1])
output['first_choice']=first_choice
output['second_choice']=second_choice
output['second_choice_correct'] = (output['yf_sector']==output['second_choice'])

first_second_accuracy = (sum(output['correct_pred_xgb1'])+sum(output['second_choice_correct']))/len(output['correct_pred_xgb1'])


X_total = pd.concat([X_test, X_val, X_train])
Y_total = pd.concat([Y_test, Y_val, Y_train])
symbol_list = pd.concat([X_test2['Symbol'], X_val2['Symbol'], X_train2['Symbol']])
y_preds = model.predict(X_total)
overall_output = X_total
overall_output['true_sector'] = Y_total
overall_output['predicted_sector'] = y_preds
overall_output['symbol_list']=symbol_list
symbol_list = pd.concat([X_test2['Symbol'], X_val2['Symbol'], X_train2['Symbol']])



input_row = overall_output[overall_output['symbol_list']==input_ticker]
true_Sector = input_row['true_sector']
predicted_sector = input_row['predicted_sector']
st.write("True Sector:",true_Sector.iloc[0])
st.write("Predicted Sector:", predicted_sector.iloc[0])
input_row_original=input_row


###

to_scale = overall_output.loc[:,['Market Cap', 'yf_dividend_rate', 'yf_price2book', 'yf_enterprise2rev', 'yf_enterprise2EBITDA', 'yf_trailing_eps', 'yf_profit_margin', 'yf_implied_vol_clean']]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(to_scale)
scaled_df = pd.DataFrame(scaled)
scaled_df['predicted']=overall_output['predicted_sector'].reindex(scaled_df.index)
pred=[]
for i in overall_output['predicted_sector']:
    pred.append(i)
scaled_df['predicted']=pred
true=[]
for i in overall_output['true_sector']:
    true.append(i)
scaled_df['true']=true
ticker=[]
for i in overall_output['symbol_list']:
    ticker.append(i)
scaled_df['ticker']=ticker

input_row = scaled_df[scaled_df['ticker']==input_ticker]
input_true = input_row['true']
input_predicted = input_row['predicted']
single_sector = scaled_df[scaled_df['true']== input_true.iloc[0]]
sector_tickers=[]
for i in single_sector['ticker']:
    sector_tickers.append(i)
dist_list = []

input_row_clean = input_row.iloc[:,0:7].dropna(axis=1)
calc_cols = input_row_clean.columns
single_sector = single_sector[calc_cols]
single_sector_clean=single_sector 



for index, row in single_sector_clean.iterrows():
    dist_list.append(np.linalg.norm(row - input_row_clean))
    
recommender_df = pd.DataFrame(dist_list)
recommender_df['ticker']=sector_tickers

top = recommender_df.sort_values(by=[0])
st.write("Closest peers:")
for i in top[1:6]['ticker']:
    st.write(i)

X_input = st.selectbox('X Axis Comparison',("Market Cap", "Dividend Rate","Price to Book", "EV to EBITDA",  "Trailing EPS", "Profit Margin",  "EV to Revenue"))

Y_input = st.selectbox('Y Axis Comparison',("Market Cap", "Dividend Rate","Price to Book", "EV to EBITDA",  "Trailing EPS", "Profit Margin",  "EV to Revenue"))

points=[]
for i in top[1:6]['ticker']:
    point_idx = int(overall_output[overall_output['symbol_list']==i].index[0])
    point=overall_output.loc[point_idx]
    #point_idx = int(scaled_df[scaled_df['ticker']==i].index[0])
    #point=scaled_df.iloc[point_idx]
    points.append(point)
points_df = pd.DataFrame(points)
#points_df.drop(['predicted', 'true'], axis=1, inplace=True)
points_df.drop(['predicted_sector', 'true_sector'], axis=1, inplace=True)
#x_labels = {0:"Market Cap", 1:"Dividend Rate",2:"Price to Book", 3:"EV to EBITDA", 4: "Trailing EPS", 5:"Profit Margin", 6: "EV to Revenue", 7:"Implied Vol"}  

x_labels = {"Market Cap": "Market Cap","yf_dividend_rate":"Dividend Rate", "yf_price2book": "Price to Book", "yf_enterprise2EBITDA": "EV to EBITDA", "yf_trailing_eps": "Trailing EPS", "yf_profit_margin": "Profit Margin", "yf_enterprise2rev": "EV to Revenue","yf_implied_vol_clean": "Implied Vol",'symbol_list': 'ticker' }
x_labels2 = {0:"Market Cap", 1:"Dividend Rate",2:"Price to Book", 3:"EV to EBITDA", 4: "Trailing EPS", 5:"Profit Margin", 6: "EV to Revenue", 7:"Implied Vol"}  


points_df.rename(columns=x_labels, inplace=True)

input_row_original.rename(columns=x_labels, inplace=True)


x=X_input
y=Y_input
X_axis = points_df[x]
Y_axis = points_df[y]
scatter_labels = points_df['ticker']
scatter_df=pd.DataFrame(X_axis)
scatter_df[y]=Y_axis
scatter_df['labels']=scatter_labels


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(scatter_df[x],scatter_df[y],)
ax.scatter(input_row_original[x],input_row_original[y], color='#990000')
ax.set_xlabel(x)
ax.set_ylabel(y)

xs=list(X_axis.values) 
ys=list(Y_axis.values)    

for i, txt in enumerate(scatter_labels):
    ax.annotate(txt, (xs[i], ys[i]))
st.write(input_row_original)
st.write(fig)