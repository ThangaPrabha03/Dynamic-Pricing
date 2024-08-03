import matplotlib.pyplot as plt 
plt.hist(data['Number_of_Riders'], bins=20) 
plt.xlabel('Value') 
plt.ylabel('Frequency') 
plt.title('Histogram of Number_of_Riders') 
plt.show()
plt.bar(data['Location_Category'].value_counts().index, 
data['Location_Category'].value_counts().values) 
plt.xlabel('Category') 
plt.ylabel('Frequency') 
plt.title('Bar Chart of Location_Category') 
plt.show()
plt.scatter(data['Number_of_Drivers'], data['Number_of_Past_Rides']) 
plt.xlabel('Number_of_Drivers') 
plt.ylabel('Number_of_Past_Rides') 
plt.title('Scatter Plot of Number_of_Drivers vs Number_of_Past_Rides') 
plt.show()
import seaborn as sns 
sns.boxplot(x='Location_Category', y='Number_of_Riders', data=data) 
plt.xlabel('Category') 
plt.ylabel('Numerical Column') 
plt.title('Box Plot of Numerical Column by Category') 
plt.show()
sns.pairplot(data) 
plt.title('Pair Plot of Numerical Variables') 
plt.show()
import plotly.express as px 
fig = px.scatter(data, x='Number_of_Past_Rides', y='Number_of_Drivers', 
hover_data=['Average_Ratings']) 
fig.show() 
import dash 
import dash_core_components as dcc 
import dash_html_components as html 
app = dash.Dash(__name__) 
app.layout = html.Div([ 
 dcc.Graph( 
 id='interactive-plot', 
 figure={ 
 'data': [ 
 {'x': data['Number_of_Drivers'], 'y': data['Number_of_Past_Rides'], 
'mode': 'markers', 'type': 'scatter'} 
 ], 
 'layout': { 
 'title': 'Interactive Scatter Plot', 
 'xaxis': {'title': 'Time_of_Booking'}, 
 'yaxis': {'title': 'Vehicle_Type'} 
 } 
 } 
 ) 
]) 
if __name__ == '__main__': 
 app.run_server(debug=True) 