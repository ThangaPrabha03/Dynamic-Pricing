print(df.head()) 
print(df.tail()) 
print(df.info()) 
print(df.describe()) 
print(df.isnull().sum()) 
df = df.dropna() 
print(df['Number_of_Riders'].unique()) 
transposed_df = df.T 
transposed_df
grouped_df = df.groupby('Number_of_Riders') 
aggregated_df = grouped_df.agg({'Number_of_Riders': 'mean'}) 
grouped_df 
aggregated_df
sns.histplot(df['Number_of_Riders'], bins=20) 
plt.show() 
sns.scatterplot(x=df['Number_of_Riders'], y=df['Number_of_Drivers']) 
plt.show() 
sns.pairplot(df) 
plt.show() 
