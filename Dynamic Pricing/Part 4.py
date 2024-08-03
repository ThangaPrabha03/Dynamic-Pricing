import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
# Load data 
data = pd.read_csv('dynamic_pricing.csv') 
# Handle missing values 
data.fillna(method='ffill', inplace=True) 
# Encoding categorical variables (if any) 
label_encoders = {} 
for column in data.select_dtypes(include=['object']).columns: 
le = LabelEncoder() 
data[column] = le.fit_transform(data[column]) 
label_encoders[column] = le 
# Split data into features and target 
X = data.drop('Number_of_Drivers',axis=1) # assuming 'price' is the target 
variable 
y = data['Number_of_Drivers'] 
# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 
# Feature scaling 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)
import matplotlib.pyplot as plt 
import seaborn as sns 
# Plotting distributions 
sns.histplot(y, kde=True) 
plt.title('Price Distribution') 
plt.show() 
# Correlation matrix 
plt.figure(figsize=(12, 8)) 
sns.heatmap(data.corr(), annot=True, cmap='coolwarm') 
plt.title('Correlation Matrix') 
plt.show() 
# Assuming some feature engineering is needed 
# Example: Create interaction terms 
X['feature_interaction'] = X['Number_of_Past_Rides'] * 
X['Average_Ratings']
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import 
RandomForestRegressor,GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error,mean_absolute_error, 
r2_score 
# Initialize models 
models = { 
 'Linear Regression': LinearRegression(), 
 'Random Forest': RandomForestRegressor(random_state=42), 
 'Gradient Boosting': GradientBoostingRegressor(random_state=42) 
}
# Train models 
for name, model in models.items(): 
 model.fit(X_train, y_train) 
 print(f"{name} trained.") 
 # Evaluate models 
results = {} 
for name, model in models.items(): 
 y_pred = model.predict(X_test) 
 results[name] = { 
 'RMSE': mean_squared_error(y_test, y_pred, squared=False), 
 'MAE': mean_absolute_error(y_test, y_pred), 
 'R^2': r2_score(y_test, y_pred) 
 } 
# Print evaluation results 
for name, Location_Category in results.items(): 
 print(f"Model: {name}") 
 for Location_Category, value in Location_Category.items(): 
  print(f"{Location_Category}: {value}") 
  print("\n")
  from sklearn.model_selection import GridSearchCV 
# Example: Hyperparameter tuning for Random Forest 
param_grid = { 
 'n_estimators': [100, 200, 300], 
 'max_depth': [None, 10, 20, 30] 
} 
grid_search =GridSearchCV(RandomForestRegressor(random_state=42), 
param_grid, cv=5, scoring='neg_mean_squared_error') 
grid_search.fit(X_train, y_train) 
best_rf = grid_search.best_estimator_ 
print(f"Best parameters for Random Forest:{grid_search.best_params_}") 
# Evaluate the tuned model 
y_pred = best_rf.predict(X_test) 
tuned_results = { 
 'RMSE': mean_squared_error(y_test, y_pred, squared=False), 
 'MAE': mean_absolute_error(y_test, y_pred), 
 'R^2': r2_score(y_test, y_pred) 
} 
print("Tuned Random Forest performance:") 
for score, params in grid_search.cv_results_.items(): 
    print(f"Score: {score}, Parameters: {params}") 
best_rf = grid_search.best_estimator_ 
print(f"Best parameters for Random Forest:{grid_search.best_params_}") 
# Evaluate the tuned model 
y_pred = best_rf.predict(X_test) 
tuned_results = { 
 'RMSE': mean_squared_error(y_test, y_pred, squared=False), 
 'MAE': mean_absolute_error(y_test, y_pred), 
 'R^2': r2_score(y_test, y_pred) 
} 
print("Tuned Random Forest performance:") 
for Location_Category, value in tuned_results.items(): 
 print(f"{Location_Category}: {value}") 