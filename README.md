**Internship Task 6 — House Price Prediction using Machine Learning**

**Overview**
It builds a House Price Prediction system using the Kaggle Housing dataset. The goal is to preprocess real estate data and train a Linear Regression model to predict house prices based on property features such as area, bedrooms, bathrooms, and amenities.

**Dataset**
File: Housing.csv (Kaggle)
Shape: 545 rows × 13 columns
Target Variable: price

**Column**           **Type**                        **Description**
price	             Numeric	               House price (target variable)
area	             Numeric	               Area of the house in square feet
bedrooms	         Numeric	               Number of bedrooms
bathrooms	         Numeric	               Number of bathrooms
stories	           Numeric	               Number of stories/floors
parking	           Numeric	               Number of parking spaces
mainroad	         Binary (yes/no)	       Whether connected to main road
guestroom	         Binary (yes/no)	       Whether guestroom is available
basement	         Binary (yes/no)	       Whether basement is available
hotwaterheating	   Binary (yes/no)	       Whether hot water heating is available
airconditioning	   Binary (yes/no)	       Whether air conditioning is available
prefarea	         Binary (yes/no)	       Whether in a preferred area
furnishingstatus	 Ordinal	               Furnishing level: unfurnished / semi-furnished / furnished

**Technologies Used**
**Library**       	**Purpose**
pandas	       Data loading and manipulation
numpy        	 Numerical computations
matplotlib	   Data visualization and prediction plots
seaborn	       Statistical plots and heatmaps
scikit-learn	 Preprocessing, modeling, and evaluation

**Project Workflow**
1. Data Loading & Exploration
Loaded Housing.csv using pandas
Checked shape, column names, data types
Viewed descriptive statistics using `.describe()` and `.info()`
Confirmed no missing values in the dataset

2. Feature Identification
Separated features into three groups:
Numeric: area, bedrooms, bathrooms, stories, parking
Binary (yes/no): mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea
Ordinal: furnishingstatus

3. Preprocessing Pipeline (scikit-learn)
Used ColumnTransformer to apply different encodings to different column types:
**Column Group**	               **Encoder**                                          	**Reason**
Binary yes/no columns	    OneHotEncoder(drop='if_binary')          	Converts to 0/1, drops redundant column
furnishingstatus	        OrdinalEncoder with defined order       	Preserves natural order: unfurnished < semi-furnished < furnished
Numeric columns	          StandardScaler                           	Scales to zero mean and unit variance

4. Train / Test Split
80% Training / 20% Testing
random_state=42 for reproducibility

6. Model Training
Linear Regression trained on preprocessed data

8. Model Evaluation
Metric	Description
MAE (Mean Absolute Error)	Average absolute difference between actual and predicted price
RMSE (Root Mean Squared Error)	Penalizes large errors more heavily
R² Score	Proportion of variance explained by the model (1.0 = perfect)

8. Visualization
Custom scatter plot showing:
Blue circles — Actual house prices
Red triangles — Predicted house prices
Gray vertical lines — Error gap between actual and predicted
Samples sorted by actual price for clear trend visualization

**Install required libraries**
pip install pandas numpy scikit-learn matplotlib seaborn

**Project Structure**

Internship_task_6/
│
├── internship_task_6.ipynb        # Main notebook with all code and output
├── Housing.csv                    # Kaggle dataset
└── README.md                      # Project documentation (this file)

**Key Findings**
The dataset has no missing values, requiring no imputation
area, airconditioning, and prefarea are among the strongest predictors of price
OrdinalEncoder is used correctly for furnishingstatus since it has a meaningful order
OneHotEncoder(drop='if_binary') cleanly handles all yes/no columns without redundancy
The sorted scatter plot with error lines provides a clear visual of where the model over or under-predicts
