# 🔥 Calories Burnt Prediction

This project aims to predict the amount of calories burnt during exercise based on various physical and exercise-related parameters. It uses an XGBoost Regressor model, a powerful gradient boosting algorithm known for its performance in regression tasks. The notebook covers data loading, merging, exploratory data analysis, preprocessing, model training, and evaluation.

## 📊 Dataset

This project utilizes two datasets, which are merged to create a comprehensive dataset for prediction:

**calories.csv**: Contains User_ID and Calories (the target variable).

**exercise.csv**: Contains User_ID and various exercise-related features.

After merging, the combined dataset includes the following features:

- **User_ID**: Unique identifier for each user.
- **Gender**: Gender of the user (male/female).
- **Age**: Age of the user.
- **Height**: Height of the user in cm.
- **Weight**: Weight of the user in kg.
- **Duration**: Duration of exercise in minutes.
- **Heart_Rate**: Average heart rate during exercise.
- **Body_Temp**: Body temperature after exercise.
- **Calories**: Amount of calories burnt (the target variable).

## ✨ Features

**Data Loading and Merging**: Loads both calories.csv and exercise.csv into pandas DataFrames and merges them based on User_ID to create a single, unified dataset.

**Initial Data Inspection**: Provides an initial look at the combined dataset's structure, including its dimensions (.shape) and a summary of data types and non-null values (.info()).

**Missing Value Check**: Confirms the absence of missing values across all columns, ensuring data completeness.

**Statistical Summary**: Generates descriptive statistics for numerical features, offering insights into their central tendency, dispersion, and shape.

**Exploratory Data Analysis (EDA)**:
- **Gender Distribution**: Visualizes the distribution of males and females using a count plot.
- **Age Distribution**: Plots the distribution of the Age column to understand age demographics.
- **Height Distribution**: Shows the distribution of Height.
- **Weight Distribution**: Displays the distribution of Weight.
- **Correlation Heatmap**: Visualizes the correlation matrix between all numerical features using a heatmap. This helps identify strong relationships between features and the target variable (Calories).

**Categorical to Numerical Conversion**: Transforms the Gender categorical column into a numerical representation using label encoding (0 for female, 1 for male), which is necessary for machine learning models.

**Feature and Target Separation**: Splits the preprocessed data into features (X, excluding User_ID) and the target variable (Y, Calories).

**Data Splitting**: Divides the dataset into training and testing sets (80% training, 20% testing) to train and evaluate the model's generalization performance.

**XGBoost Regressor Model Training**: Trains an XGBoost Regressor model on the training data. XGBoost is chosen for its efficiency and accuracy in handling complex regression problems.

**Model Evaluation**: Evaluates the model's performance on both the training and test datasets using the Mean Absolute Error (MAE) and R-squared error. MAE measures the average magnitude of the errors in a set of predictions, without considering their direction, while R-squared measures the proportion of the variance in the dependent variable that is predictable from the independent variables.

## 🛠️ Technologies Used

- **Python**
- **pandas**: For efficient data loading, merging, and manipulation.
- **numpy**: For numerical operations, especially with arrays.
- **matplotlib.pyplot**: For creating static visualizations.
- **seaborn**: For producing attractive and informative statistical graphics, including count plots, distribution plots, and heatmaps.
- **scikit-learn**: For core machine learning functionalities, including:
  - **train_test_split**: To divide the dataset into training and testing sets.
  - **metrics**: For calculating evaluation metrics like Mean Absolute Error.
- **xgboost**: For the powerful XGBRegressor model.

## 📦 Requirements

To run this project, you will need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## 🚀 Getting Started

To get this project up and running on your local machine, follow these simple steps.

### Installation

1. Clone the repository (if applicable):
```bash
git clone <repository_url>
cd <repository_name>
```

2. Install the required Python packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Usage

1. **Place the datasets**: Ensure both `calories.csv` and `exercise.csv` files are located in the same directory as the Jupyter notebook (`Calories_Burnt_Prediction.ipynb`).

2. **Run the Jupyter Notebook**: Open and execute all the cells in the `Calories_Burnt_Prediction.ipynb` notebook using a Jupyter environment (e.g., Jupyter Lab, Jupyter Notebook, Google Colab).

The notebook will:
- Load and merge the data.
- Perform comprehensive exploratory data analysis with various visualizations.
- Preprocess the data.
- Train the XGBoost Regressor model.
- Output the model's performance metrics (MAE and R-squared) on both training and test data.
- Provide predictions on the test data.

## 📈 Results

The notebook provides the Mean Absolute Error (MAE) and R-squared error for the XGBoost Regressor model on both the training and test datasets.

**Mean Absolute Error (MAE)**:
- Training Data: Approximately 1.76
- Test Data: Approximately 1.78

**R-squared Error**:
- Training Data: Approximately 0.9995
- Test Data: Approximately 0.9989

These results indicate that the XGBoost model performs exceptionally well in predicting calories burnt, with very low error and high R-squared values on both training and unseen test data, suggesting excellent fit and generalization capabilities.

## 🧑‍💻 Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit pull requests.

## 📄 License

This project is open-source.
