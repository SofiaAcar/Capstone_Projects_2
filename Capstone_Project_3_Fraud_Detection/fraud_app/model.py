{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "jx3Ut_rjMuHR"
   },
   "source": [
    "___\n",
    "\n",
    "<p style=\"text-align: center;\"><img src=\"https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV\" class=\"img-fluid\" alt=\"CLRSWY\"></p>\n",
    "\n",
    "___"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "-24P_wByMuHX"
   },
   "source": [
    "# WELCOME!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Ow9AD-4vMuHX"
   },
   "source": [
    "Welcome to \"***Fraud Detection Project***\". This is the last project of the Capstone Series.\n",
    "\n",
    "One of the challenges in this project is the absence of domain knowledge. So without knowing what the column names are, you will only be interested in their values. The other one is the class frequencies of the target variable are quite imbalanced.\n",
    "\n",
    "You will implement ***Logistic Regression, Random Forest, Neural Network*** algorithms and ***SMOTE*** technique. Also visualize performances of the models using ***Seaborn, Matplotlib*** and ***Yellowbrick*** in a variety of ways.\n",
    "\n",
    "At the end of the project, you will have the opportunity to deploy your model by ***Flask API***.\n",
    "\n",
    "Before diving into the project, please take a look at the Determines and Tasks.\n",
    "\n",
    "- ***NOTE:*** *This tutorial assumes that you already know the basics of coding in Python and are familiar with model deployement (flask api) as well as the theory behind Logistic Regression, Random Forest, Neural Network.*\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dqbMkIZ-MuHY"
   },
   "source": [
    "---\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "spCFDhO7MuHY"
   },
   "source": [
    "# #Determines\n",
    "The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where it has **492 frauds** out of **284,807** transactions. The dataset is **highly unbalanced**, the positive class (frauds) account for 0.172% of all transactions.\n",
    "\n",
    "**Feature Information:**\n",
    "\n",
    "**Time**: This feature is contains the seconds elapsed between each transaction and the first transaction in the dataset. \n",
    "\n",
    "**Amount**:  This feature is the transaction Amount, can be used for example-dependant cost-senstive learning. \n",
    "\n",
    "**Class**: This feature is the target variable and it takes value 1 in case of fraud and 0 otherwise.\n",
    "\n",
    "---\n",
    "\n",
    "The aim of this project is to predict whether a credit card transaction is fraudulent. Of course, this is not easy to do.\n",
    "First of all, you need to analyze and recognize your data well in order to draw your roadmap and choose the correct arguments you will use. Accordingly, you can examine the frequency distributions of variables. You can observe variable correlations and want to explore multicollinearity. You can show the distribution of the target variable's classes over other variables. \n",
    "Also, it is useful to take missing values and outliers.\n",
    "\n",
    "After these procedures, you can move on to the model building stage by doing the basic data pre-processing you are familiar with. \n",
    "\n",
    "Start with Logistic Regression and evaluate model performance. You will apply the SMOTE technique used to increase the sample for unbalanced data. Next, rebuild your Logistic Regression model with SMOTE applied data to observe its effect.\n",
    "\n",
    "Then, you will use three different algorithms in the model building phase. You have applied Logistic Regression and Random Forest in your previous projects. However, the Deep Learning Neural Network algorithm will appear for the first time.\n",
    "\n",
    "In the final step, you will deploy your model using ***Flask API***."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "YOl6z9mXMuHY"
   },
   "source": [
    "---\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1o6X3hLLMuHZ"
   },
   "source": [
    "# #Tasks\n",
    "\n",
    "#### 1. Exploratory Data Analysis & Data Cleaning\n",
    "\n",
    "- Import Modules, Load Data & Data Review\n",
    "- Exploratory Data Analysis\n",
    "- Data Cleaning\n",
    "\n",
    "\n",
    "\n",
    "    \n",
    "#### 2. Data Preprocessing\n",
    "\n",
    "- Scaling\n",
    "- Train - Test Split\n",
    "\n",
    "\n",
    "#### 3. Model Building\n",
    "\n",
    "- Logistic Regression without SMOTE\n",
    "- Apply SMOTE\n",
    "- Logistic Regression with SMOTE\n",
    "- Random Forest Classifier with SMOTE\n",
    "- Neural Network\n",
    "\n",
    "#### 4. Model Deployement\n",
    "\n",
    "- Save and Export the Model as .pkl\n",
    "- Save and Export Variables as .pkl \n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9sDSWJywMuHZ"
   },
   "source": [
    "---\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fbFMU3AdMuHZ"
   },
   "source": [
    "## 1. Exploratory Data Analysis & Data Cleaning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5nmI08_GMuHZ"
   },
   "source": [
    "### Import Modules, Load Data & Data Review"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "yKZtJybfMuHa"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import statsmodels.api as sm\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "from scipy.stats import zscore\n",
    "from scipy import stats\n",
    "from sklearn.preprocessing import scale, StandardScaler\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score\n",
    "from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"creditcard.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>...</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Amount</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.359807</td>\n",
       "      <td>-0.072781</td>\n",
       "      <td>2.536347</td>\n",
       "      <td>1.378155</td>\n",
       "      <td>-0.338321</td>\n",
       "      <td>0.462388</td>\n",
       "      <td>0.239599</td>\n",
       "      <td>0.098698</td>\n",
       "      <td>0.363787</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.018307</td>\n",
       "      <td>0.277838</td>\n",
       "      <td>-0.110474</td>\n",
       "      <td>0.066928</td>\n",
       "      <td>0.128539</td>\n",
       "      <td>-0.189115</td>\n",
       "      <td>0.133558</td>\n",
       "      <td>-0.021053</td>\n",
       "      <td>149.62</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.191857</td>\n",
       "      <td>0.266151</td>\n",
       "      <td>0.166480</td>\n",
       "      <td>0.448154</td>\n",
       "      <td>0.060018</td>\n",
       "      <td>-0.082361</td>\n",
       "      <td>-0.078803</td>\n",
       "      <td>0.085102</td>\n",
       "      <td>-0.255425</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.225775</td>\n",
       "      <td>-0.638672</td>\n",
       "      <td>0.101288</td>\n",
       "      <td>-0.339846</td>\n",
       "      <td>0.167170</td>\n",
       "      <td>0.125895</td>\n",
       "      <td>-0.008983</td>\n",
       "      <td>0.014724</td>\n",
       "      <td>2.69</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-1.358354</td>\n",
       "      <td>-1.340163</td>\n",
       "      <td>1.773209</td>\n",
       "      <td>0.379780</td>\n",
       "      <td>-0.503198</td>\n",
       "      <td>1.800499</td>\n",
       "      <td>0.791461</td>\n",
       "      <td>0.247676</td>\n",
       "      <td>-1.514654</td>\n",
       "      <td>...</td>\n",
       "      <td>0.247998</td>\n",
       "      <td>0.771679</td>\n",
       "      <td>0.909412</td>\n",
       "      <td>-0.689281</td>\n",
       "      <td>-0.327642</td>\n",
       "      <td>-0.139097</td>\n",
       "      <td>-0.055353</td>\n",
       "      <td>-0.059752</td>\n",
       "      <td>378.66</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-0.966272</td>\n",
       "      <td>-0.185226</td>\n",
       "      <td>1.792993</td>\n",
       "      <td>-0.863291</td>\n",
       "      <td>-0.010309</td>\n",
       "      <td>1.247203</td>\n",
       "      <td>0.237609</td>\n",
       "      <td>0.377436</td>\n",
       "      <td>-1.387024</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.108300</td>\n",
       "      <td>0.005274</td>\n",
       "      <td>-0.190321</td>\n",
       "      <td>-1.175575</td>\n",
       "      <td>0.647376</td>\n",
       "      <td>-0.221929</td>\n",
       "      <td>0.062723</td>\n",
       "      <td>0.061458</td>\n",
       "      <td>123.50</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.0</td>\n",
       "      <td>-1.158233</td>\n",
       "      <td>0.877737</td>\n",
       "      <td>1.548718</td>\n",
       "      <td>0.403034</td>\n",
       "      <td>-0.407193</td>\n",
       "      <td>0.095921</td>\n",
       "      <td>0.592941</td>\n",
       "      <td>-0.270533</td>\n",
       "      <td>0.817739</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.009431</td>\n",
       "      <td>0.798278</td>\n",
       "      <td>-0.137458</td>\n",
       "      <td>0.141267</td>\n",
       "      <td>-0.206010</td>\n",
       "      <td>0.502292</td>\n",
       "      <td>0.219422</td>\n",
       "      <td>0.215153</td>\n",
       "      <td>69.99</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Time        V1        V2        V3        V4        V5        V6        V7  \\\n",
       "0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   \n",
       "1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   \n",
       "2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   \n",
       "3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   \n",
       "4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   \n",
       "\n",
       "         V8        V9  ...       V21       V22       V23       V24       V25  \\\n",
       "0  0.098698  0.363787  ... -0.018307  0.277838 -0.110474  0.066928  0.128539   \n",
       "1  0.085102 -0.255425  ... -0.225775 -0.638672  0.101288 -0.339846  0.167170   \n",
       "2  0.247676 -1.514654  ...  0.247998  0.771679  0.909412 -0.689281 -0.327642   \n",
       "3  0.377436 -1.387024  ... -0.108300  0.005274 -0.190321 -1.175575  0.647376   \n",
       "4 -0.270533  0.817739  ... -0.009431  0.798278 -0.137458  0.141267 -0.206010   \n",
       "\n",
       "        V26       V27       V28  Amount  Class  \n",
       "0 -0.189115  0.133558 -0.021053  149.62      0  \n",
       "1  0.125895 -0.008983  0.014724    2.69      0  \n",
       "2 -0.139097 -0.055353 -0.059752  378.66      0  \n",
       "3 -0.221929  0.062723  0.061458  123.50      0  \n",
       "4  0.502292  0.219422  0.215153   69.99      0  \n",
       "\n",
       "[5 rows x 31 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "K22reBkbMuHa"
   },
   "source": [
    "### Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "id": "2_yeQU4WJQ0D"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 284807 entries, 0 to 284806\n",
      "Data columns (total 31 columns):\n",
      " #   Column  Non-Null Count   Dtype  \n",
      "---  ------  --------------   -----  \n",
      " 0   Time    284807 non-null  float64\n",
      " 1   V1      284807 non-null  float64\n",
      " 2   V2      284807 non-null  float64\n",
      " 3   V3      284807 non-null  float64\n",
      " 4   V4      284807 non-null  float64\n",
      " 5   V5      284807 non-null  float64\n",
      " 6   V6      284807 non-null  float64\n",
      " 7   V7      284807 non-null  float64\n",
      " 8   V8      284807 non-null  float64\n",
      " 9   V9      284807 non-null  float64\n",
      " 10  V10     284807 non-null  float64\n",
      " 11  V11     284807 non-null  float64\n",
      " 12  V12     284807 non-null  float64\n",
      " 13  V13     284807 non-null  float64\n",
      " 14  V14     284807 non-null  float64\n",
      " 15  V15     284807 non-null  float64\n",
      " 16  V16     284807 non-null  float64\n",
      " 17  V17     284807 non-null  float64\n",
      " 18  V18     284807 non-null  float64\n",
      " 19  V19     284807 non-null  float64\n",
      " 20  V20     284807 non-null  float64\n",
      " 21  V21     284807 non-null  float64\n",
      " 22  V22     284807 non-null  float64\n",
      " 23  V23     284807 non-null  float64\n",
      " 24  V24     284807 non-null  float64\n",
      " 25  V25     284807 non-null  float64\n",
      " 26  V26     284807 non-null  float64\n",
      " 27  V27     284807 non-null  float64\n",
      " 28  V28     284807 non-null  float64\n",
      " 29  Amount  284807 non-null  float64\n",
      " 30  Class   284807 non-null  int64  \n",
      "dtypes: float64(30), int64(1)\n",
      "memory usage: 67.4 MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>...</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Amount</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>284807.000000</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>...</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>2.848070e+05</td>\n",
       "      <td>284807.000000</td>\n",
       "      <td>284807.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>94813.859575</td>\n",
       "      <td>3.919560e-15</td>\n",
       "      <td>5.688174e-16</td>\n",
       "      <td>-8.769071e-15</td>\n",
       "      <td>2.782312e-15</td>\n",
       "      <td>-1.552563e-15</td>\n",
       "      <td>2.010663e-15</td>\n",
       "      <td>-1.694249e-15</td>\n",
       "      <td>-1.927028e-16</td>\n",
       "      <td>-3.137024e-15</td>\n",
       "      <td>...</td>\n",
       "      <td>1.537294e-16</td>\n",
       "      <td>7.959909e-16</td>\n",
       "      <td>5.367590e-16</td>\n",
       "      <td>4.458112e-15</td>\n",
       "      <td>1.453003e-15</td>\n",
       "      <td>1.699104e-15</td>\n",
       "      <td>-3.660161e-16</td>\n",
       "      <td>-1.206049e-16</td>\n",
       "      <td>88.349619</td>\n",
       "      <td>0.001727</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>47488.145955</td>\n",
       "      <td>1.958696e+00</td>\n",
       "      <td>1.651309e+00</td>\n",
       "      <td>1.516255e+00</td>\n",
       "      <td>1.415869e+00</td>\n",
       "      <td>1.380247e+00</td>\n",
       "      <td>1.332271e+00</td>\n",
       "      <td>1.237094e+00</td>\n",
       "      <td>1.194353e+00</td>\n",
       "      <td>1.098632e+00</td>\n",
       "      <td>...</td>\n",
       "      <td>7.345240e-01</td>\n",
       "      <td>7.257016e-01</td>\n",
       "      <td>6.244603e-01</td>\n",
       "      <td>6.056471e-01</td>\n",
       "      <td>5.212781e-01</td>\n",
       "      <td>4.822270e-01</td>\n",
       "      <td>4.036325e-01</td>\n",
       "      <td>3.300833e-01</td>\n",
       "      <td>250.120109</td>\n",
       "      <td>0.041527</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>-5.640751e+01</td>\n",
       "      <td>-7.271573e+01</td>\n",
       "      <td>-4.832559e+01</td>\n",
       "      <td>-5.683171e+00</td>\n",
       "      <td>-1.137433e+02</td>\n",
       "      <td>-2.616051e+01</td>\n",
       "      <td>-4.355724e+01</td>\n",
       "      <td>-7.321672e+01</td>\n",
       "      <td>-1.343407e+01</td>\n",
       "      <td>...</td>\n",
       "      <td>-3.483038e+01</td>\n",
       "      <td>-1.093314e+01</td>\n",
       "      <td>-4.480774e+01</td>\n",
       "      <td>-2.836627e+00</td>\n",
       "      <td>-1.029540e+01</td>\n",
       "      <td>-2.604551e+00</td>\n",
       "      <td>-2.256568e+01</td>\n",
       "      <td>-1.543008e+01</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>54201.500000</td>\n",
       "      <td>-9.203734e-01</td>\n",
       "      <td>-5.985499e-01</td>\n",
       "      <td>-8.903648e-01</td>\n",
       "      <td>-8.486401e-01</td>\n",
       "      <td>-6.915971e-01</td>\n",
       "      <td>-7.682956e-01</td>\n",
       "      <td>-5.540759e-01</td>\n",
       "      <td>-2.086297e-01</td>\n",
       "      <td>-6.430976e-01</td>\n",
       "      <td>...</td>\n",
       "      <td>-2.283949e-01</td>\n",
       "      <td>-5.423504e-01</td>\n",
       "      <td>-1.618463e-01</td>\n",
       "      <td>-3.545861e-01</td>\n",
       "      <td>-3.171451e-01</td>\n",
       "      <td>-3.269839e-01</td>\n",
       "      <td>-7.083953e-02</td>\n",
       "      <td>-5.295979e-02</td>\n",
       "      <td>5.600000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>84692.000000</td>\n",
       "      <td>1.810880e-02</td>\n",
       "      <td>6.548556e-02</td>\n",
       "      <td>1.798463e-01</td>\n",
       "      <td>-1.984653e-02</td>\n",
       "      <td>-5.433583e-02</td>\n",
       "      <td>-2.741871e-01</td>\n",
       "      <td>4.010308e-02</td>\n",
       "      <td>2.235804e-02</td>\n",
       "      <td>-5.142873e-02</td>\n",
       "      <td>...</td>\n",
       "      <td>-2.945017e-02</td>\n",
       "      <td>6.781943e-03</td>\n",
       "      <td>-1.119293e-02</td>\n",
       "      <td>4.097606e-02</td>\n",
       "      <td>1.659350e-02</td>\n",
       "      <td>-5.213911e-02</td>\n",
       "      <td>1.342146e-03</td>\n",
       "      <td>1.124383e-02</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>139320.500000</td>\n",
       "      <td>1.315642e+00</td>\n",
       "      <td>8.037239e-01</td>\n",
       "      <td>1.027196e+00</td>\n",
       "      <td>7.433413e-01</td>\n",
       "      <td>6.119264e-01</td>\n",
       "      <td>3.985649e-01</td>\n",
       "      <td>5.704361e-01</td>\n",
       "      <td>3.273459e-01</td>\n",
       "      <td>5.971390e-01</td>\n",
       "      <td>...</td>\n",
       "      <td>1.863772e-01</td>\n",
       "      <td>5.285536e-01</td>\n",
       "      <td>1.476421e-01</td>\n",
       "      <td>4.395266e-01</td>\n",
       "      <td>3.507156e-01</td>\n",
       "      <td>2.409522e-01</td>\n",
       "      <td>9.104512e-02</td>\n",
       "      <td>7.827995e-02</td>\n",
       "      <td>77.165000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>172792.000000</td>\n",
       "      <td>2.454930e+00</td>\n",
       "      <td>2.205773e+01</td>\n",
       "      <td>9.382558e+00</td>\n",
       "      <td>1.687534e+01</td>\n",
       "      <td>3.480167e+01</td>\n",
       "      <td>7.330163e+01</td>\n",
       "      <td>1.205895e+02</td>\n",
       "      <td>2.000721e+01</td>\n",
       "      <td>1.559499e+01</td>\n",
       "      <td>...</td>\n",
       "      <td>2.720284e+01</td>\n",
       "      <td>1.050309e+01</td>\n",
       "      <td>2.252841e+01</td>\n",
       "      <td>4.584549e+00</td>\n",
       "      <td>7.519589e+00</td>\n",
       "      <td>3.517346e+00</td>\n",
       "      <td>3.161220e+01</td>\n",
       "      <td>3.384781e+01</td>\n",
       "      <td>25691.160000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>8 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                Time            V1            V2            V3            V4  \\\n",
       "count  284807.000000  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   \n",
       "mean    94813.859575  3.919560e-15  5.688174e-16 -8.769071e-15  2.782312e-15   \n",
       "std     47488.145955  1.958696e+00  1.651309e+00  1.516255e+00  1.415869e+00   \n",
       "min         0.000000 -5.640751e+01 -7.271573e+01 -4.832559e+01 -5.683171e+00   \n",
       "25%     54201.500000 -9.203734e-01 -5.985499e-01 -8.903648e-01 -8.486401e-01   \n",
       "50%     84692.000000  1.810880e-02  6.548556e-02  1.798463e-01 -1.984653e-02   \n",
       "75%    139320.500000  1.315642e+00  8.037239e-01  1.027196e+00  7.433413e-01   \n",
       "max    172792.000000  2.454930e+00  2.205773e+01  9.382558e+00  1.687534e+01   \n",
       "\n",
       "                 V5            V6            V7            V8            V9  \\\n",
       "count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   \n",
       "mean  -1.552563e-15  2.010663e-15 -1.694249e-15 -1.927028e-16 -3.137024e-15   \n",
       "std    1.380247e+00  1.332271e+00  1.237094e+00  1.194353e+00  1.098632e+00   \n",
       "min   -1.137433e+02 -2.616051e+01 -4.355724e+01 -7.321672e+01 -1.343407e+01   \n",
       "25%   -6.915971e-01 -7.682956e-01 -5.540759e-01 -2.086297e-01 -6.430976e-01   \n",
       "50%   -5.433583e-02 -2.741871e-01  4.010308e-02  2.235804e-02 -5.142873e-02   \n",
       "75%    6.119264e-01  3.985649e-01  5.704361e-01  3.273459e-01  5.971390e-01   \n",
       "max    3.480167e+01  7.330163e+01  1.205895e+02  2.000721e+01  1.559499e+01   \n",
       "\n",
       "       ...           V21           V22           V23           V24  \\\n",
       "count  ...  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   \n",
       "mean   ...  1.537294e-16  7.959909e-16  5.367590e-16  4.458112e-15   \n",
       "std    ...  7.345240e-01  7.257016e-01  6.244603e-01  6.056471e-01   \n",
       "min    ... -3.483038e+01 -1.093314e+01 -4.480774e+01 -2.836627e+00   \n",
       "25%    ... -2.283949e-01 -5.423504e-01 -1.618463e-01 -3.545861e-01   \n",
       "50%    ... -2.945017e-02  6.781943e-03 -1.119293e-02  4.097606e-02   \n",
       "75%    ...  1.863772e-01  5.285536e-01  1.476421e-01  4.395266e-01   \n",
       "max    ...  2.720284e+01  1.050309e+01  2.252841e+01  4.584549e+00   \n",
       "\n",
       "                V25           V26           V27           V28         Amount  \\\n",
       "count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  284807.000000   \n",
       "mean   1.453003e-15  1.699104e-15 -3.660161e-16 -1.206049e-16      88.349619   \n",
       "std    5.212781e-01  4.822270e-01  4.036325e-01  3.300833e-01     250.120109   \n",
       "min   -1.029540e+01 -2.604551e+00 -2.256568e+01 -1.543008e+01       0.000000   \n",
       "25%   -3.171451e-01 -3.269839e-01 -7.083953e-02 -5.295979e-02       5.600000   \n",
       "50%    1.659350e-02 -5.213911e-02  1.342146e-03  1.124383e-02      22.000000   \n",
       "75%    3.507156e-01  2.409522e-01  9.104512e-02  7.827995e-02      77.165000   \n",
       "max    7.519589e+00  3.517346e+00  3.161220e+01  3.384781e+01   25691.160000   \n",
       "\n",
       "               Class  \n",
       "count  284807.000000  \n",
       "mean        0.001727  \n",
       "std         0.041527  \n",
       "min         0.000000  \n",
       "25%         0.000000  \n",
       "50%         0.000000  \n",
       "75%         0.000000  \n",
       "max         1.000000  \n",
       "\n",
       "[8 rows x 31 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(284807, 31)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Time      0\n",
       "V1        0\n",
       "V2        0\n",
       "V3        0\n",
       "V4        0\n",
       "V5        0\n",
       "V6        0\n",
       "V7        0\n",
       "V8        0\n",
       "V9        0\n",
       "V10       0\n",
       "V11       0\n",
       "V12       0\n",
       "V13       0\n",
       "V14       0\n",
       "V15       0\n",
       "V16       0\n",
       "V17       0\n",
       "V18       0\n",
       "V19       0\n",
       "V20       0\n",
       "V21       0\n",
       "V22       0\n",
       "V23       0\n",
       "V24       0\n",
       "V25       0\n",
       "V26       0\n",
       "V27       0\n",
       "V28       0\n",
       "Amount    0\n",
       "Class     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    284315\n",
       "1       492\n",
       "Name: Class, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Class.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>...</th>\n",
       "      <th>V20</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Amount</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Class</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>94838.202258</td>\n",
       "      <td>0.008258</td>\n",
       "      <td>-0.006271</td>\n",
       "      <td>0.012171</td>\n",
       "      <td>-0.007860</td>\n",
       "      <td>0.005453</td>\n",
       "      <td>0.002419</td>\n",
       "      <td>0.009637</td>\n",
       "      <td>-0.000987</td>\n",
       "      <td>0.004467</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.000644</td>\n",
       "      <td>-0.001235</td>\n",
       "      <td>-0.000024</td>\n",
       "      <td>0.000070</td>\n",
       "      <td>0.000182</td>\n",
       "      <td>-0.000072</td>\n",
       "      <td>-0.000089</td>\n",
       "      <td>-0.000295</td>\n",
       "      <td>-0.000131</td>\n",
       "      <td>88.291022</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>80746.806911</td>\n",
       "      <td>-4.771948</td>\n",
       "      <td>3.623778</td>\n",
       "      <td>-7.033281</td>\n",
       "      <td>4.542029</td>\n",
       "      <td>-3.151225</td>\n",
       "      <td>-1.397737</td>\n",
       "      <td>-5.568731</td>\n",
       "      <td>0.570636</td>\n",
       "      <td>-2.581123</td>\n",
       "      <td>...</td>\n",
       "      <td>0.372319</td>\n",
       "      <td>0.713588</td>\n",
       "      <td>0.014049</td>\n",
       "      <td>-0.040308</td>\n",
       "      <td>-0.105130</td>\n",
       "      <td>0.041449</td>\n",
       "      <td>0.051648</td>\n",
       "      <td>0.170575</td>\n",
       "      <td>0.075667</td>\n",
       "      <td>122.211321</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2 rows × 30 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "               Time        V1        V2        V3        V4        V5  \\\n",
       "Class                                                                   \n",
       "0      94838.202258  0.008258 -0.006271  0.012171 -0.007860  0.005453   \n",
       "1      80746.806911 -4.771948  3.623778 -7.033281  4.542029 -3.151225   \n",
       "\n",
       "             V6        V7        V8        V9  ...       V20       V21  \\\n",
       "Class                                          ...                       \n",
       "0      0.002419  0.009637 -0.000987  0.004467  ... -0.000644 -0.001235   \n",
       "1     -1.397737 -5.568731  0.570636 -2.581123  ...  0.372319  0.713588   \n",
       "\n",
       "            V22       V23       V24       V25       V26       V27       V28  \\\n",
       "Class                                                                         \n",
       "0     -0.000024  0.000070  0.000182 -0.000072 -0.000089 -0.000295 -0.000131   \n",
       "1      0.014049 -0.040308 -0.105130  0.041449  0.051648  0.170575  0.075667   \n",
       "\n",
       "           Amount  \n",
       "Class              \n",
       "0       88.291022  \n",
       "1      122.211321  \n",
       "\n",
       "[2 rows x 30 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"Class\").mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAggAAAFzCAYAAABb8fH8AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3dfZBV1Znv8e8DqDjx5YKIF2gcVDDhxZs2tC8xdS1fSlGSoAQdMRMkghozkgrMiyFO3WDpGHE0g2Yw3tELI5iJxJdRMGpMj0xCxiRiY9oLyFVIJNJIkAABNQVCs+4fvRsPrLZtsA/94vdTdeqc8+y91lnbsulfr73XPpFSQpIkqVSXth6AJElqfwwIkiQpY0CQJEkZA4IkScoYECRJUsaAIEmSMt3aegDtRa9evdKAAQPaehiSJB0wS5Ys+UNK6eimthkQCgMGDKCmpqathyFJ0gETEb97v22eYpAkSRkDQge3Zs0azj77bAYPHszQoUO56667AKitreX000+nsrKSqqoqFi9evEe7119/ncMOO4w77rhjd+2CCy7gk5/8JEOHDuXaa6+lvr4egEWLFvGpT32Kbt268cgjj+zRT9euXamsrKSyspJRo0Y1Ocbt27dz2WWXMXDgQE477TRWr17div8FJEnlYEDo4Lp168Z3vvMdVqxYwa9+9SvuvvtuXn75Za6//nqmTZtGbW0tN910E9dff/0e7aZMmcKFF164R+2hhx7ipZdeYtmyZWzYsIGHH34YgGOPPZb777+fL37xi9nnH3roodTW1lJbW8uCBQuaHOOsWbPo0aMHq1atYsqUKXzjG99opaOXJJWL1yB0cH369KFPnz4AHH744QwePJi1a9cSEWzduhWALVu20Ldv391tHn/8cY4//ng+9rGP7dHXEUccAcDOnTt59913iQig4foMgC5d9i9Pzp8/nxtvvBGASy65hEmTJpFS2t2/JHVUO3bsoK6ujm3btrX1UJrVvXt3KioqOOigg1rcxoDQiaxevZpf//rXnHbaadx5552MGDGCv/3bv2XXrl384he/AOCdd97htttuo7q6eo/TC41GjBjB4sWLufDCC7nkkks+8DO3bdtGVVUV3bp1Y+rUqVx88cXZPmvXrqV///5Aw4zHkUceycaNG+nVq9eHPGJJalt1dXUcfvjhDBgwoN3+0ZNSYuPGjdTV1XHccce1uJ2nGDqJt99+mzFjxnDnnXdyxBFHcM899zBjxgzWrFnDjBkzmDhxIgDTpk1jypQpHHbYYU3288wzz7Bu3Tq2b9/OwoULP/BzX3/9dWpqavjBD37A5MmT+c1vfpPt09Q3hrbXHyRJ2hfbtm3jqKOOatf/pkUERx111D7PchgQOoEdO3YwZswY/vIv/5IvfOELAMyZM2f360svvXT3RYrPP/88119/PQMGDODOO+/k29/+NjNnztyjv+7duzNq1Cjmz5//gZ/deOri+OOP56yzzuLXv/51tk9FRQVr1qwBGk5fbNmyhZ49e+7/AUtSO9Kew0Gj/RmjAaGDSykxceJEBg8ezF//9V/vrvft25ef/exnACxcuJBBgwYB8POf/5zVq1ezevVqJk+ezA033MCkSZN4++23WbduHdDwS/ypp57iE5/4RLOfvXnzZrZv3w7AH/7wB5577jmGDBmS7Tdq1CjmzJkDwCOPPMI555zTIX6gJKm1/P73v2fs2LGccMIJDBkyhJEjR/Lqq68ybNiwth7a+/IahA7uueee44EHHuCkk06isrISgG9/+9vcd999fP3rX2fnzp10796de++9t9l+3nnnHUaNGsX27dupr6/nnHPO4dprrwXghRdeYPTo0WzevJknnniCadOmsXz5clasWMFXvvIVunTpwq5du5g6derugPCtb32LqqoqRo0axcSJExk3bhwDBw6kZ8+ezJs3r7z/USSpHUkpMXr0aMaPH7/737/a2lrWr1/fxiNrXjR1fvijqKqqKnknRUnSvlixYgWDBw9udp+FCxdy4403smjRoj3qq1ev5nOf+xzLli1j9erVjBs3jnfeeQeAmTNncsYZZ7Bu3Touu+wytm7dys6dO7nnnns444wzmDhxIjU1NUQEEyZMYMqUKfs11ohYklKqamp/ZxAkSSqjZcuWMXz48Gb36d27N9XV1XTv3p2VK1dy+eWX774AfMSIEfz93/899fX1/OlPf6K2tpa1a9eybNkyAP74xz+WZdwGBEmS2tiOHTuYNGkStbW1dO3alVdffRWAU045hQkTJrBjxw4uvvhiKisrOf744/ntb3/L1772NT772c9y/vnnl2VMXqQoSVIZDR06lCVLljS7z4wZMzjmmGN46aWXqKmp4d133wXgzDPPZNGiRfTr149x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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,6))\n",
    "ax = sns.countplot(x=df.Class, hue=df.Class)\n",
    "for p in ax.patches:\n",
    "    ax.annotate((p.get_height()), (p.get_x()+0.10, p.get_height()+4000));"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1efe5ef3be0>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,6))\n",
    "sns.distplot(df.Amount, kde = False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NGyEoz9fJQ0E"
   },
   "source": [
    "### Data Cleaning\n",
    "Check Missing Values and Outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "BvpEPuGAMuHa"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " count of outlier in Time column by Class 0= 0 \n",
      " count of outlier in Time column by Class 1= 0 \n",
      "\n",
      " count of outlier in V1 column by Class 0= 6912 \n",
      " count of outlier in V1 column by Class 1= 52 \n",
      "\n",
      " count of outlier in V2 column by Class 0= 13327 \n",
      " count of outlier in V2 column by Class 1= 46 \n",
      "\n",
      " count of outlier in V3 column by Class 0= 3095 \n",
      " count of outlier in V3 column by Class 1= 53 \n",
      "\n",
      " count of outlier in V4 column by Class 0= 10918 \n",
      " count of outlier in V4 column by Class 1= 0 \n",
      "\n",
      " count of outlier in V5 column by Class 0= 12124 \n",
      " count of outlier in V5 column by Class 1= 45 \n",
      "\n",
      " count of outlier in V6 column by Class 0= 22829 \n",
      " count of outlier in V6 column by Class 1= 15 \n",
      "\n",
      " count of outlier in V7 column by Class 0= 8668 \n",
      " count of outlier in V7 column by Class 1= 30 \n",
      "\n",
      " count of outlier in V8 column by Class 0= 23974 \n",
      " count of outlier in V8 column by Class 1= 98 \n",
      "\n",
      " count of outlier in V9 column by Class 0= 8090 \n",
      " count of outlier in V9 column by Class 1= 17 \n",
      "\n",
      " count of outlier in V10 column by Class 0= 9128 \n",
      " count of outlier in V10 column by Class 1= 19 \n",
      "\n",
      " count of outlier in V11 column by Class 0= 492 \n",
      " count of outlier in V11 column by Class 1= 10 \n",
      "\n",
      " count of outlier in V12 column by Class 0= 15054 \n",
      " count of outlier in V12 column by Class 1= 6 \n",
      "\n",
      " count of outlier in V13 column by Class 0= 3365 \n",
      " count of outlier in V13 column by Class 1= 0 \n",
      "\n",
      " count of outlier in V14 column by Class 0= 13800 \n",
      " count of outlier in V14 column by Class 1= 4 \n",
      "\n",
      " count of outlier in V15 column by Class 0= 2883 \n",
      " count of outlier in V15 column by Class 1= 8 \n",
      "\n",
      " count of outlier in V16 column by Class 0= 7908 \n",
      " count of outlier in V16 column by Class 1= 0 \n",
      "\n",
      " count of outlier in V17 column by Class 0= 7038 \n",
      " count of outlier in V17 column by Class 1= 0 \n",
      "\n",
      " count of outlier in V18 column by Class 0= 7345 \n",
      " count of outlier in V18 column by Class 1= 0 \n",
      "\n",
      " count of outlier in V19 column by Class 0= 10131 \n",
      " count of outlier in V19 column by Class 1= 5 \n",
      "\n",
      " count of outlier in V20 column by Class 0= 27649 \n",
      " count of outlier in V20 column by Class 1= 41 \n",
      "\n",
      " count of outlier in V21 column by Class 0= 14273 \n",
      " count of outlier in V21 column by Class 1= 49 \n",
      "\n",
      " count of outlier in V22 column by Class 0= 1289 \n",
      " count of outlier in V22 column by Class 1= 24 \n",
      "\n",
      " count of outlier in V23 column by Class 0= 18411 \n",
      " count of outlier in V23 column by Class 1= 59 \n",
      "\n",
      " count of outlier in V24 column by Class 0= 4771 \n",
      " count of outlier in V24 column by Class 1= 3 \n",
      "\n",
      " count of outlier in V25 column by Class 0= 5314 \n",
      " count of outlier in V25 column by Class 1= 36 \n",
      "\n",
      " count of outlier in V26 column by Class 0= 5603 \n",
      " count of outlier in V26 column by Class 1= 2 \n",
      "\n",
      " count of outlier in V27 column by Class 0= 39018 \n",
      " count of outlier in V27 column by Class 1= 70 \n",
      "\n",
      " count of outlier in V28 column by Class 0= 30206 \n",
      " count of outlier in V28 column by Class 1= 46 \n",
      "\n",
      " count of outlier in Amount column by Class 0= 31862 \n",
      " count of outlier in Amount column by Class 1= 69 \n",
      "\n",
      " count of outlier in Class column by Class 0= 0 \n",
      " count of outlier in Class column by Class 1= 0 \n",
      "\n"
     ]
    }
   ],
   "source": [
    "for col in df:\n",
    "    for counter in range(len(df[\"Class\"].unique())):\n",
    "        \n",
    "        Q1 = df[df[\"Class\"]==counter][col].quantile(0.25)\n",
    "        Q3 = df[df[\"Class\"]==counter][col].quantile(0.75)\n",
    "        IQR = Q3-Q1\n",
    "        lower_lim = Q1-1.5*IQR\n",
    "        upper_lim = Q3+1.5*IQR\n",
    "        print(f\" count of outlier in {col} column by Class {counter}= {df[df['Class']==counter][(df[df['Class']==counter][col] < lower_lim) | (df[df['Class']==counter][col] > upper_lim)][col].count()} \")\n",
    "    print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'X2' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-13-fc316cca30c5>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mX2\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'X2' is not defined"
     ]
    }
   ],
   "source": [
    "X2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[[\"Time\", \"Amount\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1efe5fdff40>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(y = df.Time, x = df.Class)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(y = df.Amount, x = df.Class);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>...</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Amount</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0.333333</th>\n",
       "      <td>65096.333333</td>\n",
       "      <td>-0.640588</td>\n",
       "      <td>-0.336101</td>\n",
       "      <td>-0.466159</td>\n",
       "      <td>-0.602006</td>\n",
       "      <td>-0.455403</td>\n",
       "      <td>-0.602144</td>\n",
       "      <td>-0.333849</td>\n",
       "      <td>-0.126824</td>\n",
       "      <td>-0.412954</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.160147</td>\n",
       "      <td>-0.357089</td>\n",
       "      <td>-0.104434</td>\n",
       "      <td>-0.152608</td>\n",
       "      <td>-0.216690</td>\n",
       "      <td>-0.238731</td>\n",
       "      <td>-0.042315</td>\n",
       "      <td>-0.032122</td>\n",
       "      <td>9.99</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0.666667</th>\n",
       "      <td>128593.000000</td>\n",
       "      <td>1.177454</td>\n",
       "      <td>0.520578</td>\n",
       "      <td>0.723818</td>\n",
       "      <td>0.461542</td>\n",
       "      <td>0.364415</td>\n",
       "      <td>0.125498</td>\n",
       "      <td>0.359008</td>\n",
       "      <td>0.197366</td>\n",
       "      <td>0.356571</td>\n",
       "      <td>...</td>\n",
       "      <td>0.107777</td>\n",
       "      <td>0.339791</td>\n",
       "      <td>0.086350</td>\n",
       "      <td>0.323565</td>\n",
       "      <td>0.242539</td>\n",
       "      <td>0.141397</td>\n",
       "      <td>0.049289</td>\n",
       "      <td>0.041034</td>\n",
       "      <td>50.00</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                   Time        V1        V2        V3        V4        V5  \\\n",
       "0.333333   65096.333333 -0.640588 -0.336101 -0.466159 -0.602006 -0.455403   \n",
       "0.666667  128593.000000  1.177454  0.520578  0.723818  0.461542  0.364415   \n",
       "\n",
       "                V6        V7        V8        V9  ...       V21       V22  \\\n",
       "0.333333 -0.602144 -0.333849 -0.126824 -0.412954  ... -0.160147 -0.357089   \n",
       "0.666667  0.125498  0.359008  0.197366  0.356571  ...  0.107777  0.339791   \n",
       "\n",
       "               V23       V24       V25       V26       V27       V28  Amount  \\\n",
       "0.333333 -0.104434 -0.152608 -0.216690 -0.238731 -0.042315 -0.032122    9.99   \n",
       "0.666667  0.086350  0.323565  0.242539  0.141397  0.049289  0.041034   50.00   \n",
       "\n",
       "          Class  \n",
       "0.333333    0.0  \n",
       "0.666667    0.0  \n",
       "\n",
       "[2 rows x 31 columns]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "quartiles = df.quantile(q = [1/3, 2/3])\n",
    "quartiles"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fraud(x, p,d):\n",
    "    if x <= d[p][1/3]:\n",
    "        return 1\n",
    "    elif x <= d[p][2/3]:\n",
    "        return 2\n",
    "    else:\n",
    "        return 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "d = df.Amount.apply(fraud, args = (\"Amount\", quartiles))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAncAAAHgCAYAAADHQUsEAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3de5SXZb338fcXBkVLjaMbGBIUksMgIwfDfEKLFILyhCaYZ9w+lZZmZdlenRdZdvCwNVuSpfIYyPLwwLOXGUjiqbYIiIiwHUhIxhMHU0kX6uD1/DE30wwNOOjM/IaL92ut3/rdv+t3X9d8b9fN+JnrPkVKCUmSJOWhXakLkCRJUvMx3EmSJGXEcCdJkpQRw50kSVJGDHeSJEkZMdxJkiRlpKzUBbQVXbt2TX369Cl1GZIkSe9q8eLFG1NK3Rr7znBX6NOnD4sWLSp1GZIkSe8qIv62o+88LCtJkpQRw50kSVJGWizcRcRvI2J9RCyv19Y5IuZFxKrivVO97y6PiNUR8XREjK3XPjwiniy+uzYiomjfOyJuL9ofjYg+9fqcXfyMVRFxdkttoyRJUlvTkufc3QxcB9xar+1bwPyU0k8i4lvF529GxCBgEjAY6AncFxEfSSltBW4ALgD+G7gHGAf8AZgC/D2l1C8iJgE/BU6LiM7A94ARQAIWR8SclNLfW3BbJUnSbubtt9+murqaLVu2lLqUHerYsSPl5eV06NChyX1aLNyllB6sP5tWOAE4pli+BVgAfLNon5lSehNYExGrgSMiYi2wf0rpLwARcStwIrXh7gTg+8VYdwDXFbN6Y4F5KaWXiz7zqA2EM5p7GyVJ0u6rurqa/fbbjz59+lAcGGxTUkps2rSJ6upq+vbt2+R+rX3O3YEppRcAivfuRXsvYF299aqLtl7F8vbtDfqklGqAV4EuOxlLkiSpzpYtW+jSpUubDHYAEUGXLl12eWaxrVxQ0dh/1bST9vfap+EPjbggIhZFxKINGzY0qVBJkpSPthrstnkv9bV2uHspInoAFO/ri/ZqoHe99cqB54v28kbaG/SJiDLgAODlnYz1L1JKN6aURqSURnTr1uh9ACVJ0h7uxRdfZNKkSRxyyCEMGjSI8ePHU1VVRUVFRalLa1Rrh7s5wLarV88GZtdrn1RcAdsX6A8sLA7dbo6IUcX5dGdt12fbWKcAf0opJeCPwHER0am4Gve4ok2SJGmXpJQ46aSTOOaYY/jrX//KihUr+PGPf8xLL71U6tJ2qCVvhTID+AtwaERUR8QU4CfAsRGxCji2+ExK6SlgFrACuBe4sLhSFuCLwG+A1cBfqb2YAuAmoEtx8cWl1F55S3EhxY+Ax4rXD7ddXCFJkrQr7r//fjp06MAXvvCFurbKykp69/7nQcK1a9fy8Y9/nGHDhjFs2DD+/Oc/A/DCCy8wevRoKisrqaio4KGHHmLr1q2cc845VFRUMGTIEK666qpmr7nFwl1KaXJKqUdKqUNKqTyldFNKaVNKaUxKqX/x/nK99aemlA5JKR2aUvpDvfZFKaWK4ruLitk5UkpbUkqnppT6pZSOSCk9U6/Pb4v2fiml37XUNrY111xzDRUVFQwePJirr74agNNOO43KykoqKyvp06cPlZWVdetfccUV9OvXj0MPPZQ//vGfk5tvvfUWF1xwAR/5yEcYMGAAd955JwA333wz3bp1qxvvN7/5TetuoCRJrWz58uUMHz58p+t0796defPmsWTJEm6//Xa+8pWvAPD73/+esWPHsnTpUp544gkqKytZunQpzz33HMuXL+fJJ5/k3HPPbfaafbZsJpYvX860adNYuHAhe+21F+PGjWPChAncfvvtdet87Wtf44ADDgBgxYoVzJw5k6eeeornn3+eT33qU1RVVdG+fXumTp1K9+7dqaqq4p133uHll/858Xnaaadx3XXXtfr2SZLUVr399ttcdNFFLF26lPbt21NVVQXAyJEjOe+883j77bc58cQTqays5OCDD+aZZ57hy1/+MhMmTOC4445r9nraytWyep9WrlzJqFGj2HfffSkrK+Poo4/m7rvvrvs+pcSsWbOYPHkyALNnz2bSpEnsvffe9O3bl379+rFw4UIAfvvb33L55ZcD0K5dO7p27dr6GyRJUhswePBgFi9evNN1rrrqKg488ECeeOIJFi1axFtvvQXA6NGjefDBB+nVqxdnnnkmt956K506deKJJ57gmGOO4frrr+f8889v9poNd5moqKjgwQcfZNOmTbzxxhvcc889rFv3z9v9PfTQQxx44IH0798fgOeee67B+QLl5eU899xzvPLKKwB85zvfYdiwYZx66qkNThq98847OeywwzjllFMajC9JUo4++clP8uabbzJt2rS6tscee4y//e1vdZ9fffVVevToQbt27Zg+fTpbt9ZeNvC3v/2N7t278+///u9MmTKFJUuWsHHjRt555x0mTpzIj370I5YsWdLsNRvuMjFw4EC++c1vcuyxxzJu3DiGDh1KWdk/j7rPmDGjbtYOamfythcR1NTUUF1dzVFHHcWSJUs48sgj+frXvw7AZz/7WdauXcuyZcv41Kc+xdln+9heSVLeIoK7776befPmccghhzB48GC+//3v07Nnz7p1vvSlL3HLLbcwatQoqqqq+MAHPgDAggULqKys5PDDD+fOO+/k4osv5rnnnuOYY46hsrKSc845hyuuuKL5a27sf/J7ohEjRqRFixaVuoxm8+1vf5vy8nK+9KUvUVNTQ69evVi8eDHl5bW3Ddy2M207/Dp27Fi+//3vM2rUKD74wQ+yefNm2rVrx7p16xg3bhxPPfVUg/G3bt1K586defXVV1t3wyRJaiYrV65k4MCBpS7jXTVWZ0QsTimNaGx9Z+4ysn597T2hn332We666666mbr77ruPAQMG1AU7gOOPP56ZM2fy5ptvsmbNGlatWsURRxxBRPDZz36WBQsWADB//nwGDRoE1F7Svc2cOXN2i38QkiTtabxaNiMTJ05k06ZNdOjQgeuvv55OnToBMHPmzAaHZKH2BNHPfe5zDBo0iLKyMq6//nrat28PwE9/+lPOPPNMLrnkErp168bvfld7N5lrr72WOXPmUFZWRufOnbn55ptbdfskSdK787BsIbfDspIkaedyPSzrzF0LGP6NW0tdwm5h8c/OKnUJkiRlx3PuJEmSMmK4kyRJyojhTpIkqYTuvfdeDj30UPr168dPfvKT9z2e59xJkiTR/OfMN+Xc8q1bt3LhhRcyb948ysvLGTlyJMcff3zdbcjeC2fuJEmSSmThwoX069ePgw8+mL322otJkyYxe/bs9zWm4U7SbuWaa66hoqKCwYMHc/XVVwO1z0I+7LDDqKys5LjjjuP555+vW3/ZsmUceeSRDB48mCFDhrBlyxY2b95MZWVl3atr165ccsklDX7OHXfcQUTgLZJUCruyn69du5Z99tmnbn/+whe+APCu+/msWbMYNGgQgwcP5vTTT2/9jRSw42e9vx8elpW021i+fDnTpk1j4cKF7LXXXowbN44JEybwjW98gx/96EdA7c22f/jDH/LrX/+a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\n",
      "text/plain": [
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,8))\n",
    "ax = sns.countplot(d, hue = df.Class)\n",
    "for p in ax.patches:\n",
    "    ax.annotate((p.get_height()), (p.get_x()+0.15, p.get_height()+100));"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "d = df.Time.apply(fraud, args = (\"Time\", quartiles))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAnEAAAHgCAYAAADKXztDAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3defRddX3v/9c7gyIIXoZgQ75AAK2GDIRJsS5TlWul+HNAsA2lMopWqVNrtfy6VqVlUdSr16EgzjJcBGT6SV2WNkvlYgeJAYOBcFEqKEGUgKKiTZg+vz9ykptgEr6U78k3n+TxWOus7zmfs/fOZy/2Nzyzz9nnVGstAAD0ZcJ4TwAAgCdOxAEAdEjEAQB0SMQBAHRIxAEAdEjEAQB0aNJ4T2BT22WXXdr06dPHexoAAI/r+uuvv7e1NmV9z211ETd9+vQsWrRovKcBAPC4quoHG3rOy6kAAB0ScQAAHRJxAAAd2ureEwcAbF0eeuihLFu2LCtWrBjvqWzQNttsk5GRkUyePHnU64g4AGCLtmzZsmy//faZPn16qmq8p/MbWmu57777smzZsuy1116jXs/LqQDAFm3FihXZeeedN8uAS5Kqys477/yEzxSKOABgi7e5Btxq/5X5iTgAYKv34x//OPPnz88+++yTfffdN4cffni++93vZtasWeM9tQ3ynjgAYKvWWssRRxyR4447LhdffHGSZPHixfnJT34yzjPbOGfiAICt2te//vVMnjw5f/Inf7JmbO7cudl9993XPL7jjjvyohe9KAcccEAOOOCA/Nu//VuS5O677868efMyd+7czJo1K9/4xjfyyCOP5Pjjj8+sWbMye/bsfPjDHx7KvEUcAIyDj370o5k1a1ZmzpyZj3zkI+s898EPfjBVlXvvvTfJqo/IOO644zJ79uzMmDEjZ5555ppl/+qv/iq77757nv70p6+zjWuvvTYHHHBAJk2alMsuu2z4O9Sxm266KQceeOBGl9l1112zYMGC3HDDDbnkkkvytre9LUnyhS98IS9/+cuzePHi3HjjjZk7d24WL16cu+66KzfddFOWLFmSE044YSjzFnGdeSK/9BdeeGHmzp275jZhwoQsXrw4yYZ/6X/wgx/k0EMPzZw5c/LiF784y5Yt2zQ7BmsZq+P8oosuyuzZszNnzpwcdthha9Y599xzM2XKlDXrfOYzn9m0O8hW76abbsqnP/3pLFy4MDfeeGO+/OUv53vf+16S5M4778yCBQuyxx57rFn+0ksvzcqVK7NkyZJcf/31+eQnP5k77rgjSfLKV74yCxcu/I0/Y4899si5556bP/qjP9ok+7Sle+ihh3LyySdn9uzZed3rXpelS5cmSQ4++OB8/vOfz2mnnZYlS5Zk++23z957753vf//7eetb35qrr746O+yww1DmJOI68kR/6Y855pgsXrw4ixcvzgUXXJDp06dn7ty5STb8S/+ud70rxx57bL7zne/kr//6r3Pqqadump2DgbE6zh9++OG8/e1vz9e//vV85zvfyZw5c3LWWWetWe8P//AP16z3hje8YZPvJ1u3W265JYcccki23XbbTJo0Kb/7u7+bK6+8Mknyzne+Mx/4wAfWuVqxqvKrX/0qDz/8cP7zP/8zT3nKU9aEwSGHHJKpU6f+xp8xffr0zJkzJxMm+F/945k5c2auv/76jS7z4Q9/OM985jNz4403ZtGiRXnwwQeTJPPmzcu1116badOm5fWvf33OP//87Ljjjrnxxhvz4he/OGefffbQ/o7xX7YjT/SXfm0XXXRRjj766DWPN/RLv3Tp0hx66KFJkpe85CX50pe+NIQ9gQ0bq+O8tZbWWn71q1+ltZZf/OIX2W233TbZfsDGzJo1K9dee23uu+++/PrXv85XvvKV3Hnnnbnqqqsybdq07Lfffussf9RRR2W77bbL1KlTs8cee+Rd73pXdtppp3Ga/ZbnpS99aVauXJlPf/rTa8a+9a1v5Qc/+MGaxz//+c8zderUTJgwIRdccEEeeeSRJKtewdp1111z8skn56STTsoNN9yQe++9N48++miOPPLInH766bnhhhuGMm8R15En+ku/tksuuWSdiNuQ/fbbL5dffnmS5Morr8wvf/nL3HfffWO2D/B4xuo4nzx5cs4555zMnj07u+22W5YuXZqTTjppzbKXX3555syZk6OOOip33nnn0PcL1jZjxoy85z3vycte9rIcdthh2W+//TJp0qScccYZ+du//dvfWH7hwoWZOHFifvSjH+X222/Phz70oXz/+98fh5lvmaoqV155ZRYsWJB99tknM2fOzGmnnbbOP/ze8pa35LzzzsshhxyS7373u9luu+2SJNdcc03mzp2b/fffP5dffnne/va356677sqLX/zizJ07N8cff/w672EcU6v/tbq13A488MDWs8985jNt//33by960Yvam970pvaOd7yjPe95z2v3339/a621Pffcsy1fvnyddb75zW+2WbNmrXd722233TqP77rrrnbEEUe0uXPntre97W1t2rRpa7YNm8pYHOcPPvhge+lLX9puu+229uijj7ZTTjmlnX766a211u699962YsWK1lpr55xzTnvJS16yifYM1u/UU09tH/nIR9qUKVPannvu2fbcc882ceLEtvvuu7e77767veUtb2nnn3/+muVPOOGEdskll6yzjcf+fb7acccd1y699NKhzn9zt3Tp0vGewqisb55JFrUNNI0zcZ1Zfar22muvzU477ZTp06fn9ttvz3777Zfp06dn2bJlOeCAA/LjH/94zToXX3zxqM7CJcluu+2WK664It/+9rdzxhlnJEme8YxnDGVfYEPG4jhffXHDPvvsk6rKH/zBH6z5SICdd945T33qU5MkJ5988uO+FwaG4Z577kmS/PCHP8wVV1yRY489Nvfcc0/uuOOO3HHHHRkZGckNN9yQ3/qt38oee+yRr33ta2veIvDNb34zz33uc8d5DxhvIq4zT+SXPkkeffTRXHrppZk/f/6otr/6dfwkOfPMM3PiiScOZ0dgI8biOJ82bVqWLl2a5cuXJ0kWLFiQGTNmJFn1uU6rXXXVVWvGYVM68sgjs+++++aVr3xlzj777Oy4444bXPaUU07JAw88kFmzZuXggw/OCSeckDlz5iRJ3v3ud2dkZCS//vWvMzIyktNOOy3Jqvd0jYyM5NJLL82b3vSmzJw5c1PsFpuQb2zozJFHHpn77rsvkydPftxf+mTV5wSNjIxk7733Xmf83e9+d77whS+s+aV/wxvekNNOOy3XXHNNTj311FRV5s2bl7PPPnuYuwPrNRbH+W677Zb3vve9mTdvXiZPnpw999wz5557bpLkYx/7WK666qpMmjQpO+2005px2JS+8Y1vbPT51R8hkiRPf/rTc+mll653uQ984AP5wAc+8BvjBx98sI+J2sLVqpdbtx4HHXRQW7Ro0Zhs68C/OH9MtrOlu/5/HDveU+BJcJyPjuO8b47z0en1OL/lllu6OOO+vnlW1fWttYPWt7yXUwEAOiTiAAA6JOIAAIbs6quvznOe85w861nPyvve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uD8AAJuNoUVca+1fsv73rCXJoRtY54wkZ6xnfFGSWesZX5FBBAIAbE18YwMAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHFukO++8My95yUsyY8aMzJw5Mx/96EeTJJdeemlmzpyZCRMmZNGiRWuWv/DCCzN37tw1twkTJmTx4sXjNX0YlRNPPDG77rprZs2atWbstNNOy7Rp09Ycy1/5yleSJAsXLlwztt9+++XKK68cr2nDE7K+4/wv/uIv8tznPjdz5szJEUcckfvvvz9JsmDBghx44IGZPXt2DjzwwHzta18br2lvEiKOLdKkSZPyoQ99KLfccku++c1v5uyzz87SpUsza9asXHHFFZk3b946yx9zzDFZvHhxFi9enAsuuCDTp0/P3Llzx2n2MDrHH398rr766t8Yf+c737nmeD788MOTJLNmzcqiRYuyePHiXH311XnTm96Uhx9+eFNPGZ6w9R3nL3vZy3LTTTflO9/5Tn77t387Z555ZpJkl112yT/8wz9kyZIlOe+88/L6179+PKa8yYg4tkhTp07NAQcckCTZfvvtM2PGjNx1112ZMWNGnvOc52x03YsuuihHH330ppgmPCnz5s3LTjvtNKplt91220yaNClJsmLFilTVMKcGY2Z9x/nv/d7vrTmeDznkkCxbtixJsv/++2e33XZLksycOTMrVqzIypUrN+2ENyERxxbvjjvuyLe//e08//nPH9Xyl1xyiYija2eddVbmzJmTE088MT/72c/WjF933XWZOXNmZs+enU984hNr/icIPfvc5z6X3//93/+N8csvvzz7779/nvrUp47DrDYNEccW7YEHHsiRRx6Zj3zkI9lhhx0ed/nrrrsu22677TrvvYCevPnNb85//Md/ZPHixZk6dWr+/M//fM1zz3/+83PzzTfnW9/6Vs4888ysWLFiHGcKT94ZZ5yRSZMm5Zhjjlln/Oabb8573vOefPKTnxynmW0a3UdcVR1WVbdW1W1V9ZfjPR82Hw899FCOPPLIHHPMMXnta187qnUuvvhiZ+Ho2jOf+cxMnDgxEyZMyF9/sBoAAAV4SURBVMknn5yFCxf+xjIzZszIdtttl5tuumkcZghj47zzzsuXv/zlXHjhheu8PWDZsmU54ogjcv7552efffYZxxkOX9cRV1UTk5yd5PeT7Jvk6Krad3xnxeagtZaTTjopM2bMyJ/92Z+Nap1HH300l156aebPnz/k2cHw3H333WvuX3nllWvOKt9+++1rLmT4wQ9+kFtvvTXTp08fjynCk3b11Vfn/e9/f6666qpsu+22a8bvv//+vOIVr8iZZ56ZF77wheM4w02j9zdEPC/Jba217ydJVV2c5NVJlo7rrBh3//qv/5oLLrggs2fPXnOV6d/93d9l5cqVeetb35rly5fnFa94RebOnZt/+qd/SpJce+21GRkZyd577z2eU4dRO/roo3PNNdfk3nvvzcjISP7mb/4m11xzTRYvXpyqyvTp09e8nPQv//Ived/73pfJkydnwoQJ+fjHP55ddtllnPcAHt/6jvMzzzwzK1euzMte9rIkqy5u+MQnPpGzzjort912W04//fScfvrpSZJ//ud/zq677jqeuzA01Vob7zn8l1XVUUkOa629YfD49Ume31r70w2tc9BBB7W1Px/syTjwL84fk+1s6a7c/n+M9xS6sMdfLxnvKayX43x0HOej4zjvm+N8dMbyOK+q61trB63vud7PxK3vGvnfqNKqemOSNw4ePlBVtw51Vqxjz2SXJPeO9zw2e+/1kQ89c5yPkuO8a47zURrb43zPDT3Re8QtS7L7Wo9HkvzosQu11j6V5FObalKsq6oWbehfEbClcJyzNXCcb166vrAhybeSPLuq9qqqpySZn+SqcZ4TAMDQdX0mrrX2cFX9aZJ/SjIxyedaazeP87QAAIau64hLktbaV5J8ZbznwUZ5KZutgeOcrYHjfDPS9dWpAABbq97fEwcAsFUScQxNVX2uqu6pKt/twxarqnavqq9X1S1VdXNVvX285wRjraq2qaqFVXXj4Dj/m/GeE15OZYiqal6SB5Kc31rzjfJskapqapKprbUbqmr7JNcneU1rzTfHsMWoVV9Oul1r7YGqmpzkX5K8vbX2zXGe2lbNmTiGprV2bZKfjvc8YJhaa3e31m4Y3P9lkluSTBvfWcHYaqs8MHg4eXBzFmiciTiAMVJV05Psn+S68Z0JjL2qmlhVi5Pck2RBa81xPs5EHMAYqKqnJ7k8yTtaa78Y7/nAWGutPdJam5tV3470vKryNplxJuIAnqTBe4QuT3Jha+2K8Z4PDFNr7f4k1yQ5bJynstUTcQBPwuAN359Ncktr7X+O93xgGKpqSlX9t8H9pyX570n+z/jOChHH0FTVRUn+PclzqmpZVZ003nOCIXhhktcneWlVLR7cDh/vScEYm5rk61X1naz63vIFrbUvj/Octno+YgQAoEPOxAEAdEjEAQB0SMQBAHRIxAEAdEjEAQB0SMQBrEdV7bzWR4b8uKruGtx/oKo+Pt7zA/ARIwCPo6pOS/JAa+2D4z0XgNWciQN4AqrqxVX15cH906rqvKr656q6o6peW1UfqKolVXX14Ou4UlUHVtX/rqrrq+qfqmrq+O4FsCUQcQBPzj5JXpHk1Un+V5Kvt9ZmJ/nPJK8YhNzfJzmqtXZgks8lOWO8JgtsOSaN9wQAOvePrbWHqmpJkolJrh6ML0kyPclzksxKsmDV16xmYpK7x2GewBZGxAE8OSuTpLX2aFU91P7vG40fzaq/YyvJza21F4zXBIEtk5dTAYbr1iRTquoFSVJVk6tq5jjPCdgCiDiAIWqtPZjkqCTvr6obkyxO8jvjOytgS+AjRgAAOuRMHABAh0QcAECHRBwAQIdEHABAh0QcAECHRBwAQIdEHABAh0QcAECH/n8UsIjphcTUbwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,8))\n",
    "ax = sns.countplot(d, hue = df.Class)\n",
    "for p in ax.patches:\n",
    "    ax.annotate((p.get_height()), (p.get_x()+0.15, p.get_height()+100));"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "tMOO7g-sMuHb"
   },
   "source": [
    "---\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Yf6VvH6WMuHb"
   },
   "source": [
    "## 2. Data Preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "OV28RJBeMuHb"
   },
   "source": [
    "#### Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(\"Class\", axis=1)\n",
    "y = df[\"Class\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "cellView": "form",
    "id": "9YbdPguXMuHb"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>...</th>\n",
       "      <th>V20</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Amount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-1.996583</td>\n",
       "      <td>-0.694242</td>\n",
       "      <td>-0.044075</td>\n",
       "      <td>1.672773</td>\n",
       "      <td>0.973366</td>\n",
       "      <td>-0.245117</td>\n",
       "      <td>0.347068</td>\n",
       "      <td>0.193679</td>\n",
       "      <td>0.082637</td>\n",
       "      <td>0.331128</td>\n",
       "      <td>...</td>\n",
       "      <td>0.326118</td>\n",
       "      <td>-0.024923</td>\n",
       "      <td>0.382854</td>\n",
       "      <td>-0.176911</td>\n",
       "      <td>0.110507</td>\n",
       "      <td>0.246585</td>\n",
       "      <td>-0.392170</td>\n",
       "      <td>0.330892</td>\n",
       "      <td>-0.063781</td>\n",
       "      <td>0.244964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-1.996583</td>\n",
       "      <td>0.608496</td>\n",
       "      <td>0.161176</td>\n",
       "      <td>0.109797</td>\n",
       "      <td>0.316523</td>\n",
       "      <td>0.043483</td>\n",
       "      <td>-0.061820</td>\n",
       "      <td>-0.063700</td>\n",
       "      <td>0.071253</td>\n",
       "      <td>-0.232494</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.089611</td>\n",
       "      <td>-0.307377</td>\n",
       "      <td>-0.880077</td>\n",
       "      <td>0.162201</td>\n",
       "      <td>-0.561131</td>\n",
       "      <td>0.320694</td>\n",
       "      <td>0.261069</td>\n",
       "      <td>-0.022256</td>\n",
       "      <td>0.044608</td>\n",
       "      <td>-0.342475</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.996562</td>\n",
       "      <td>-0.693500</td>\n",
       "      <td>-0.811578</td>\n",
       "      <td>1.169468</td>\n",
       "      <td>0.268231</td>\n",
       "      <td>-0.364572</td>\n",
       "      <td>1.351454</td>\n",
       "      <td>0.639776</td>\n",
       "      <td>0.207373</td>\n",
       "      <td>-1.378675</td>\n",
       "      <td>...</td>\n",
       "      <td>0.680975</td>\n",
       "      <td>0.337632</td>\n",
       "      <td>1.063358</td>\n",
       "      <td>1.456320</td>\n",
       "      <td>-1.138092</td>\n",
       "      <td>-0.628537</td>\n",
       "      <td>-0.288447</td>\n",
       "      <td>-0.137137</td>\n",
       "      <td>-0.181021</td>\n",
       "      <td>1.160686</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-1.996562</td>\n",
       "      <td>-0.493325</td>\n",
       "      <td>-0.112169</td>\n",
       "      <td>1.182516</td>\n",
       "      <td>-0.609727</td>\n",
       "      <td>-0.007469</td>\n",
       "      <td>0.936150</td>\n",
       "      <td>0.192071</td>\n",
       "      <td>0.316018</td>\n",
       "      <td>-1.262503</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.269855</td>\n",
       "      <td>-0.147443</td>\n",
       "      <td>0.007267</td>\n",
       "      <td>-0.304777</td>\n",
       "      <td>-1.941027</td>\n",
       "      <td>1.241904</td>\n",
       "      <td>-0.460217</td>\n",
       "      <td>0.155396</td>\n",
       "      <td>0.186189</td>\n",
       "      <td>0.140534</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1.996541</td>\n",
       "      <td>-0.591330</td>\n",
       "      <td>0.531541</td>\n",
       "      <td>1.021412</td>\n",
       "      <td>0.284655</td>\n",
       "      <td>-0.295015</td>\n",
       "      <td>0.071999</td>\n",
       "      <td>0.479302</td>\n",
       "      <td>-0.226510</td>\n",
       "      <td>0.744326</td>\n",
       "      <td>...</td>\n",
       "      <td>0.529939</td>\n",
       "      <td>-0.012839</td>\n",
       "      <td>1.100011</td>\n",
       "      <td>-0.220123</td>\n",
       "      <td>0.233250</td>\n",
       "      <td>-0.395202</td>\n",
       "      <td>1.041611</td>\n",
       "      <td>0.543620</td>\n",
       "      <td>0.651816</td>\n",
       "      <td>-0.073403</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 30 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       Time        V1        V2        V3        V4        V5        V6  \\\n",
       "0 -1.996583 -0.694242 -0.044075  1.672773  0.973366 -0.245117  0.347068   \n",
       "1 -1.996583  0.608496  0.161176  0.109797  0.316523  0.043483 -0.061820   \n",
       "2 -1.996562 -0.693500 -0.811578  1.169468  0.268231 -0.364572  1.351454   \n",
       "3 -1.996562 -0.493325 -0.112169  1.182516 -0.609727 -0.007469  0.936150   \n",
       "4 -1.996541 -0.591330  0.531541  1.021412  0.284655 -0.295015  0.071999   \n",
       "\n",
       "         V7        V8        V9  ...       V20       V21       V22       V23  \\\n",
       "0  0.193679  0.082637  0.331128  ...  0.326118 -0.024923  0.382854 -0.176911   \n",
       "1 -0.063700  0.071253 -0.232494  ... -0.089611 -0.307377 -0.880077  0.162201   \n",
       "2  0.639776  0.207373 -1.378675  ...  0.680975  0.337632  1.063358  1.456320   \n",
       "3  0.192071  0.316018 -1.262503  ... -0.269855 -0.147443  0.007267 -0.304777   \n",
       "4  0.479302 -0.226510  0.744326  ...  0.529939 -0.012839  1.100011 -0.220123   \n",
       "\n",
       "        V24       V25       V26       V27       V28    Amount  \n",
       "0  0.110507  0.246585 -0.392170  0.330892 -0.063781  0.244964  \n",
       "1 -0.561131  0.320694  0.261069 -0.022256  0.044608 -0.342475  \n",
       "2 -1.138092 -0.628537 -0.288447 -0.137137 -0.181021  1.160686  \n",
       "3 -1.941027  1.241904 -0.460217  0.155396  0.186189  0.140534  \n",
       "4  0.233250 -0.395202  1.041611  0.543620  0.651816 -0.073403  \n",
       "\n",
       "[5 rows x 30 columns]"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "# transform into a dataframe\n",
    "X_scaled = pd.DataFrame(X_scaled, index = X.index, columns = X.columns)\n",
    "X_scaled.head()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "hlm6gCsKMuHb"
   },
   "source": [
    "#### Train - Test Split\n",
    "\n",
    "As in this case, for extremely imbalanced datasets you may want to make sure that classes are balanced across train and test data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "id": "AuzpxEmKMuHb"
   },
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, stratify = y, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HO4HAIofMuHc"
   },
   "source": [
    "---\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MwQdl4PdJQ0I"
   },
   "source": [
    "## 3. Model Building\n",
    "It was previously stated that you need to make class prediction with three different algorithms. As in this case, different approaches are required to obtain better performance on unbalanced data.\n",
    "\n",
    "This dataset is severely **unbalanced** (most of the transactions are non-fraud). So the algorithms are much more likely to classify new observations to the majority class and high accuracy won't tell us anything. To address the problem of imbalanced dataset we can use undersampling and oversampling data approach techniques. Oversampling increases the number of minority class members in the training set. The advantage of oversampling is that no information from the original training set is lost unlike in undersampling, as all observations from the minority and majority classes are kept. On the other hand, it is prone to overfitting. \n",
    "\n",
    "There is a type of oversampling called **[SMOTE](https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/)** (Synthetic Minority Oversampling Technique), which we are going to use to make our dataset balanced. It creates synthetic points from the minority class.\n",
    "\n",
    "- It is important that you can evaluate the effectiveness of SMOTE. For this reason, implement the Logistic Regression algorithm in two different ways, with SMOTE applied and without.\n",
    "\n",
    "***Note***: \n",
    "\n",
    "- *Do not forget to import the necessary libraries and modules before starting the model building!*\n",
    "\n",
    "- *If you are going to use the cross validation method to be more sure of the performance of your model for unbalanced data, you should make sure that the class distributions in the iterations are equal. For this case, you should use **[StratifiedKFold](https://www.analyseup.com/python-machine-learning/stratified-kfold.html)** instead of regular cross validation method.*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zKZcwgucJQ0I"
   },
   "source": [
    "### Logistic Regression without SMOTE\n",
    "\n",
    "- The steps you are going to cover for this algorithm are as follows: \n",
    "\n",
    "   i. Import Libraries\n",
    "   \n",
    "   *ii. Model Training*\n",
    "   \n",
    "   *iii. Prediction and Model Evaluating*\n",
    "   \n",
    "   *iv. Plot Precision and Recall Curve*\n",
    "   \n",
    "   *v. Apply and Plot StratifiedKFold*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "o48s5BCdMuHd"
   },
   "source": [
    "***i. Import Libraries***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "id": "3G3cx-UjMuHd"
   },
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6KD76bc5MuHd"
   },
   "source": [
    "***ii. Model Training***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "id": "g7GAK-u3MuHd"
   },
   "outputs": [],
   "source": [
    "log_model = LogisticRegression(C = 10).fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "uvKAJVTNMuHd"
   },
   "source": [
    "***iii. Prediction and Model Evaluating***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "id": "Kb68hH1TMuHd"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00     42648\n",
      "           1       0.65      0.70      0.68        74\n",
      "\n",
      "    accuracy                           1.00     42722\n",
      "   macro avg       0.82      0.85      0.84     42722\n",
      "weighted avg       1.00      1.00      1.00     42722\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred = log_model.predict(X_test)\n",
    "confusion_matrix(y_test, y_pred)\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " accuracy score for log_model : 0.9989425195062047\n",
      "\n",
      " precision-0 score for log_model : 0.9994827720745094\n",
      "\n",
      " recall-0 score for log_model : 0.9994579340592884\n",
      "\n",
      " f1-0 score for log_model : 0.9994703369723181\n",
      "\n",
      " precision-1 score for log_model : 0.696111615109092\n",
      "\n",
      " recall-1 score for log_model : 0.7004065040650407\n",
      "\n",
      " f1-1 score for log_model : 0.6950815050428355\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import make_scorer\n",
    "from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score\n",
    "custom_scorer = {'accuracy': make_scorer(accuracy_score),\n",
    "                 'precision-0': make_scorer(precision_score, average='weighted', labels=[0]),\n",
    "                 'recall-0': make_scorer(recall_score, average='weighted', labels = [0]),\n",
    "                 'f1-0': make_scorer(f1_score, average='weighted', labels = [0]),\n",
    "                 'precision-1': make_scorer(precision_score, average='weighted', labels=[1]),\n",
    "                 'recall-1': make_scorer(recall_score, average='weighted', labels = [1]),\n",
    "                 'f1-1': make_scorer(f1_score, average='weighted', labels = [1])\n",
    "                 }\n",
    "\n",
    "for i, j in custom_scorer.items():\n",
    "    scores = cross_val_score(log_model, X_train, y_train, cv = 10, scoring = j).mean()\n",
    "    print(f\" {i} score for log_model : {scores}\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " accuracy score for log_model : 0.9988062236006771\n",
      "\n",
      " precision-0 score for log_model : 0.9992734226793442\n",
      "\n",
      " recall-0 score for log_model : 0.9995310393292408\n",
      "\n",
      " f1-0 score for log_model : 0.9994021692481688\n",
      "\n",
      " precision-1 score for log_model : 0.6751190476190476\n",
      "\n",
      " recall-1 score for log_model : 0.5767857142857142\n",
      "\n",
      " f1-1 score for log_model : 0.6085317460317461\n",
      "\n"
     ]
    }
   ],
   "source": [
    "custom_scorer = {'accuracy': make_scorer(accuracy_score),\n",
    "                 'precision-0': make_scorer(precision_score, average='weighted', labels=[0]),\n",
    "                 'recall-0': make_scorer(recall_score, average='weighted', labels = [0]),\n",
    "                 'f1-0': make_scorer(f1_score, average='weighted', labels = [0]),\n",
    "                 'precision-1': make_scorer(precision_score, average='weighted', labels=[1]),\n",
    "                 'recall-1': make_scorer(recall_score, average='weighted', labels = [1]),\n",
    "                 'f1-1': make_scorer(f1_score, average='weighted', labels = [1])\n",
    "                 }\n",
    "\n",
    "for i, j in custom_scorer.items():\n",
    "    scores = cross_val_score(log_model, X_test, y_test, cv = 10, scoring = j).mean()\n",
    "    print(f\" {i} score for log_model : {scores}\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjQAAAGACAYAAAC6OPj9AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deVhUdf//8dewo5hGrqm5lEuIppaae4KVG7jcuZSaaJm3SZtpaLcLirtouWSZdluiiOaeS2ju6V1qaUiuaaImUkkqIgIy5/dHV/PVFEELhs+v5+O67utyzpw55z0wDc/7zDlgsyzLEgAAgMFcnD0AAADAX0XQAAAA4xE0AADAeAQNAAAwHkEDAACMR9AAAADjETRANs6cOaM6der8bdvbtGmTxowZc9t1tm7dqmnTpuV6/RkzZujxxx9X+/bt1b59ewUHBysgIEDjx49XQf2NDH379tUPP/zwt20vKSlJQ4YMUVBQkIKDg9W5c2d98cUXf2mbhw4dUsuWLdWpUyedOXPmjh8/bdo0rVy58i/N8Ievv/5a1apVU1hY2E339ezZM1ev0etfV3+Wm9cZYAI3Zw8A/FMEBgYqMDDwtuscOHBAFy9ezPX6ktSmTRuNGDHCcfvixYsKDg5WkyZN1LRp0782dB6YM2fO37at5ORkdevWTa+99prGjx8vm82mw4cPq3fv3vL29lbjxo3varubNm1SgwYNNHbs2Lt6/GuvvXZXj8tOiRIltGXLFqWlpcnb21uS9NNPP+nHH3/M1eOvf139WW5fZ0BBR9AAdyElJUWjRo3S4cOHZbPZ1LRpUw0cOFBubm7atm2bIiMj5eLioocffli7du1SdHS0du/erdjYWM2ePVsbNmzQ+++/L5vNJldXV7311lvy8PBQTEyMsrKyVKRIEVWoUMGx/i+//KKRI0fqxIkTcnFxUbdu3fT888/fcrZff/1VV69eVdGiRSVJx48f19ixY3XhwgVlZWWpZ8+eeuaZZyRJH374oZYuXarChQvrscce06ZNm7R582YNGTJEFy5c0OnTp/XEE0/otddeU2RkpPbs2aOsrCz5+flp2LBh8vHxUXR0tGJiYuTu7i5PT0+NHj1aDz30ULbLAwICNG3aNNWsWVOLFy9WVFSUXFxcVLx4cQ0fPlyVKlXSkCFD5OPjoyNHjujcuXOqVq2aJk6cqMKFC9/wXKOjo1W3bl116NDBsax69eqaPn267rnnHknS3r17NWnSJKWlpcnd3V2vv/66mjVrpuXLl2vjxo1ycXFRQkKCvLy8NHHiRH3//fdatGiRsrKydPXqVTVu3NjxfZCk5cuXO27v3btXEyZMkN1ulyT169dPTz/9tIYMGaIqVarohRdeuOP9P/jggzd9T4sVK6by5cvriy++UFBQkCRp5cqVCgoKUkxMjCTpypUrCg8PV0JCgi5cuKDChQsrMjJSKSkpN72uli5dqrS0NPn4+Khjx46KjY3VtGnT9K9//UvPPfecunfvrk8//VTz58/XkiVLHBEFFGgWgFs6ffq0Vbt27Vve99Zbb1kRERGW3W630tPTrT59+lizZ8+2kpOTrfr161uHDh2yLMuyli9fblWtWtU6ffq0tWzZMuull16yLMuyAgMDrX379lmWZVk7duywZsyYYVmWZU2fPt0aNWqUZVnWDesPGDDAmjhxomVZlnXp0iWrbdu21smTJ63p06dbDRo0sIKDg60nn3zSql+/vhUSEmKtX7/esizLyszMtNq0aWPFx8c7Htu6dWtr37591vbt262nn37aunjxomW3262hQ4daLVq0sCzLssLCwqxevXo5nu+MGTOsCRMmWHa73bIsy5oyZYo1cuRI69q1a1aNGjWspKQky7Isa8WKFVZMTEy2yy3Lslq0aGHFxcVZu3btslq2bGmdP3/e8Xxbt25t2e12KywszOratauVnp5uZWRkWB06dLCWLl160/ehX79+1oIFC7L9HiYnJ1sNGza09u/fb1mWZR09etSqX7++derUKWvZsmXWo48+aiUmJlqWZVmjR4+23nrrrdt+H/58+/nnn7fWrFljWZZlHTp0yAoPD3d8/ebOnXvX+7/eV199ZbVt29b6/PPPrRdeeMGxvG3btlZ8fLzjNbp+/XorIiLCcf/w4cOt0aNH3/L51KtXz0pJSbnp+Rw+fNiqX7++tXXrVqtRo0bW8ePHs/3aAgUNR2iAu7B9+3YtWrRINptNHh4e6tatmz755BNVqlRJDz74oKpXry5J6tix4y3PT2jbtq1CQ0PVvHlzNW7cWH379r3t/nbt2qXBgwdLkooUKaI1a9Y47vvjI6eMjAxFRETohx9+UEBAgCTp5MmTOnXqlN5++23H+levXtXBgwd14sQJ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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1efe6cbc0a0>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from yellowbrick.classifier import ConfusionMatrix\n",
    "# The ConfusionMatrix visualizer taxes a model\n",
    "cm = ConfusionMatrix(log_model, classes=y_test.unique())\n",
    "\n",
    "# Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model\n",
    "cm.fit(X_train, y_train)\n",
    "\n",
    "# To create the ConfusionMatrix, we need some test data. Score runs predict() on the data\n",
    "# and then creates the confusion_matrix from scikit-learn.\n",
    "cm.score(X_test, y_test)\n",
    "\n",
    "# How did we do?\n",
    "cm.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "l193OP5fMuHd"
   },
   "source": [
    "\n",
    "You're evaluating \"accuracy score\"? Is your performance metric reflect real success? You may need to use different metrics to evaluate performance on unbalanced data. You should use **[precision and recall metrics](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#:~:text=The%20precision%2Drecall%20curve%20shows,a%20low%20false%20negative%20rate.)**."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fUDt5voIMuHe"
   },
   "source": [
    "***iv. Plot Precision and Recall Curve***\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "id": "WI0OI9SDMuHe"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average precision-recall score: 0.59\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import average_precision_score\n",
    "y_score = log_model.decision_function(X_test)\n",
    "average_precision = average_precision_score(y_test, y_score, pos_label = 1)\n",
    "\n",
    "print('Average precision-recall score: {0:0.2f}'.format(average_precision))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, '2-class Precision-Recall curve: AP=0.59')"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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w9GjRwEAMTExaNGihTivYcOG0Ol0SEhIAABERUWhefPm5iqFiIioUjHblnffvn0RGRmJwMBACIKA+fPnY/fu3cjKykJAQADmzZuHkJAQCIIAT09P9OjRw1ylEBERVSpmC2+5XI45c+aYtLm5uYmvO3fujPDwcHOtnoiIqNLiTVqIiIgkhuFNREQkMSXabX779m2sX78e6enpEARBbF+wYIHZCiMiIqLClSi8P/jgA7Rv3x7t27eHTCYzd01ERERUjBKFd15eHqZMmWLuWoiIiKgESnTM28vLC4cOHYJerzd3PURERPQUJdry3rdvH2i6fAUAACAASURBVNavX2/SJpPJcP78ebMURUREREUrUXgfO3bM3HUQERFRCZUovLOzs7Fy5UocP34cBoMBnTp1wvvvvw87Oztz10dERET/UKJj3nPmzEF2djbmz5+PRYsWITc3Fx9//LG5ayMiIqJClGjL+6+//sKuXbvE6VmzZsHX19dsRREREVHRSrTlLQgCMjIyxOmMjAwoFAqzFUVERERFK9GW9+jRo/H666+jV69eEAQBhw8fxrhx48xdGxERERWiROHt7++PNm3a4NSpUzAajVixYgVatmxp7tqIiIioEMXuNj98+DAAYMeOHTh37hzs7e3h4OCA8+fPY8eOHeVSIBEREZkqdsv77Nmz6NmzJ06ePFno/FdffdUsRREREVHRig3viRMnAjB9elhmZiYSExPRvHlz81ZGREREhSrR2eZbtmzB1KlTkZKSAj8/P0ycOBFffvmluWsjIiKiQpQovDds2ID//ve/2LNnD3r37o3du3dj//795q6NiIiIClGi8AaA2rVr48iRI+jRoweUSiVycnLMWRcREREVoUTh3axZM/z73//GrVu30LlzZ3zwwQdo06aNuWsjIiKiQpToOu/58+cjOjoazZs3h0qlwqBBg9C9e3dz10ZERESFKDa8N23ahICAAPHktCcvGTt37hwmTJhg3uqIiIiogGJ3mwuCUF51EBERUQkVu+UdGBgIABg/fjyOHDmC3r17IyUlBYcOHYK/v3+5FEhERESmSnTC2syZM00uDTt58iSf501ERGQhJTphLS4uDrt37wYAODs7Y/HixRg4cKBZCyMiIqLClWjL22g04t69e+L0gwcPIJeX+BJxIiIiKkMl2vIeP348hgwZAi8vLwBAbGwsZsyYYdbCiIiIqHAlCu+BAweiY8eOiImJgVKpxEcffYTatWubuzYiIiIqRIn2fev1emzfvh0HDx5Ex44dsXnzZuj1enPXRkRERIUoUXjPmTMHWVlZOHfuHJRKJW7cuIHp06ebuzYiIiIqRInC+6+//sJ///tfKJVK2NraYtGiRbhw4YK5ayMiIqJClCi8ZTIZ9Ho9ZDIZACA1NVV8TUREROWrRCesjRw5EmPGjEFycjLmzZuHiIgIvPfee+aujYiIiApRovDu1q0b3N3dcfLkSRgMBqxevRqtWrUyd21ERERUiBKF94gRI7B37140a9bM3PUQERHRU5QovFu1aoUdO3agbdu2sLGxEdvr169vtsKIiIiocCUK79jYWJw5c8bkEaEymQwHDx40W2FERERUuGLDOykpCWFhYbC3t4enpycmTZoER0fH8qqNiIiIClHspWLTp09H7dq1ERISgtzcXCxYsKC86iIiIqIiPHXL+7vvvgMAeHt749VXXy2XooiIiKhoxW55W1lZmbx+cpqIiIgs45keys27qhEREVlesbvNL1++jN69e4vTSUlJ6N27NwRB4NnmREREFlJseP/666/lVQcRERGVULHh3aBBg/Kqg4iIiEromY55Pwuj0YhZs2YhICAAwcHBSEhIKHS5mTNnYsmSJeYqg4iIqNIxW3hHRERAr9dj06ZNCAkJwcKFCwsss3HjRly6dMlcJRAREVVKZgtvjUaDrl27AgA8PDwQFxdnMj86OhqxsbEICAgwVwlERESVUonubf48tFot1Gq1OK1QKJCXlwelUol79+5h5cqVWLlyJfbu3VviPv/5BaC0NBpNmfZXVXEcS49jWHqVZQwvJekAAHfv3oFGk1eu664sY2hJ5TWGZgtvtVoNnU4nThuNRiiVj1a3b98+pKamYty4cUhOTsbDhw/RtGlTvPbaa8X26e7uDmtr6zKpT6PRwMvLq0z6qso4jqXHMSy9yjSG2vgk4GAC6tWrDy+vl8ptvZVpDC2lrMcwJyenyI1Ws4V3u3btcPjwYfj6+iImJgYtWrQQ540cORIjR44EAGzbtg1Xr159anATERHRI2YL7759+yIyMhKBgYEQBAHz58/H7t27kZWVxePcREREpWC28JbL5ZgzZ45Jm5ubW4HluMVNRET0bMx2tjkRERGZB8ObiIhIYhjeREREEsPwJiIikhiG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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import plot_precision_recall_curve\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "disp = plot_precision_recall_curve(log_model, X_test, y_test)\n",
    "disp.ax_.set_title('2-class Precision-Recall curve: '\n",
    "                   'AP={0:0.2f}'.format(average_precision))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CAzArHfTMuHe"
   },
   "source": [
    "***v. Apply and Plot StratifiedKFold***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import StratifiedKFold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "id": "8ugUuOhhMuHe"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Class Ratio: 0.001727485630620034\n"
     ]
    }
   ],
   "source": [
    "print('Class Ratio:',sum(df['Class'])/len(df['Class']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "skf = StratifiedKFold(n_splits= 10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0         0\n",
       "1         0\n",
       "2         0\n",
       "3         0\n",
       "4         0\n",
       "         ..\n",
       "284802    0\n",
       "284803    0\n",
       "284804    0\n",
       "284805    0\n",
       "284806    0\n",
       "Name: Class, Length: 284807, dtype: int64"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "target = df.loc[:,'Class']\n",
    "target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fold 1 Class Ratio: 0.0017204452090867595\n",
      "Fold 2 Class Ratio: 0.0017204452090867595\n",
      "Fold 3 Class Ratio: 0.0017204452090867595\n",
      "Fold 4 Class Ratio: 0.0017204452090867595\n",
      "Fold 5 Class Ratio: 0.0017204452090867595\n",
      "Fold 6 Class Ratio: 0.0017555563358028158\n",
      "Fold 7 Class Ratio: 0.0017555563358028158\n",
      "Fold 8 Class Ratio: 0.001720505617977528\n",
      "Fold 9 Class Ratio: 0.001720505617977528\n",
      "Fold 10 Class Ratio: 0.001720505617977528\n"
     ]
    }
   ],
   "source": [
    "fold_no = 1\n",
    "for train_index, test_index in skf.split(df, df.Class):\n",
    "    train = df.loc[train_index,:]\n",
    "    test = df.loc[test_index,:]\n",
    "    print('Fold',str(fold_no),'Class Ratio:',sum(test['Class'])/len(test['Class']))\n",
    "    fold_no += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "skf = StratifiedKFold(n_splits= 10)\n",
    "def train_model(train, test, fold_no):\n",
    "    X = df.drop(\"Class\", axis= 1).columns\n",
    "    y = ['Class']\n",
    "    X_train = train[X]\n",
    "    y_train = train[y]\n",
    "    X_test = test[X]\n",
    "    y_test = test[y]\n",
    "    log_model.fit(X_train,y_train)\n",
    "    predictions = log_model.predict(X_test)\n",
    "    return f1_score(y_test, predictions, pos_label=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'f1_score' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-38-f76281f19eaf>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      4\u001b[0m     \u001b[0mtrain\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mdf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mtrain_index\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m     \u001b[0mtest\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mdf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mtest_index\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 6\u001b[1;33m     \u001b[0mscore\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtrain_model\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtrain\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mtest\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mfold_no\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      7\u001b[0m     \u001b[0mfold_no\u001b[0m \u001b[1;33m+=\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-37-611aec426f75>\u001b[0m in \u001b[0;36mtrain_model\u001b[1;34m(train, test, fold_no)\u001b[0m\n\u001b[0;32m      9\u001b[0m     \u001b[0mlog_model\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my_train\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     10\u001b[0m     \u001b[0mpredictions\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlog_model\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX_test\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 11\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0mf1_score\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpredictions\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpos_label\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'f1_score' is not defined"
     ]
    }
   ],
   "source": [
    "fold_no = 1\n",
    "score = []\n",
    "for train_index, test_index in skf.split(df, df.Class):\n",
    "    train = df.loc[train_index,:]\n",
    "    test = df.loc[test_index,:]\n",
    "    score.append(train_model(train,test,fold_no))\n",
    "    fold_no += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.6401514045143794 0.2524924943600838\n"
     ]
    }
   ],
   "source": [
    "print(pd.Series(score).mean(), pd.Series(score).std())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy : 99.89% (0.03%)\n",
      "precision-0 : 99.95% (0.02%)\n",
      "recall-0 : 99.95% (0.02%)\n",
      "f1-0 : 99.95% (0.01%)\n",
      "precision-1 : 69.61% (9.48%)\n",
      "recall-1 : 70.04% (10.26%)\n",
      "f1-1 : 69.51% (8.38%)\n"
     ]
    }
   ],
   "source": [
    "kfold = StratifiedKFold(n_splits= 10)\n",
    "custom_scorer = {'accuracy': make_scorer(accuracy_score),\n",
    "                 'precision-0': make_scorer(precision_score, average='weighted', labels=[0]),\n",
    "                 'recall-0': make_scorer(recall_score, average='weighted', labels = [0]),\n",
    "                 'f1-0': make_scorer(f1_score, average='weighted', labels = [0]),\n",
    "                 'precision-1': make_scorer(precision_score, average='weighted', labels=[1]),\n",
    "                 'recall-1': make_scorer(recall_score, average='weighted', labels = [1]),\n",
    "                 'f1-1': make_scorer(f1_score, average='weighted', labels = [1])\n",
    "                 }\n",
    "for i, j in custom_scorer.items():\n",
    "    results = cross_val_score(log_model, X_train, y_train, cv=kfold, n_jobs=-1, scoring = j)\n",
    "    print(i, \": %.2f%% (%.2f%%)\" % (results.mean()*100, results.std()*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy : 99.88% (0.04%)\n",
      "precision-0 : 99.93% (0.03%)\n",
      "recall-0 : 99.95% (0.03%)\n",
      "f1-0 : 99.94% (0.02%)\n",
      "precision-1 : 67.51% (15.10%)\n",
      "recall-1 : 57.68% (19.95%)\n",
      "f1-1 : 60.85% (17.65%)\n"
     ]
    }
   ],
   "source": [
    "kfold = StratifiedKFold(n_splits= 10)\n",
    "custom_scorer = {'accuracy': make_scorer(accuracy_score),\n",
    "                 'precision-0': make_scorer(precision_score, average='weighted', labels=[0]),\n",
    "                 'recall-0': make_scorer(recall_score, average='weighted', labels = [0]),\n",
    "                 'f1-0': make_scorer(f1_score, average='weighted', labels = [0]),\n",
    "                 'precision-1': make_scorer(precision_score, average='weighted', labels=[1]),\n",
    "                 'recall-1': make_scorer(recall_score, average='weighted', labels = [1]),\n",
    "                 'f1-1': make_scorer(f1_score, average='weighted', labels = [1])\n",
    "                 }\n",
    "for i, j in custom_scorer.items():\n",
    "    results = cross_val_score(log_model, X_test, y_test, cv=kfold, n_jobs=-1, scoring = j)\n",
    "    print(i, \": %.2f%% (%.2f%%)\" % (results.mean()*100, results.std()*100))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "bwELs8xsJQ0Q"
   },
   "source": [
    "- Didn't the performance of the model you implemented above satisfy you? If your model is biased towards the majority class and minority class recall is not sufficient, apply **SMOTE**."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4f8q5y12MuHe"
   },
   "source": [
    "### Apply SMOTE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "id": "rlz070TfMuHf"
   },
   "outputs": [],
   "source": [
    "from imblearn.over_sampling import SMOTE\n",
    "from imblearn.under_sampling import RandomUnderSampler\n",
    "from imblearn.pipeline import Pipeline\n",
    "os = SMOTE(random_state=42)\n",
    "os_data_X, os_data_y=os.fit_sample(X_train, y_train)\n",
    "os_data_X = pd.DataFrame(data=os_data_X,columns= X.columns)\n",
    "os_data_y= pd.DataFrame(data=os_data_y,columns=[\"Class\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9wvBCEvpJQ0U"
   },
   "source": [
    "### Logistic Regression with SMOTE\n",
    "\n",
    "- The steps you are going to cover for this algorithm are as follows:\n",
    "   \n",
    "   *i. Train-Test Split (Again)*\n",
    "   \n",
    "   *ii. Model Training*\n",
    "   \n",
    "   *iii. Prediction and Model Evaluating*\n",
    "   \n",
    "   *iv. Plot Precision and Recall Curve*\n",
    "   \n",
    "   *v. Apply and Plot StratifiedKFold*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "IJqXQ_aHMuHf"
   },
   "source": [
    "***i. Train-Test Split (Again)***\n",
    "\n",
    "Use SMOTE applied data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(483334, 30)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "os_data_X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(483334, 1)"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "os_data_y.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "evc6DLPcMuHf"
   },
   "source": [
    "***ii. Model Training***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "id": "hz36IA3EMuHf"
   },
   "outputs": [],
   "source": [
    "log_model_smote = LogisticRegression(C = 10).fit(os_data_X, os_data_y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kqJHSV5FMuHf"
   },
   "source": [
    "***iii. Prediction and Model Evaluating***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "id": "J_lxSdHyMuHg"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[41896   752]\n",
      " [    9    65]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.98      0.99     42648\n",
      "           1       0.08      0.88      0.15        74\n",
      "\n",
      "    accuracy                           0.98     42722\n",
      "   macro avg       0.54      0.93      0.57     42722\n",
      "weighted avg       1.00      0.98      0.99     42722\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred = log_model_smote.predict(X_test)\n",
    "print(confusion_matrix(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " precision-1 score for log_model : 0.696111615109092\n",
      "\n",
      " recall-1 score for log_model : 0.7004065040650407\n",
      "\n",
      " f1-1 score for log_model : 0.6950815050428355\n",
      "\n"
     ]
    }
   ],
   "source": [
    "custom_scorer = {\n",
    "                 'precision-1': make_scorer(precision_score, average='weighted', labels=[1]),\n",
    "                 'recall-1': make_scorer(recall_score, average='weighted', labels = [1]),\n",
    "                 'f1-1': make_scorer(f1_score, average='weighted', labels = [1])\n",
    "                 }\n",
    "\n",
    "for i, j in custom_scorer.items():\n",
    "    scores = cross_val_score(log_model_smote, X_train, y_train, cv = 10, scoring = j).mean()\n",
    "    print(f\" {i} score for log_model : {scores}\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " precision-1 score for log_model : 0.6751190476190476\n",
      "\n",
      " recall-1 score for log_model : 0.5767857142857142\n",
      "\n",
      " f1-1 score for log_model : 0.6085317460317461\n",
      "\n"
     ]
    }
   ],
   "source": [
    "custom_scorer = {\n",
    "                 'precision-1': make_scorer(precision_score, average='weighted', labels=[1]),\n",
    "                 'recall-1': make_scorer(recall_score, average='weighted', labels = [1]),\n",
    "                 'f1-1': make_scorer(f1_score, average='weighted', labels = [1])\n",
    "                 }\n",
    "\n",
    "for i, j in custom_scorer.items():\n",
    "    scores = cross_val_score(log_model_smote, X_test, y_test, cv = 10, scoring = j).mean()\n",
    "    print(f\" {i} score for log_model : {scores}\\n\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "AFGgLGBqMuHg"
   },
   "source": [
    "***iv.  Plot Precision and Recall Curve***\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "id": "CWdU7r-UMuHg"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average precision-recall score: 0.67\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import average_precision_score\n",
    "y_score = log_model_smote.decision_function(X_test)\n",
    "average_precision = average_precision_score(y_test, y_score, pos_label = 1)\n",
    "\n",
    "print('Average precision-recall score: {0:0.2f}'.format(average_precision))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, '2-class Precision-Recall curve: AP=0.67')"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import plot_precision_recall_curve\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "disp = plot_precision_recall_curve(log_model_smote, X_test, y_test)\n",
    "disp.ax_.set_title('2-class Precision-Recall curve: '\n",
    "                   'AP={0:0.2f}'.format(average_precision))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HELeR2YVMuHg"
   },
   "source": [
    "***v. Apply and Plot StratifiedKFold***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "id": "hcmB-zKsMuHg"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "precision-1 : 64.45% (21.89%)\n",
      "recall-1 : 61.79% (20.08%)\n",
      "f1-1 : 61.45% (11.71%)\n"
     ]
    }
   ],
   "source": [
    "kfold = StratifiedKFold(n_splits= 10, shuffle = True)\n",
    "custom_scorer = {\n",
    "                 'precision-1': make_scorer(precision_score, average='weighted', labels=[1]),\n",
    "                 'recall-1': make_scorer(recall_score, average='weighted', labels = [1]),\n",
    "                 'f1-1': make_scorer(f1_score, average='weighted', labels = [1])\n",
    "                 }\n",
    "for i, j in custom_scorer.items():\n",
    "    results = cross_val_score(log_model_smote, X_test, y_test, cv=kfold, n_jobs=-1, scoring = j)\n",
    "    print(i, \": %.2f%% (%.2f%%)\" % (results.mean()*100, results.std()*100))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_3zm70O7JQ0Z"
   },
   "source": [
    "### Random Forest Classifier with SMOTE\n",
    "\n",
    "- The steps you are going to cover for this algorithm are as follows:\n",
    "\n",
    "   *i. Model Training*\n",
    "   \n",
    "   *ii. Prediction and Model Evaluating*\n",
    "   \n",
    "   *iii. Plot Precision and Recall Curve*\n",
    "   \n",
    "   *iv. Apply and Plot StratifiedKFold*\n",
    "   "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sr5U80HbMuHg"
   },
   "source": [
    "***i. Model Training***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "id": "kuvRr7f3MuHh"
   },
   "outputs": [],
   "source": [
    "rf = RandomForestClassifier().fit(os_data_X, os_data_y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dJ9TJdpmMuHh"
   },
   "source": [
    "***ii. Prediction and Model Evaluating***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "id": "BaNd2jTRMuHh"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[42635    13]\n",
      " [   14    60]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00     42648\n",
      "           1       0.82      0.81      0.82        74\n",
      "\n",
      "    accuracy                           1.00     42722\n",
      "   macro avg       0.91      0.91      0.91     42722\n",
      "weighted avg       1.00      1.00      1.00     42722\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred = rf.predict(X_test)\n",
    "print(confusion_matrix(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " precision-1 score for rf_model : 0.9036507936507936 std: 0.1374497778136176\n",
      "\n",
      " recall-1 score for rf_model : 0.6982142857142858 std: 0.21405493525289823\n",
      "\n",
      " f1-1 score for rf_model : 0.7968082897494663 std: 0.12387189284055077\n",
      "\n"
     ]
    }
   ],
   "source": [
    "custom_scorer = {\n",
    "                 'precision-1': make_scorer(precision_score, average='weighted', labels=[1]),\n",
    "                 'recall-1': make_scorer(recall_score, average='weighted', labels = [1]),\n",
    "                 'f1-1': make_scorer(f1_score, average='weighted', labels = [1])\n",
    "                 }\n",
    "\n",
    "for i, j in custom_scorer.items():\n",
    "    scores = cross_val_score(rf, X_test, y_test, cv = 10, scoring = j)\n",
    "    print(f\" {i} score for rf_model : {scores.mean()} std: {scores.std()}\\n\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "8bdqEhrdMuHh"
   },
   "source": [
    "***iii. Plot Precision and Recall Curve***\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "id": "smne1OBWMuHh"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Average precision-recall score: 0.67\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import average_precision_score\n",
    "y_score = rf.predict(X_test)\n",
    "average_precision = average_precision_score(y_test, y_score, pos_label = 1)\n",
    "\n",
    "print('Average precision-recall score: {0:0.2f}'.format(average_precision))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, '2-class Precision-Recall curve: AP=0.67')"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import plot_precision_recall_curve\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "disp = plot_precision_recall_curve(rf, X_test, y_test)\n",
    "disp.ax_.set_title('2-class Precision-Recall curve: '\n",
    "                   'AP={0:0.2f}'.format(average_precision))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1n8q3JXcMuHh"
   },
   "source": [
    "***iv. Apply and Plot StratifiedKFold***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "id": "WukW9Gb3MuHh"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "precision-1 : 91.14% (14.25%)\n",
      "recall-1 : 71.07% (19.54%)\n",
      "f1-1 : 80.09% (16.69%)\n"
     ]
    }
   ],
   "source": [
    "kfold = StratifiedKFold(n_splits= 10)\n",
    "custom_scorer = {\n",
    "                 'precision-1': make_scorer(precision_score, average='weighted', labels=[1]),\n",
    "                 'recall-1': make_scorer(recall_score, average='weighted', labels = [1]),\n",
    "                 'f1-1': make_scorer(f1_score, average='weighted', labels = [1])\n",
    "                 }\n",
    "for i, j in custom_scorer.items():\n",
    "    results = cross_val_score(rf, X_test, y_test, cv=kfold, n_jobs=-1, scoring = j)\n",
    "    print(i, \": %.2f%% (%.2f%%)\" % (results.mean()*100, results.std()*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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bDo6Oujo6GDHjh3MmzePNWvWAHDBBRfwy1/+kp07d/KBD3yAf/iHf5hSPole2W4ASka1lcTabwc+P8aYbwHfdvclwKvA1XHNUERERESOWW1tbQSDQbKyskhNTaWkpIRwODwiZtGiRRQUFHD88cePe58nnniCs846izPPPBOACy+8kJSUwc8WKSwspKenZ0r5JLrYfghYbWZzAcxsMZAB/MzdnwBeGx5sZgZ8LDYO4D7gU4lKVkRERESOLdFolMzMzKHrQCBANBo97Ps0Njaydu3aMfvuvfde/vZv/3ZK90lose3u+4A24KJYUwmwwd19nCHvB/rc/e3YdQ9w+liBZnatmW03s+37DhyYzrRFRERE5BgxVlk5uH47df39/WzevJnPfvazh/StW7eOlJQULr/88indKxlvkBy+leTgFpLxjPXKjFmYu3udu+e7e/77588/yhRFRERE5FgUCATo7u4euu7p6SEjI+Ow7tHc3MzKlSs59dRTR7Tfd999/PjHP+aBBx6YcgGfjGL7EWCVma0ETnD39gliXwFONLODh+8EgD1xzk9EREREjlEFBQVEIhG6urro7++nsbGRUCh0WPdoaGg4ZAtJS0sL3/rWt9i8eTPz5s2b8r0SfoKkux8ws63AvUy8qo27u5k9BXwGaASuAMITjRERERGRd6+UlBRqamooKipiYGCAsrIycnJyqK2tBaC8vJyXX36Z/Px89u/fz5w5c7jzzjvp7OxkwYIFvPHGG2zZsoXvfe97I+77xS9+kTfffJMLLrgAGHyT5MF7TsTG3y4dP2a2BtgILHP3F2JtzwBLgfnAPuBqd3/MzLIYLLQXAv8BfM7d35zo/nlnnuGP/5+vTBQiIiIiIsewmXJcu5ntcPf88foTvrIN4O6bGLUf290/PE7sb4BzE5GXiIiIiMh00gmSIiIiIiJxkpSV7XhLOWXRjPmnBRERERF599LKtoiIiIhInKjYFhERERGJk1m5jeStvd3suft/JzsNERERmUBG5T8lOwWRuNPKtoiIiIhInKjYFhERkRmlpaWF7OxsgsEg1dXVh/S/8MILnHfeecydO5c77rhjqP3Pf/4z5557LitWrCAnJ4evf/3rQ3233HILubm55OXlceGFF7Jnjw6klsSYEcW2mW01s6JRbdeZ2T2x7xeYWdTMapKToYiIiCTCwMAAlZWVNDc309nZSUNDA52dnSNiFi5cyF133cVXvjLyALu5c+fy5JNP8txzz9HR0UFLSwutra0A3HDDDezcuZOOjg5Wr17NrbfemrA5ybvbjCi2GTy2vWRUWwl/Oc7974GfJjQjERERSbi2tjaCwSBZWVmkpqZSUlJCOBweEbNo0SIKCgo4/vjjR7SbGfPnzwfgrbfe4q233sJs8Ay9BQsWDMW9/vrrQ+0i8TZTiu2HgNVmNhfAzBYDGcDPzOyDwKnA48lLT0RERBIhGo2SmZk5dB0IBIhGo1MePzAwQF5eHosWLeKCCy7gQx/60FDfzTffTGZmJg888IBWtiVhZkSx7e77gDbgolhTCbCBwSPd/xG4YbJ7mNm1ZrbdzLbvO/CnuOUqIiIi8ePuh7Qdzir0cccdR0dHBz09PbS1tfHLX/5yqG/dunV0d3dz+eWXU1OjnamSGDOi2I4ZvpXk4BaS/wk0uXv3ZIPdvc7d8909//3zT4hjmiIiIhIvgUCA7u6//Ge/p6eHjIyMw77PiSeeyEc/+lFaWloO6bvssst4+OGHjypPkamaScX2I8AqM1sJnODu7cB5wBfNbDdwB1BqZoe+LVlERERmhYKCAiKRCF1dXfT399PY2EgoFJrS2N///vf09fUB8Kc//Ymf/OQnLF26FIBIJDIUt3nz5qF2kXibMYfauPsBM9sK3EvsjZHufvnBfjO7Esh395uSkqCIiIjEXUpKCjU1NRQVFTEwMEBZWRk5OTnU1tYCUF5ezssvv0x+fj779+9nzpw53HnnnXR2dtLb28sVV1zBwMAA77zzDpdccgmrV68G4KabbmLXrl3MmTOHM888c+h+IvFmY+2NShYzWwNsBJa5+wuj+q5ksNj+4mT3WXHGqd781csnCxMREZEk0gmSMhuY2Q53zx+vf8asbAO4+yYG3xQ5Vt+/Av+ayHxERERERI7GTNqzLSIiIiIyq8yole3pcvyiTP3TlIiIiIgknVa2RURERETiRMW2iIiIiEiczMptJH/e+2teuPuTyU5DRORdbWllONkpiIgknVa2RURERETiRMW2iIgkXEtLC9nZ2QSDQaqrDz0Y2N2pqqoiGAySm5tLe3v7UN8///M/s3z5cnJycrjzzjtHjPvOd75DdnY2OTk53HjjjfGehojIpBK6jSR2QuQ/uPtjw9quAz4AZAGFwM/cffWw/geAfOAtoA34H+7+VgLTFhGRaTQwMEBlZSVbtmwhEAhQUFBAKBTi7LPPHoppbm4mEokQiUTYtm0bFRUVbNu2jV/+8pd8//vfp62tjdTUVC666CI+/vGPs2TJEp566inC4TA7d+5k7ty57N27N4mzFBEZlOiV7QagZFRbSaz9duDzY4x5AFgKnAOcAHwhngmKiEh8tbW1EQwGycrKIjU1lZKSEsLhkfu7w+EwpaWlmBmFhYX09fXR29vLr371KwoLC5k3bx4pKSl85CMfYdOmTQB897vf5aabbmLu3LkALFq0KOFzExEZLdHF9kPAajObC2Bmi4EMBleznwBeGz3A3Zs8hsGV7UAC8xURkWkWjUbJzMwcug4EAkSj0SnFLF++nKeffpp9+/bxxhtv0NTURHd3NwAvvvgizzzzDB/60If4yEc+ws9//vPETEhEZAIJ3Ubi7vvMrA24CAgzuKq9IVZIT8jMjmdw5ftL8c1SRETiaaxf+WY2pZhly5bx1a9+lQsuuID58+ezYsUKUlIG/1P29ttv8+qrr9La2srPf/5zLrnkEn7zm98ccm8RkURKxhskh28lObiFZCruAZ5292fG6jSza81su5ltf/VA/zSkKSIi8RAIBIZWowF6enrIyMiYcszVV19Ne3s7Tz/9NAsXLmTJkiVDYy6++GLMjHPPPZc5c+bwyiuvJGBGIiLjS0ax/QiwysxWAie4e/sk8ZjZ14FTgP89Xoy717l7vrvnnzQ/ddqSFRGR6VVQUEAkEqGrq4v+/n4aGxsJhUIjYkKhEPX19bg7ra2tpKWlkZ6eDjD0xsff/va3bNy4kbVr1wLwqU99iieffBIY3FLS39/PySefnMCZiYgcKuGH2rj7gdinktzLFFa1zewLQBGwyt3fiXN6IiISZykpKdTU1FBUVMTAwABlZWXk5ORQW1sLQHl5OcXFxTQ1NREMBpk3bx7r168fGv/pT3+affv2cfzxx3P33Xdz0kknAVBWVkZZWRnLly8nNTWV++67T1tIRCTpbArbpaf/oWZrgI3AMnd/Idb2DIOfOjIf2Adc7e6PmdnbwH/ylzdPbnT3Wye6//IzTvSHvvqRuOUvIiKT0wmSIvJuYGY73D1/vP6kHNfu7psAG9X24XFiZ+WR8iIiIiIy++kESRERERGROJmVq8bvWRTUP1+KiIiISNJpZVtEREREJE5UbIuIiIiIxMms3Eby+u9/zbN1q5OdhojIjHfetT9OdgoiIrOaVrZFREREROJExbaIiIiISJyo2BYRkUO0tLSQnZ1NMBikurr6kH53p6qqimAwSG5uLu3t7UN9//zP/8zy5cvJycnhzjvvHGr/wx/+wAUXXMCSJUu44IILePXVVxMxFRGRpJoRxbaZbTWzolFt15nZejPbYWYdZva8mZUnK0cRkXeLgYEBKisraW5uprOzk4aGBjo7O0fENDc3E4lEiEQi1NXVUVFRAcAvf/lLvv/979PW1sZzzz3Hj3/8YyKRCADV1dWsWrWKSCTCqlWrxiziRURmmxlRbAMNQMmothLgX4H/6u55wIeAm8wsI7GpiYi8u7S1tREMBsnKyiI1NZWSkhLC4ZFnF4TDYUpLSzEzCgsL6evro7e3l1/96lcUFhYyb948UlJS+MhHPsKmTZuGxlxxxRUAXHHFFTzyyCOJnpqISMLNlGL7IWC1mc0FMLPFQAbwtLu/GYuZy8zJV0Rk1opGo2RmZg5dBwIBotHolGKWL1/O008/zb59+3jjjTdoamqiu7sbgN/97nekp6cDkJ6ezt69exMwGxGR5JoRH/3n7vvMrA24CAgzuKq9wd3dzDKBR4EgcIO77xnrHmZ2LXAtwKkLT0hM4iIis5C7H9JmZlOKWbZsGV/96le54IILmD9/PitWrCAlZUb8p0ZEJClm0krx8K0kJbFr3L3b3XMZLLavMLNTxxrs7nXunu/u+SfNT01IwiIis1EgEBhajQbo6ekhIyNjyjFXX3017e3tPP300yxcuJAlS5YAcOqpp9Lb2wtAb28vixYtivdURESSbiYV248Aq8xsJXCCu7cP74ytaD8PfDgJuYmIvGsUFBQQiUTo6uqiv7+fxsZGQqHQiJhQKER9fT3uTmtrK2lpaUNbRA5uD/ntb3/Lxo0bWbt27dCY++67D4D77ruPT37ykwmclYhIcsyYf9tz9wNmthW4l9iqtpkFgH3u/iczOwn4a+CfkpeliMjsl5KSQk1NDUVFRQwMDFBWVkZOTg61tbUAlJeXU1xcTFNTE8FgkHnz5rF+/fqh8Z/+9KfZt28fxx9/PHfffTcnnXQSADfddBOXXHIJP/jBDzjjjDN48MEHkzI/EZFEsrH23SWLma0BNgLL3P0FM7sA+EfAAQNq3L1usvssO/NEv/fmv4lvsiIis4COaxcROTpmtsPd88frnzEr2wDuvonBovrg9RYgN3kZiYiIiIgcuRlVbE+X954S1GqNiIiIiCTdTHqDpIiIiIjIrKJiW0REREQkTlRsi4iIiIjEyazcs73/lQiP/aA42WmIyLtc0dVNyU5BRESSTCvbIiIiIiJxomJbRCTBWlpayM7OJhgMUl1dfUi/u1NVVUUwGCQ3N5f29sEDdXft2kVeXt7Q14IFC7jzzjsBeO655zjvvPM455xz+MQnPsH+/fsTOSURERlHQottM9tqZkWj2q4zs3vMrMXM+szsx6P6nzGzjtjXHjN7JJE5i4hMp4GBASorK2lubqazs5OGhgY6OztHxDQ3NxOJRIhEItTV1VFRUQFAdnY2HR0ddHR0sGPHDubNm8eaNWsA+MIXvkB1dTW/+MUvWLNmDbfffnvC5yYiIodK9Mp2A1Ayqq0k1n478PnRA9z9w+6e5+55wLMMnjApInJMamtrIxgMkpWVRWpqKiUlJYTD4REx4XCY0tJSzIzCwkL6+vro7e0dEfPEE09w1llnceaZZwKDq97nn38+ABdccAEPP/xwYiYkIiITSnSx/RCw2szmApjZYiAD+Jm7PwG8Nt5AM3sf8DHgkfinKSISH9FolMzMzKHrQCBANBo97JjGxkbWrl07dL18+XI2b94MwIMPPkh3d3c80hcRkcOU0GLb3fcBbcBFsaYSYIO7+xSGrwGecPcxNyKa2bVmtt3Mtv/xtf7pSVhEZJqN9evOzA4rpr+/n82bN/PZz352qO3ee+/l7rvv5oMf/CCvvfYaqamp05i1iIgcqWR89N/BrSTh2J9lUxy3FviX8TrdvQ6oA/jA4rSpFO8iIgkXCARGrDr39PSQkZFxWDHNzc2sXLmSU089daht6dKlPP744wC8+OKLPProo/GagoiIHIZkfBrJI8AqM1sJnODu7ZMNMLP3A+cC+q+HiBzTCgoKiEQidHV10d/fT2NjI6FQaERMKBSivr4ed6e1tZW0tDTS09OH+hsaGkZsIQHYu3cvAO+88w7f/OY3KS8vj/9kRERkUgkvtt39ALAVuJfBVe6p+CzwY3f/c7zyEhFJhJSUFGpqaigqKmLZsmVccskl5OTkUFtbS21tLQDFxcVkZWURDAa55ppruOeee4bGv/HGG2zZsoWLL754xH0bGhr4wAc+wNKlS8nIyOCqq65K6LxERGRsNrXt0tP8ULM1DH6qyDJ3fyHW9gywFJgP7AOudvfHYn1bgWp3b5nK/T+wOM2/c8tfxyN1EZEp0wmSIiKzn5ntcPf88fqTcly7u28CbFTbhyeI/2i8cxIRERERmW46QVJEREREJE6SsrIdbwtOXqJ/vhURERGRpNPKtoiIiIhInKjYFhERERGJk1m5jeTVVyI8tP6iyQNF5Jj1maum9OFEIiIiSaWVbRERERGROFGxLSKzSktLC9nZ2QSDQaqrqw/pd3eqqqoIBoPk5ubS3v6XQ2z7+vr4zGc+w9KlS1m2bBnPPvssAB0dHRQWFpKXl0d+fj5tbW0Jm4+IiBzbElpsm9lWMysa1Xadmd1jZi1m1mdmPx7V/0Uz+7WZuZmdnMh8ReTYMjAwQGVlJc3NzXR2dtLQ0EBnZ+eImObmZiKRCJFIhLq6OioqKob6vvSlL3HRRRfxwgsv8Nxzz7Fs2TIAbrzxRr7+9a/T0dHBrbfeyo033pjQeYmIyLEr0SvbDUDJqLaSWPvtwOfHGPP/gP8O/Gd8UxORY11bWxvBYJCsrCxSU1MpKSkhHA6PiAmHw5SWlmJmFBYW0tfXR29vL/v37+fpp5/m6quvBiA1NZUTTzwRADNj//79APzxj38kIyMjofMSEZFjV6LfIPkQ8E0zm+vub5rZYiAD+Jm7u5l9dPQAd/8PGPyPnYjIRKLRKJmZmUPXgUCAbdu2TRoTjUZJSUnhlFNO4aqrruK5557jgx/8IP/8z//Me9/7Xu68806Kior4yle+wjvvvMO///u/J2xOIiJybEvoyra77wPagIMfFVICbHB3T2QeIjI7jfWrZPRf1MeLefvtt2lvb6eiooL/+I//4L3vfe/Qnu/vfve7fPvb36a7u5tvf/vbQ6vfIiIik0nGGySHbyU5uIXkqJnZtWa23cy27z/QPx23FJFjTCAQoLu7e+i6p6fnkC0f48UEAgECgQAf+tCHAPjMZz4z9ObJ++67j4svvhiAz372s3qDpIiITFkyiu1HgFVmthI4wd3bJ4mfEnevc/d8d89fMD91Om4pIseYgoICIpEIXV1d9Pf309jYSCgUGhETCoWor6/H3WltbSUtLY309HROO+00MjMz2bVrFwBPPPEEZ599NgAZGRn89Kc/BeDJJ59kyZIliZ2YiIgcsxJ+qI27HzCzrcC9TNOqtogIQEpKCjU1NRQVFTEwMEBZWRk5OTnU1tYCUF5eTnFxMU1NTQSDQebNm8f69euHxn/nO9/h8ssvp7+/n6ysrKG+73//+3zpS1/i7bff5j3veQ91dXVJmZ+IiBx7LBnbpc1sDbARWObuL8TangGWAvOBfcDV7v6YmVUBNwKnAXuBJnf/wkT3P2txmn/r6+fFcwoikmQ6QVJERGYCM9vh7vnj9SfluHZ33wTYqLYPjxN7F3BXIvISEREREZlOOkFSRERERCROkrKyHW8nnbxE/8QsIiIiIkmnlW0RERERkThRsS0iIiIiEiezchvJK/teZP19FyY7DTlKV13xeLJTEBERETkqWtkWEREREYkTFdsiIiIiInGiYluOOS0tLWRnZxMMBqmurj6k392pqqoiGAySm5tLe3s7AH/+858599xzWbFiBTk5OXz9618fGvPggw+Sk5PDnDlz2L59e8LmIiIiIrNbQottM9tqZkWj2q4zs3vMrMXM+szsx6P6zczWmdmLZvar2ImS8i41MDBAZWUlzc3NdHZ20tDQQGdn54iY5uZmIpEIkUiEuro6KioqAJg7dy5PPvkkzz33HB0dHbS0tNDa2grA8uXL2bhxI+eff37C5yQiIiKzV6LfINkAlACPDWsrAW4AUoF5wP8YNeZKIBNY6u7vmNmiBOQpM1RbWxvBYJCsrCwASkpKCIfDnH322UMx4XCY0tJSzIzCwkL6+vro7e0lPT2d+fPnA/DWW2/x1ltvYTZ4kOmyZcsSPxkRERGZ9RK9jeQhYLWZzQUws8VABvAzd38CeG2MMRXAre7+DoC7701QrjIDRaNRMjMzh64DgQDRaHTKMQMDA+Tl5bFo0SIuuOACPvShDyUmcREREXlXSmix7e77gDbgolhTCbDB3X2CYWcBl5rZdjNrNrMlYwWZ2bWxmO0HXntrehOXGWOsH5WDq9NTiTnuuOPo6Oigp6eHtrY2fvnLX8YnURERERGS8wbJg1tJiP3ZMEn8XODP7p4PfB+4d6wgd69z93x3z5//vuOnLVmZWQKBAN3d3UPXPT09ZGRkHHbMiSeeyEc/+lFaWlrim7CIiIi8qyWj2H4EWGVmK4ET3L19kvge4OHY95uA3DjmJjNcQUEBkUiErq4u+vv7aWxsJBQKjYgJhULU19fj7rS2tpKWlkZ6ejq///3v6evrA+BPf/oTP/nJT1i6dGkSZiEiIiLvFgk/QdLdD5jZVgZXqCdb1YbB4vxjsfiPAC/GLTmZ8VJSUqipqaGoqIiBgQHKysrIycmhtrYWgPLycoqLi2lqaiIYDDJv3jzWr18PQG9vL1dccQUDAwO88847XHLJJaxevRqATZs28b/+1//i97//PR//+MfJy8vjscceGzcPERERkamwibdLx+mhZmuAjcAyd38h1vYMsBSYD+wDrnb3x8zsROAB4AzgAFDu7s9NdP/F/2WBf/0bhXGcgSSCjmsXERGRmc7MdsS2O48p4SvbAO6+CbBRbR8eJ7YP+HgC0hIRERERmVZJKbbj7eT3f0CroiIiIiKSdDquXUREREQkTlRsi4iIiIjEiYptEREREZE4mZV7tvf+IcJdDxQlO40ZoepyfXydiIiISLJoZVtEREREJE5UbL+LtbS0kJ2dTTAYpLq6+pB+d6eqqopgMEhubi7t7X857LOsrIxFixaxfPnyEWM6OjooLCwkLy+P/Px82tra4j4PERERkZlqRhTbZrbVzIpGtV1nZveY2bfM7Jexr0uTleNsMzAwQGVlJc3NzXR2dtLQ0EBnZ+eImObmZiKRCJFIhLq6OioqKob6rrzySlpaWg6574033sjXv/51Ojo6uPXWW7nxxhvjPhcRERGRmWpGFNsMHtteMqqtBPgdsBLIAz4E3GBmCxKb2uzU1tZGMBgkKyuL1NRUSkpKCIfDI2LC4TClpaWYGYWFhfT19dHb2wvA+eefz8KFCw+5r5mxf/9+AP74xz+SkZER/8mIiIiIzFAz5Q2SDwHfNLO57v6mmS0GMoA3gJ+6+9vA22b2HHAR8KPkpTo7RKNRMjMzh64DgQDbtm2bNCYajZKenj7ufe+8806Kior4yle+wjvvvMO///u/T3/yIiIiIseIGbGy7e77gDYGC2kYXNXeADwH/K2ZzTOzk4H/BmSOdQ8zu9bMtpvZ9gP7+xOR9jHN3Q9pM7PDjhntu9/9Lt/+9rfp7u7m29/+NldfffXRJSoiIiJyDJsRxXbM8K0kJUCDuz8ONAH/Hut/Fnh7rMHuXufu+e6eP39BaiLyPaYFAgG6u7uHrnt6eg7Z8jGVmNHuu+8+Lr74YgA++9nP6g2SIiIi8q42k4rtR4BVZrYSOMHd2wHcfZ2757n7BYABkSTmOGsUFBQQiUTo6uqiv7+fxsZGQqHQiJhQKER9fT3uTmtrK2lpaRNuIQHIyMjgpz/9KQBPPvkkS5YsidscRERERGa6mbJnG3c/YGZbgXsZXMXGzI4DTnT3fWaWC+QCjycvy9kjJSWFmpoaioqKGBgYoKysjJycHGprawEoLy+nuLiYpqYmgsEg8+bNY/369UPj165dy9atW3nllVcIBAL83d/9HVdffTXf//73+dKXvsTbb7/Ne97zHurq6pI1RREREZGks7H25SaLma0BNgLL3P0FM3sPcPDDnfcD5e7eMdl9zshK86/8fWH8Ej2G6ARJERERkfgxsx3unj9e/4xZ2QZw900MbhU5eP1n4OzkZSQiIiIicuRm0p5tEREREZFZZUatbE+XRQuXaPuEiIiIiCSdVrZFREREROJExbaIiIiISJzMym0ke16N8I0fFSU7jYT6xiXaNiMiIiIy02hlW0REREQkTlRsvwu0tLSQnZ1NMBikurr6kH53p6qqimAwSG5uLu3t7UN9ZWVlLFq0iOXLl48Yc+mll5KXl0deXh6LFy8mLy8v3tMQEREROebMiGLbzLaaWdGotuvM7Fdm1jHs689m9qkkpXlMGhgYoLKykubmZjo7O2loaKCzs3NETHNzM5FIhEgkQl1dHRUVFUN9V155JS0tLYfcd8OGDXR0dNDR0cGnP/1pLr744rjPRURERORYMyOKbQaPZy8Z1VYCXOvuee6eB3wMeAMd135Y2traCAaDZGVlkZqaSklJCeFweERMOBymtLQUM6OwsJC+vj56e3sBOP/881m4cOG493d3fvSjH7F27dq4zkNERETkWDRTiu2HgNVmNhfAzBYDGcDPhsV8Bmh29zcSn96xKxqNkpmZOXQdCASIRqOHHTOeZ555hlNPPZUlS5ZMT8IiIiIis8iMKLbdfR/QBlwUayoBNri7DwsrYXAFfExmdq2ZbTez7W/s749fsseYkS/hIDM77JjxNDQ0aFVbREREZBwzotiOGb6VZERhbWbpwDnAuJ9v5+517p7v7vnzFqTGNdFjSSAQoLu7e+i6p6eHjIyMw44Zy9tvv83GjRu59NJLpy9hERERkVlkJhXbjwCrzGwlcIK7tw/ruwTY5O5vJSWzY1hBQQGRSISuri76+/tpbGwkFAqNiAmFQtTX1+PutLa2kpaWRnp6+qT3/slPfsLSpUsJBALxSl9ERETkmDZjim13PwBsBe7l0O0ia8dokylISUmhpqaGoqIili1bxiWXXEJOTg61tbXU1tYCUFxcTFZWFsFgkGuuuYZ77rlnaPzatWs577zz2LVrF4FAgB/84AdDfY2NjdpCIiIiIjIBG2u/brKY2RpgI7DM3V+ItS0G/h+Q6e7vTOU+GWel+bX/UBi3PGcinSApIiIiknhmtsPd88frn1HHtbv7JsBGte0GTk9KQiIiIiIiR2HGbCMREREREZltZtTK9nTJOGmJtlWIiIiISNJpZVtEREREJE5UbIuIiIiIxMms3Eayuy/CVZsumjxwhlu/piXZKYiIiIjIUdDKtoiIiIhInKjYPka1tLSQnZ1NMBikurr6kH53p6qqimAwSG5uLu3tfzmQs6ysjEWLFrF8+fIx733HHXdgZrzyyitxy19ERETk3SChxbaZbTWzolFt15lZk5k9a2bPm9lOM7t0WP8XzezXZuZmdnIi852pBgYGqKyspLm5mc7OThoaGujs7BwR09zcTCQSIRKJUFdXR0VFxVDflVdeSUvL2FtUuru72bJlC2eccUZc5yAiIiLybpDole0GoGRUWwnwLaDU3XOAi4A7zezEWP//A/478J+JSnKma2trIxgMkpWVRWpqKiUlJYTD4REx4XCY0tJSzIzCwkL6+vro7e0F4Pzzz2fhwoVj3vv666/ntttuw8zG7BcRERGRqUt0sf0QsNrM5sLQUewZwNPuHgFw9z3AXuCU2PV/xE6RlJhoNEpmZubQdSAQIBqNHnbMaJs3b+b0009nxYoV05uwiIiIyLtUQj+NxN33mVkbg6vXYQZXtTe4ux+MMbNzgVTgpUTmdiwZ9nINGb0SPZWY4d544w3WrVvH448/fvQJioiIiAiQnDdIDt9KUhK7BsDM0oEfAle5+zuHc1Mzu9bMtpvZ9j/v75+2ZGeiQCBAd3f30HVPTw8ZGRmHHTPcSy+9RFdXFytWrGDx4sX09PSwcuVKXn755emfgIiIiMi7RDKK7UeAVWa2EjjB3dsBzGwB8CjwNXdvPdybunudu+e7e/57FqROa8IzTUFBAZFIhK6uLvr7+2lsbCQUCo2ICYVC1NfX4+60traSlpZGenr6uPc855xz2Lt3L7t372b37t0EAgHa29s57bTT4j0dERERkVkr4cW2ux8AtgL3ElvVNrNUYBNQ7+4PJjqnY01KSgo1NTUUFRWxbNkyLrnkEnJycqitraW2thaA4uJisrKyCAaDXHPNNdxzzz1D49euXct5553Hrl27CAQC/OAHP0jWVERERERmNRtrb2/cH2q2BtgILHP3F8zsc8B64PlhYVe6e4eZVQE3Aqcx+MbJJnf/wkT3PzmY5p+4/bw4ZZ84OkFSREREZGYzsx3unj9ef1KOa3f3TYANu74fuH+c2LuAuxKUmoiIiIjItNEJkiIiIiIicZKUle14W3ziEm3BEBEREZGk08q2iIiIiEicqNgWEREREYkTFdsiIiIiInEyK/dsR/p+y9+GK5OdxhFr/uTdyU5BRERERKaBVrZFREREROJExfYxpqWlhezsbILBINXV1Yf0uztVVVUEg0Fyc3Npb28f6isrK2PRokUsX758xJhbbrmF3Nxc8vLyuPDCC9mzZ0/c5yEiIiLybjAjim0z22pmRaParjOzJjN71syeN7OdZnZpsnKcCQYGBqisrKS5uZnOzk4aGhro7OwcEdPc3EwkEiESiVBXV0dFRcVQ35VXXklLy6EfiXjDDTewc+dOOjo6WL16Nbfeemvc5yIiIiLybjAjim2gASgZ1VYCfAsodfcc4CLgTjM7McG5zRhtbW0Eg0GysrJITU2lpKSEcDg8IiYcDlNaWoqZUVhYSF9fH729vQCcf/75LFy48JD7LliwYOj7119/HTM7JEZEREREDt9MeYPkQ8A3zWyuu79pZouBDOBpd3cAd99jZnuBU4C+pGWaRNFolMzMzKHrQCDAtm3bJo2JRqOkp6dPeO+bb76Z+vp60tLSeOqpp6Y3cREREZF3qRmxsu3u+4A2BlevYXBVe8PBQhvAzM4FUoGXxrqHmV1rZtvNbHv//j/FO+WkGPZyDBm9Cj2VmLGsW7eO7u5uLr/8cmpqao48SREREREZMiOK7ZjhW0lKYtcAmFk68EPgKnd/Z6zB7l7n7vnunp+64IS4J5sMgUCA7u7uoeuenh4yMjIOO2Yil112GQ8//PDRJysiIiIiM6rYfgRYZWYrgRPcvR3AzBYAjwJfc/fWJOaXdAUFBUQiEbq6uujv76exsZFQKDQiJhQKUV9fj7vT2tpKWlrapFtIIpHI0PebN29m6dKlcclfRERE5N1mpuzZxt0PmNlW4F5iq9pmlgpsAurd/cEkpjcjpKSkUFNTQ1FREQMDA5SVlZGTk0NtbS0A5eXlFBcX09TURDAYZN68eaxfv35o/Nq1a9m6dSuvvPIKgUCAv/u7v+Pqq6/mpptuYteuXcyZM4czzzxz6H4iIiIicnRsrD2+yWJma4CNwDJ3f8HMPgesB54fFnalu3dMdJ+04CL/r//42fglGmc6QVJERETk2GBmO9w9f7z+GbOyDeDumwAbdn0/cH/yMhIREREROXIzac+2iIiIiMisMqNWtqfLkhPP0FYMEREREUk6rWyLiIiIiMSJim0RERERkTiZldtIIn29FG/6ZrLTmFTTmq8lOwURERERiSOtbIuIiIiIxImK7RmqpaWF7OxsgsEg1dXVh/S7O1VVVQSDQXJzc2lvbx/qKysrY9GiRSxfvnzEmAcffJCcnBzmzJnD9u3b4z4HERERkXe7GVFsm9lWMysa1Xadmd1jZi1m1mdmP05Wfok2MDBAZWUlzc3NdHZ20tDQQGdn54iY5uZmIpEIkUiEuro6KioqhvquvPJKWlpaDrnv8uXL2bhxI+eff37c5yAiIiIiM6TYZvB49pJRbSWx9tuBzyc8oyRqa2sjGAySlZVFamoqJSUlhMPhETHhcJjS0lLMjMLCQvr6+ujt7QXg/PPPZ+HChYfcd9myZWRnZydkDiIiIiIyc4rth4DVZjYXwMwWAxnAz9z9CeC1JOaWcNFolMzMzKHrQCBANBo97BgRERERSa4ZUWy7+z6gDbgo1lQCbHB3n+o9zOxaM9tuZtv7978ejzQTZqxpm9lhx4iIiIhIcs2IYjtm+FaSg1tIpszd69w9393zUxe8d9qTS6RAIEB3d/fQdU9PDxkZGYcdIyIiIiLJNZOK7UeAVWa2EjjB3dsniZ+1CgoKiEQidHV10d/fT2NjI6FQaERMKBSivr4ed6e1tZW0tDTS09OTlLGIiIiIjGXGFNvufgDYCtzLYa5qzzYpKSnU1NRQVFTEsmXLuOSSS8jJyaG2tpba2loAiouLycrKIhgMcs0113DPPfcMjV+7di3nnXceu3btIhAI8IMf/ACATZs2EQgEePbZZ/n4xz9OUVHRmM8XERERkelhh7EtOu7MbA2wEVjm7i/E2p4BlgLzgX3A1e7+2ET3SQue7n99e8VEITOCTpAUERERObaZ2Q53zx+vf0Yd1+7umwAb1fbhJKUjIiIiInJUZsw2EhERERGR2WZGrWxPlyUnpmuLhoiIiIgknVa2RURERETiRMW2iIiIiEiczMptJJG+vXx8413JTmPIoxdXJTsFEREREUkCrWyLiIiIiMSJiu0ka2lpITs7m2AwSHV19SH97k5VVRXBYJDc3Fza2/9ysGZZWRmLFi1i+fLlI8b84Q9/4IILLmDJkiVccMEFvPrqq3Gfh4iIiIgcakYU22a21cyKRrVdZ2b3mNkZZva4mf3KzDrNbHGS0px2AwMDVFZW0tzcTGdnJw0NDXR2do6IaW5uJhKJEIlEqKuro6LiL4f1XHnllbS0tBxy3+rqalatWkUkEmHVqlVjFvEiIiIiEn8zothm8Hj2klFtJbH2euB2d18GnAvsTXBucdPW1kYwGCQrK4vU1FRKSkoIh8MjYsLhMKWlpZgZhYWF9PX10dvbC8D555/PwoULD7lvOBzmiiuuAOCKK67gkUceiftcRERERORQM6XYfghYbWZzAWKr1xnAH4AUd98C4O4H3P2NpGU5zaLRKJmZmUPXgUCAaDR62DGj/e53vyM9PR2A9PR09u6dNX8/ERERETmmzIhi2933AW3ARbGmEmADsAToM7ONZvYfZna7mR2XrDynm7sf0mZmhx0jIiIiIjPTjCi2Y4ZvJTm4hSQF+DDwFaAAyAKuHGuwmV1rZtvNbHv/Hw/EP9tpEAgE6O7uHrru6ekhIyPjsGNGO/XUU4e2mvT29rJo0aJpzFpEREREpmomFduPAKvMbCVwgru3Az3Af7j7b9z97VjMyrEGu3udu+e7e35q2vxE5XxUCgoKiEQidHV10d/fT2NjI6FQaERMKBSivr4ed6e1tZW0tLShLSLjCYVC3HfffQDcd999fPKTn4zbHERERERkfDOm2Hb3A8BW4F4GV7UBfg6cZGanxK4/BnQeOvrYlJKSQk1NDUVFRSxbtoxLLrmEnJwcamtrqa2tBaC4uJisrCyCwSDXXHMN99xzz9D4tWvXct5557Fr1y4CgQA/+MEPALjpppvYsmULS5YsYcuWLdx0001JmZ+IiIjIu52NtSc4WcxsDbARWObuL8TaLgD+ETBgB3Ctu/dPdJ+04Bn+N7d9Jd7pTplOkBQRERGZncxsh7vnj9c/o45rd/dNDBbVw9u2ALnJyUhERERE5MjNmG0kIiIiIiKzzYxa2Z4uS05cpK0bIiIiIpJ0WtkWEREREYkTFdsiIiIiInGiYltEREREJE5m5Z7tyKuv8PGH/yXZafDop7+Q7BREREREJIm0si0iIiIiEicqtpOgpaWF7OxsgsEg1dXVh/S7O1VVVQSDQXJzc2lvb5907HPPPcd5553HOeecwyc+8Qn279+fkLmIiIiIyPhmRLFtZlvNrGhU23Vmdo+Z3WZmz5vZr8zsLjOz8e5zLBgYGKCyspLm5mY6OztpaGigs3PkCfTNzc1EIhEikQh1dXVUVFRMOvYLX/gC1dXV/OIXv2DNmjXcfvvtCZ+biIiIiIw0I4ptoAEoGdVWAmwA/prBEySXAwXARxKb2vRqa2sjGAySlZVFamoqJSUlhMPhETHhcJjS0lLMjMLCQvr6+ujt7Z1w7K5duzj//PMBuOCCC3j44YcTPjcRERERGWmmFNsPAavNbC6AmS0GMoB+4D1AKjAXOB74XZJynBbRaJTMzMyh60AgQDQanVLMRGOXL1/O5s2bAXjwwQfp7u6O5zREREREZApmRLHt7vuANuCiWFMJsMHdnwWeAnpjX4+5+6/GuoeZXWtm281se//+1xKR9hFx90PaRu+MGS9morH33nsvd999Nx/84Ad57bXXSE1NnaaMRURERORIzaSP/ju4lSQc+7PMzILAMiAQi9liZue7+9OjB7t7HVAHkHbW4kOr0hkiEAiMWHXu6ekhIyNjSjH9/f3jjl26dCmPP/44AC+++CKPPvpoPKchIiIiIlMwI1a2Yx4BVpnZSuAEd28H1gCt7n7A3Q8AzUBhEnM8agUFBUQiEbq6uujv76exsZFQKDQiJhQKUV9fj7vT2tpKWloa6enpE47du3cvAO+88w7f/OY3KS8vT/jcRERERGSkGVNsx4rprcC9DK5yA/wW+IiZpZjZ8Qy+OXLMbSTHipSUFGpqaigqKmLZsmVccskl5OTkUFtbS21tLQDFxcVkZWURDAa55ppruOeeeyYcC9DQ0MAHPvABli5dSkZGBldddVXS5igiIiIig2ysfcDJYmZrgI3AMnd/wcyOA+4BzgccaHH3/z3ZfdLOWux/c9vX4pvsFOgESREREZHZzcx2uHv+eP0zac827r4JsGHXA8D/SF5GIiIiIiJHbsZsIxERERERmW1m1Mr2dFly0snawiEiIiIiSaeVbRERERGROFGxLSIiIiISJ7NyG8mvX/0Dqx96IGHP+/FnLk/Ys0RERETk2KGVbRERERGROFGxHUctLS1kZ2cTDAaprq4+pN/dqaqqIhgMkpubS3t7+6RjL730UvLy8sjLy2Px4sXk5eUlYioiIiIicgQSuo3EzLYC/+Dujw1ruw64EDgJWAAMAOvcfUOs/xngfbHwRUCbu38qcVkfmYGBASorK9myZQuBQICCggJCoRBnn332UExzczORSIRIJMK2bduoqKhg27ZtE47dsGHD0Pgvf/nLpKWlJWN6IiIiIjIFid6z3QCUAI8NaysBvgrscfeImWUAO8zsMXfvc/cPHww0s4eBcEIzPkJtbW0Eg0GysrIAKCkpIRwOjyi2w+EwpaWlmBmFhYX09fXR29vL7t27Jx3r7vzoRz/iySefTOzERERERGTKEr2N5CFgtZnNBTCzxUAG8LS7RwDcfQ+wFzhl+EAzex/wMeCRBOZ7xKLRKJmZmUPXgUCAaDQ6pZipjH3mmWc49dRTWbJkSZxmICIiIiJHK6HFtrvvA9qAi2JNJcAGd/eDMWZ2LpAKvDRq+BrgCXffP9a9zexaM9tuZtv7948ZklDDpjTEzKYUM5WxDQ0NrF279iizFBEREZF4SsZH/x3cShKO/Vl2sMPM0oEfAle4+zujxq0F/mW8m7p7HVAHcOJZWYdWqwkWCATo7u4euu7p6SEjI2NKMf39/ROOffvtt9m4cSM7duyI4wxERERE5Ggl49NIHgFWmdlK4AR3bwcwswXAo8DX3L11+AAzez9wbqz/mFBQUEAkEqGrq4v+/n4aGxsJhUIjYkKhEPX19bg7ra2tpKWlkZ6ePunYn/zkJyxdupRAIJDoaYmIiIjIYUj4yra7H4h9Ksm9DK5yY2apwCag3t0fHGPYZ4Efu/ufE5boUUpJSaGmpoaioiIGBgYoKysjJyeH2tpaAMrLyykuLqapqYlgMMi8efNYv379hGMPamxs1BYSERERkWOAjbU/OO4PNVsDbASWufsLZvY5YD3w/LCwK929Ixa/Fah295ap3P/Es7L8b77199Ob9AR0gqSIiIjIu5OZ7XD3/PH6k3Jcu7tvAmzY9f3A/RPEfzQBaYmIiIiITCudICkiIiIiEidJWdmOt+BJC7W1Q0RERESSTivbIiIiIiJxomJbRERERCROZuU2kl+/2scnHtoY9+f8389cHPdniIiIiMixSyvbIiIiIiJxomI7TlpaWsjOziYYDFJdXX1Iv7tTVVVFMBgkNzeX9vb2Scdeeuml5OXlkZeXx+LFi8nLy0vEVERERETkCMVtG0nsiPUnYpenAQPA74EggydF/s94PTvZBgYGqKysZMuWLQQCAQoKCgiFQpx99tlDMc3NzUQiESKRCNu2baOiooJt27ZNOHbDhg1D47/85S+TlpaWjOmJiIiIyBTFrdh2931AHoCZfQM44O53xOt5M0lbWxvBYJCsrCwASkpKCIfDI4rtcDhMaWkpZkZhYSF9fX309vaye/fuSce6Oz/60Y948sknEzsxERERETksCd9GYmYfNbMfx77/hpndZ2aPm9luM7vYzG4zs1+YWYuZHR+L+6CZ/dTMdpjZY2aWnui8D0c0GiUzM3PoOhAIEI1GpxQzlbHPPPMMp556KkuWLInTDERERERkOsyEPdtnAR8HPsngke1Pufs5wJ+Aj8cK7u8An3H3DwL3AuuSlexUuPshbWY2pZipjG1oaGDt2rVHmaWIiIiIxNtM+Oi/Znd/y8x+ARwHtMTafwEsBrKB5cCWWNF5HNA7+iZmdi1wLcAJJ58c/6wnEAgE6O7uHrru6ekhIyNjSjH9/f0Tjn377bfZuHEjO3bsiOMMRERERGQ6zISV7TcB3P0d4C3/y9LuOwz+ZcCA5909L/Z1jrtfOPom7l7n7vnunp+6ILlvHCwoKCASidDV1UV/fz+NjY2EQqERMaFQiPr6etyd1tZW0tLSSE9Pn3TsT37yE5YuXUogEEj0tERERETkMM2Ele3J7AJOMbPz3P3Z2LaSD7j788lObDwpKSnU1NRQVFTEwMAAZWVl5OTkUFtbC0B5eTnFxcU0NTURDAaZN28e69evn3DsQY2NjdpCIiIiInKMmPHFtrv3m9lngLvMLI3BnO8EZmyxDVBcXExxcfGItvLy8qHvzYy77757ymMP+td//ddpy1FERERE4ishxba7f2PY91uBraPbY9fzxxnTAZwfzxxFRERERKbbTNizLSIiIiIyK834bSRHInjSifzfz1yc7DRERERE5F1OK9siIiIiInGiYltEREREJE5m5TaSX7+6n08+1DJ54BEIf+aiuNxXRERERGYfrWyLiIiIiMSJim0RERERkThRsT1NWlpayM7OJhgMUl1dfUi/u1NVVUUwGCQ3N5f29vYpjf3Od75DdnY2OTk53HjjjXGfh4iIiIhMnxmxZ9vMtgL/4O6PDWu7DvgA8D+AX8Saf+vuoYQnOImBgQEqKyvZsmULgUCAgoICQqEQZ5999lBMc3MzkUiESCTCtm3bqKioYNu2bROOfeqppwiHw+zcuZO5c+eyd+/eJM5SRERERA7XTFnZbgBKRrWVxNr/5O55sa8ZV2gDtLW1EQwGycrKIjU1lZKSEsLh8IiYcDhMaWkpZkZhYSF9fX309vZOOPa73/0uN910E3PnzgVg0aJFCZ+biIiIiBy5mVJsPwSsNrO5AGa2GMgAfpbMpKYqGo2SmZk5dB0IBIhGo1OKmWjsiy++yDPPPMOHPvQhPvKRj/Dzn/88zjMRERERkek0I4ptd98HtAEHP1evBNjg7g68x8y2m1mrmX1qvHuY2bWxuO39+/8Y/6SHGUzzkHymFDPR2LfffptXX32V1tZWbr/9di655JIx40VERERkZpoRxXbM8K0kB7eQAJzh7vnAZcCdZnbWWIPdvc7d8909P3VBWvyzHSYQCNDd3T103dPTQ0ZGxpRiJhobCAS4+OKLMTPOPfdc5syZwyuvvBLn2YiIiIjIdJlJxfYjwCozWwmc4O7tAO6+J/bnb4CtwF8lK8HxFBQUEIlE6Orqor+/n8bGRkKhkdvLQ6EQ9fX1uDutra2kpaWRnp4+4dhPfepTPPnkk8DglpL+/n5OPvnkhM9PRERERI7MjPg0EgB3PxD7VJJ7ia1qm9lJwBvu/qaZnQz8NXBb8rIcW0pKCjU1NRQVFTEwMEBZWRk5OTnU1tYCUF5eTnFxMU1NTQSDQebNm8f69esnHAtQVlZGWVkZy5cvJzU1lfvuu++Q7SkiIiIiMnPZTNoDbGZrgI3AMnd/wcz+K/A94B0GV+HvdPcfTHafE8/6gH/kW3fFJUcd1y4iIiIiB5nZjtiW5zHNmJVtAHffBNiw638HzkleRiIiIiIiR25GFdvTJXjSAq1Ai4iIiEjSzaQ3SIqIiIiIzCoqtkVERERE4kTFtoiIiIhInMzKPdsvvXqANQ9P30nvmz79N9N2LxERERF599DKtoiIiIhInKjYPgotLS1kZ2cTDAaprq4+pN/dqaqqIhgMkpubS3t7+6Rjv/GNb3D66aeTl5dHXl4eTU1NCZmLiIiIiEy/hBbbZrbVzIpGtV1nZveYWYuZ9ZnZj0f1/8DMnjOznWb2kJnNT2TO4xkYGKCyspLm5mY6OztpaGigs7NzRExzczORSIRIJEJdXR0VFRVTGnv99dfT0dFBR0cHxcXFCZ2XiIiIiEyfRK9sNwAlo9pKYu23A58fY8z17r7C3XOB3wJfjG+KU9PW1kYwGCQrK4vU1FRKSkoIh8MjYsLhMKWlpZgZhYWF9PX10dvbO6WxIiIiInLsS3Sx/RCw2szmApjZYiAD+Jm7PwG8NnqAu++PxRpwAjAjzpePRqNkZmYOXQcCAaLR6JRiJhtbU1NDbm4uZWVlvPrqq3GchYiIiIjEU0KLbXffB7QBB493LAE2uPuEBbSZrQdeBpYC3xkn5loz225m29/c3zd9SY9jrJQH/z4wecxEYysqKnjppZfo6OggPT2dL3/5y9OUsYiIiIgkWjLeIDl8K8nBLSQTcverGFwB/xVw6Tgxde6e7+75cxecOE2pji8QCNDd3T103dPTQ0ZGxpRiJhp76qmnctxxxzFnzhyuueYa2tra4jwTEREREYmXZBTbjwCrzGwlcIK7t08SD4C7DwAbgE/HMbcpKygoIBKJ0NXVRX9/P42NjYRCoRExoVCI+vp63J3W1lbS0tJIT0+fcGxvb+/Q+E2bNrF8+fKEzktEREREpk/CD7Vx9wNmthW4l0lWtWP7tM9y91/Hvv8E8EL8s5xcSkoKNTU1FBUVMTAwQFlZGTk5OdTW1gJQXl5OcXExTU1NBINB5s2bx/r16yccC3DjjTfS0dGBmbF48WK+973vJW2OIiIiInJ0bJLt0vF5qNkaYCOwzN1fiLU9w+Ce7PnAPuBqYAvwDLAAMOA5oOLgmybHc9JZS/2jt/3LtOWrEyRFREREZCxmtsPd88frT8px7e6+icHieXjbh8cJ/+v4ZyQiIiIiMv10gqSIiIiISJwkZWU73s46ab62foiIiIhI0mllW0REREQkTlRsi4iIiIjEyazcRvKbV//EZx/eeVT3ePDTudOUjYiIiIi8W2llW0REREQkTlRsH4GWlhays7MJBoNUV1cf0u/uVFVVEQwGyc3Npb29fdKx3/jGNzj99NPJy8sjLy+PpqamhMxFREREROInocW2mW01s6JRbdeZ2T1m1mJmfWb241H9q8ys3cw6zOxnZhZMZM6jDQwMUFlZSXNzM52dnTQ0NNDZ2Tkiprm5mUgkQiQSoa6ujoqKiimNvf766+no6KCjo4Pi4uKEzktEREREpl+iV7YbgJJRbSWx9tuBz48x5rvA5e6eB/wb8LV4JjiZtrY2gsEgWVlZpKamUlJSQjgcHhETDocpLS3FzCgsLKSvr4/e3t4pjRURERGR2SPRxfZDwGozmwtgZouBDOBn7v4E8NoYY5zB49oB0oA9CchzXNFolMzMzKHrQCBANBqdUsxkY2tqasjNzaWsrIxXX301jrMQERERkURIaLHt7vuANuCiWFMJsMHdfYJhXwCazKyHwZXvQzdJJ9BYqZrZlGImGltRUcFLL71ER0cH6enpfPnLX56mjEVEREQkWZLxBsnhW0kObiGZyPVAsbsHgPXAP40VZGbXmtl2M9v+5v74rQoHAgG6u7uHrnt6esjIyJhSzERjTz31VI477jjmzJnDNddcQ1tbW9zmICIiIiKJkYxi+xFglZmtBE5w9/bxAs3sFGCFu2+LNW0A/utYse5e5+757p4/d8FJ053zkIKCAiKRCF1dXfT399PY2EgoFBoREwqFqK+vx91pbW0lLS2N9PT0Ccf29vYOjd+0aRPLly+P2xxEREREJDESfqiNux8ws63AvUy+qv0qkGZmH3D3F4ELgF/FOcUJpaSkUFNTQ1FREQMDA5SVlZGTk0NtbS0A5eXlFBcX09TURDAYZN68eaxfv37CsQA33ngjHR0dmBmLFy/me9/7XtLmKCIiIiLTwybeLh2nh5qtATYCy9z9hVjbM8BSYD6wD7ja3R+Lxd4KvMNg8V3m7r+Z6P4Lz8rxVbdNVsdPTCdIioiIiMhkzGyHu+eP15+U49rdfRNgo9o+PEHspkTkJSIiIiIynXSCpIiIiIhInCRlZTvesk46QdtARERERCTptLItIiIiIhInKrZFREREROJkVm4j6e7rp2pT9+SB47hrTebkQSIiIiIik9DKtoiIiIhInKjYFhERERGJExXbh6mlpYXs7GyCwSDV1dWH9Ls7VVVVBINBcnNzaW9vn/LYO+64AzPjlVdeiescRERERCQxplRsm9kaM3MzWxrvhCbI4Tozm5es5wMMDAxQWVlJc3MznZ2dNDQ00NnZOSKmubmZSCRCJBKhrq6OioqKKY3t7u5my5YtnHHGGQmdk4iIiIjEz1RXttcCPwNK4pjLZK4Dklpst7W1EQwGycrKIjU1lZKSEsLh8IiYcDhMaWkpZkZhYSF9fX309vZOOvb666/ntttuw8xGP1ZEREREjlGTFttmNh/4a+BqYsW2mX3UzH5qZj8ysxfNrNrMLjezNjP7hZmdFYs708yeMLOdsT/PiLX/q5l9ZtgzDgy771Yze8jMXjCzB2xQFZABPGVmT037qzBF0WiUzMy/fFJJIBAgGo1OKWaisZs3b+b0009nxYoVcZ6BiIiIiCTSVFa2PwW0uPuLwB/MbGWsfQXwJeAc4PPAB9z9XOBfgP8Vi6kB6t09F3gAuGsKz/srBlexzwaygL9297uAPcB/c/f/NtYgM7vWzLab2fY/7f/DFB5z+Nx9rOdOKWa89jfeeIN169Zx6623Tl+iIiIiIjIjTKXYXgs0xr5vjF0D/Nzde939TeAl4PFY+y+AxbHvzwP+Lfb9D4G/mcLz2ty9x93fATqG3WtC7l7n7vnunn/CgoVTGXLYAoEA3d1/+fzunp4eMjIyphQzXvtLL71EV1cXK1asYPHixfT09LBy5UpefvnluMxBRERERBJnwkNtzOz9wMeA5WbmwHGAA03Am8NC3xl2/c4E9z24vPs2sULfBpeGU4fFDL/vwGQ5JlJBQQGRSISuri5OP/10Ghsb+bd/+7cRMaFQiJqaGkpKSti2bRtpaWmkp6dzyimnjDk2JyeHvXv3Do1fvHgx27dv5+STT0709ERERERkmk1WyH6GwW0g/+Ngg5n9lKmtUAP8O4P7vH8IXM7gmywBdgMfBH4EfBI4fgr3eg14H5C0z8VLSUmhpqaGoqIiBgYGKCsrIycnh9raWgDKy8spLi6mqamJYDDIvHnzWL9+/YRjRURERGT2mqzYXguM/kDoh4EKBreOTKYKuNfMbgB+D1wVa/8+EDazNuAJ4PUp3KsOaDaz3vH2bSdCcXExxcXFI9rKy8uHvjcz7r777imPHW337t1HnaOIiIiIzAw21hv3jnWnBnP90tsfPeLxd63JnDxIRERERN71zGyHu+eP1z9j9kNPp8wTU1Uwi4iIiEjS6bh2EREREZE4UbEtIiIiIhInKrZFREREROJkVu7Z3tv3Fndv+t0Rj69cc+o0ZiMiIiIi71Za2RYRERERiRMV24eppaWF7OxsgsEg1dWjP4Ic3J2qqiqCwSC5ubm0t7dPeewdd9yBmfHKK0k7t0dEREREplFCi20z22pmRaParjOzJjN71syeN7OdZnbpsP5VZtZuZh1m9jMzCyYy5+EGBgaorKykubmZzs5OGhoa6OzsHBHT3NxMJBIhEolQV1dHRUXFlMZ2d3ezZcsWzjjjjITOSURERETiJ9Er2w0MHt8+XAnwLaDU3XOAi4A7zezEWP93gcvdPQ/4N+BriUn1UG1tbQSDQbKyskhNTaWkpIRwODwiJhwOU1paiplRWFhIX18fvb29k469/vrrue222zCzRE9LREREROIk0cX2Q8BqM5sLYGaLgQzgaXePALj7HmAvcEpsjAMLYt+nAXsSmfBw0WiUzMy/HJYTCASIRqNTiplo7ObNmzn99NNZsWJFnGcgIiIiIomU0E8jcfd9ZtbG4Op1mMFV7Q0+7Mx4MzsXSAVeijV9AWgysz8B+4HCse5tZtcC1wKcdEogXvmP9dwpxYzX/sYbb7Bu3Toef/zx6UtURERERGaEZLxBcvhWkpLYNQBmlg78ELjK3d+JNV8PFLt7AFgP/NNYN3X3OnfPd/f8+QsWxiXxQCBAd3f30HVPTw8ZGRlTihmv/aWXXqKrq4sVK1awePFienp6WLlyJS+//HJc5iAiIiIiiZOMYvsRYJWZrQROcPd2ADNbADwKfM3dW2NtpwAr3H1bbOwG4L8mPuVBBQUFRCIRurq66O/vp7GxkVAoNCImFApRX1+Pu9Pa2kpaWhrp6enjjj3nnHPYu3cvu3fvZvfu3QQCAdrb2znttNOSNEsRERERmS4JP9TG3Q+Y2VbgXmKr2maWCmwC6t39wWHhrwJpZvYBd38RuAD4VYJTHpKSkkJNTQ1FRUUMDAxQVlZGTk4OtbW1AJSXl1NcXExTUxPBYJB58+axfv36CceKiIiIyOxlY+0ljvtDzdYAG4Fl7v6CmX2OwS0izw8Lu9LdO2KxtwLvMFh8l7n7bya6/xnBFf7V2498D7ROkBQRERGRqTCzHe6eP15/Uo5rd/dNgA27vh+4f4LYTQlKTURERERk2ugESRERERGROEnKyna8LTrxeG0FEREREZGk08q2iIiIiEicqNgWEREREYmTWbmNpO/Vt9n40CtHNPbiz5w8zdmIiIiIyLuVVrZFREREROJExfZhaGlpITs7m2AwSHV19SH97k5VVRXBYJDc3Fza29unPPaOO+7AzHjllSNbkRcRERGRmSehxbaZbTWzolFt15lZk5k9a2bPm9lOM7t0WP9/MbNtZhYxsw2x0yYTbmBggMrKSpqbm+ns7KShoYHOzs4RMc3NzUQiESKRCHV1dVRUVExpbHd3N1u2bOGMM85I6JxEREREJL4SvbLdAJSMaisBvgWUunsOcBFwp5mdGOv/FvBtd1/C4AmSVyco1xHa2toIBoNkZWWRmppKSUkJ4XB4REw4HKa0tBQzo7CwkL6+Pnp7eycde/3113PbbbdhZqMfKyIiIiLHsEQX2w8Bq81sLoCZLQYygKfdPQLg7nuAvcApNlh9fiw2DuA+4FMJzhmAaDRKZmbm0HUgECAajU4pZqKxmzdv5vTTT2fFihVxnoGIiIiIJFpCP43E3feZWRuDq9dhBle1N7i7H4wxs3OBVOAl4P1An7u/HevuAU4f695mdi1wLcDJJwfikftYz5xSzHjtb7zxBuvWrePxxx+fvkRFREREZMZIxhskh28lKYldA2Bm6cAPgavc/R1grH0Vh1augLvXuXu+u+enLXj/NKc8uBrd3d09dN3T00NGRsaUYsZrf+mll+jq6mLFihUsXryYnp4eVq5cycsvvzzt+YuIiIhI4iWj2H4EWGVmK4ET3L0dwMwWAI8CX3P31ljsK8CJZnZwBT4A7ElwvgAUFBQQiUTo6uqiv7+fxsZGQqHQiJhQKER9fT3uTmtrK2lpaaSnp4879pxzzmHv3r3s3r2b3bt3EwgEaG9v57TTTkvGFEVERERkmiX8UBt3P2BmW4F7ia1qxz5hZBNQ7+4PDot1M3sK+AzQCFzB4PaThEtJSaGmpoaioiIGBgYoKysjJyeH2tpaAMrLyykuLqapqYlgMMi8efNYv379hGNFREREZHazsfYTx/2hZmuAjcAyd3/BzD4HrAeeHxZ2pbt3mFkWg4X2QuA/gM+5+5sT3T94Vp7f9q2fHFFuOkFSRERERKbKzHa4e/54/Uk5rt3dNzFsP7a73w/cP07sb4BzE5SaiIiIiMi00QmSIiIiIiJxkpSV7Xg78aQUbQcRERERkaTTyraIiIiISJyo2BYRERERiZNZuY3ktT+8zVMP/P6Ixv63y0+Z5mxERERE5N1KK9siIiIiInGiYvswtLS0kJ2dTTAYpLq6+pB+d6eqqopgMEhubi7t7e1THnvHHXdgZrzyyitxnYOIiIiIJE5Ci20z22pmRaParjOze8ysxcz6zOzHo/o/ZmbtZvZLM7tv2NHtCTUwMEBlZSXNzc10dnbS0NBAZ2fniJjm5mYikQiRSIS6ujoqKiqmNLa7u5stW7ZwxhlnJHROIiIiIhJfiV7ZbgBKRrWVxNpvBz4/vMPM5gD3ASXuvhz4TwaPbE+4trY2gsEgWVlZpKamUlJSQjg88uT4cDhMaWkpZkZhYSF9fX309vZOOvb666/ntttuw8xGP1ZEREREjmGJLrYfAlab2VwAM1sMZAA/c/cngNdGxb8feNPdX4xdbwE+naBcR4hGo2RmZg5dBwIBotHolGImGrt582ZOP/10VqxYEecZiIiIiEiiJXRLhrvvM7M24CIgzOCq9gZ393GGvAIcb2b57r4d+AyQOU5sXI2V4uiV6PFixmt/4403WLduHY8//vj0JSoiIiIiM0Yy3iA5fCvJwS0kY4oV4SXAt2NF+mvA22PFmtm1ZrbdzLb/cf++aU55cDW6u7t76Lqnp4eMjIwpxYzX/tJLL9HV1cWKFStYvHgxPT09rFy5kpdffnna8xcRERGRxEtGsf0IsMrMVgInuHv7RMHu/qy7f9jdzwWeBiLjxNW5e76756cteP+0J11QUEAkEqGrq4v+/n4aGxsJhUIjYkKhEPX19bg7ra2tpKWlkZ6ePu7Yc845h71797J79252795NIBCgvb2d0047bdrzFxEREZHES/gne7j7ATPbCtzLBKvaB5nZInffG9vn/VVgXZxTHFNKSgo1NTUUFRUxMDBAWVkZOTk51NbWAlBeXk5xcTFNTU0Eg0HmzZvH+vXrJxwrIiIiIrObjb9dOo4PNVsDbASWufsLsbZngKXAfGAfcLW7P2ZmtwOrGVyF/6673znZ/bOz8rz277ccUW46QVJEREREpsrMdrh7/nj9SfnManffBNiotg+PE3sDcEMi8hIRERERmU46QVJEREREJE6SsrIdb+9bmKLtICIiIiKSdFrZFhERERGJExXbIiIiIiJxomJbRERERCROZuWe7TdeeZv/+Je9RzT2r76waJqzEREREZF3K61si4iIiIjEiYrtKWppaSE7O5tgMEh1dfUh/e5OVVUVwWCQ3Nxc2tvbpzz2jjvuwMx45ZVX4joHEREREUmshBbbZrbVzIpGtV1nZk1m9qyZPW9mO83s0mH9ZmbrzOxFM/uVmVUlMmeAgYEBKisraW5uprOzk4aGBjo7O0fENDc3E4lEiEQi1NXVUVFRMaWx3d3dbNmyhTPOOCOhcxIRERGR+Ev0ynYDUDKqrQT4FlDq7jnARcCdZnZirP9KIBNY6u7LgMbEpPoXbW1tBINBsrKySE1NpaSkhHA4PCImHA5TWlqKmVFYWEhfXx+9vb2Tjr3++uu57bbbMLPRjxURERGRY1yii+2HgNVmNhfAzBYDGcDT7h4BcPc9wF7g4Kk0FcCt7v5OrP/I3vl4FKLRKJmZmUPXgUCAaDQ6pZiJxm7evJnTTz+dFStWxHkGIiIiIpIMCS223X0f0Mbg6jUMrmpvcHc/GGNm5wKpwEuxprOAS81su5k1m9mSse5tZtfGYra/+tq+6c57rOdNKWa89jfeeIN169Zx6623Tl+iIiIiIjKjJOMNksO3kpTErgEws3Tgh8BVB1eygbnAn909H/g+cO9YN3X3OnfPd/f8k973/mlNOBAI0N3dPXTd09NDRkbGlGLGa3/ppZfo6upixYoVLF68mJ6eHlauXMnLL788rbmLiIiISPIko9h+BFhlZiuBE9y9HcDMFgCPAl9z99Zh8T3Aw7HvNwG5CcwVgIKCAiKRCF1dXfT399PY2EgoFBoREwqFqK+vx91pbW0lLS2N9PT0cceec8457N27l927d7N7924CgQDt7e2cdtppiZ6eiIiIiMRJwg+1cfcDZraVwRXqBgAzS2WwkK539wdHDXkE+Fgs/iPAiwlLNiYlJYWamhqKiooYGBigrKyMnJwcamtrASgvL6e4uJimpiaCwSDz5s1j/fr1E44VERERkdnPxtpTHPeHmq0BNgLL3P0FM/scsB54fljYle7eEftUkgeAM4ADQLm7PzfR/c9enOcPfO3xI8pNJ0iKiIiIyFSZ2Y7YducxJeW4dnffBNiw6/uB+8eJ7QM+npjMRERERESmj06QFBERERGJk6SsbMfbvJNTtB1ERERERJJOK9siIiIiInGiYltEREREJE5m5TaS/t+9xe47D+9wmMXX6fOtRURERGR6aWVbRERERCROVGxPQUtLC9nZ2QSDQaqrqw/pd3eqqqoIBoPk5ubS3t4+6dhbbrmF3Nxc8vLyuPDCC9mzZ09C5iIiIiIiiZPQYtvMtppZ0ai268zsHjNrMbM+M/vxqP5/NbMuM+uIfeUlMueBgQEqKytpbm6ms7OThoYGOjs7R8Q0NzcTiUSIRCLU1dVRUVEx6dgbbriBnTt30tHRwerVq7n11lsTOS0RERERSYBEr2w3ACWj2kpi7bcDnx9n3A3unhf76ohjfodoa2sjGAySlZVFamoqJSUlhMPhETHhcJjS0lLMjMLCQvr6+ujt7Z1w7IIFC4bGv/7665gZIiIiIjK7JPoNkg8B3zSzue7+ppktBjKAn7m7m9lHE5zPpKLRKJmZmUPXgUCAbdu2TRoTjUYnHXvzzTdTX19PWloaTz31VBxnISIiIiLJkNCVbXffB7QBF8WaSoAN7u6TDF1nZjvN7NtmNnesADO71sy2m9n2fa/vm86cx3rWlGImG7tu3Tq6u7u5/PLLqampmYZsRURERGQmScYbJIdvJTm4hWQi/wdYChQAC4GvjhXk7nXunu/u+e9/7/unK1cCgQDd3d1D1z09PWRkZEwpZipjAS677DIefvjhactZRERERGaGZBTbjwCrzGwlcIK7t08U7O69PuhNYD1wbgJyHFJQUEAkEqGrq4v+/n4aGxsJhUIjYkKhEPX19bg7ra2tpKWlkZ6ePuHYSCQyNH7z5s0sXbo0kdMSERERkQRI+KE27n7AzLYC9zL5qjZmlu7uvTa4/+JTwC/jm+FIKSkp1NTUUFRUxMDAAGVlZeTk5FBbWwtAeXk5xcXFNDU1EQwGmTdvHuvXr59wLMBNN93Erl27mDNnDmeeeebQ/URERERk9rDJt0vH4aFma4CNwDJ3fyHW9gyD20XmA/uAq939MTN7EjgFMKADKHf3AxPdPzdzhW/+8mOHlZNOkBQRERGRw2VmO9w9f7z+pBzX7u6bGCyeh7d9eJzYjyUkKRERERGRaaYTJEVERERE4iQpK9vxlnrq8doWIiIiIiJJp5VtEREREZE4UbEtIiIiIhIns3IbyVu/e5OX7/j1YY057SvBOGUjIiIiIu9WWtkWEREREYkTFduTaGlpITs7m2AwSHV19SH97k5VVRXBYJDc3Fza29snHXvLLbeQm5tLXl4eF154IXv27EnIXEREREQksRJabJvZVjMrGtV2nZk1mdmzZva8me00s0uH9T9gZrvM7Jdmdq+ZHZ+ofAcGBqisrKS5uZnOzk4aGhro7OwcEdPc3EwkEiESiVBXV0dFRcWkY2+44QZ27txJR0cHq1ev5tZbb03UlEREREQkgRK9st0AlIxqKwG+BZS6ew5wEXCnmZ0Y63+AwZMlzwFOAL6QmFShra2NYDBIVlYWqamplJSUEA6HR8SEw2FKS0sxMwoLC+nr66O3t3fCsQsWLBga//rrrzN4Er2IiIiIzDaJfoPkQ8A3zWyuu79pZouBDOBpj50b7+57zGwvg0e097l708HBZtYGBBKVbDQaJTMzc+g6EAiwbdu2SWOi0eikY2+++Wbq6+tJS0vjqaeeiuMsRERERCRZErqy7e77gDYGV69hcFV7w8FCG8DMzgVSgZeGj41tH/k80JKYbAf3Y482ehV6vJjJxq5bt47u7m4uv/xyampqpiFbEREREZlpkvEGyeFbSUpi1wCYWTrwQ+Aqd39n1Lh7GFwBf2asm5rZtWa23cy27zvwh2lJNBAI0N3dPXTd09NDRkbGlGKmMhbgsssu4+GHH56WfEVERERkZklGsf0IsMrMVgInuHs7gJktAB4FvuburcMHmNnXGdxW8r/Hu6m717l7vrvnv3/+wmlJtKCggEgkQldXF/39/TQ2NhIKhUbEhEIh6uvrcXdaW1tJS0sjPT19wrGRSGRo/ObNm1m6dOm05CsiIiIiM0vCD7Vx9wNmthW4l9iqtpmlApuAend/cHi8mX0BKAJWjbHaHVcpKSnU1NRQVFTEwMAAZWVl5OTkUFtbC0B5eTnFxcU0NTURDAaZN28e69evn3AswE033cSuXbuYM2cOZ5555tD9RERERGR2sbH2Fsf9oWZrgI3AMnd/wcw+B6wHnh8WdqW7d5jZ28B/Aq/F2je6+4Sflbci8xx/7EubDisnnSApIiIiIofLzHa4e/54/Uk5rt3dNwE27Pp+4P5xYmflkfIiIiIiMvvpBEkRERERkTiZlavGx586V9tCRERERCTptLItIiIiIhInKrZFREREROJExbaIiIiISJzMyj3bb/3uDX53547DGnPqdR+MUzYiIiIi8m6llW0RERERkThRsT2JlpYWsrOzCQaDVFdXH9Lv7lRVVREMBsnNzaW9vX3Ssbfccgu5ubnk5eVx4YUXsmfPnoTMRUREREQSK6HFtpltNbOiUW3XmVmTmT1rZs+b2U4zu3RY/7+aWZeZdcS+8hKV78DAAJWVlTQ3N9PZ2UlDQwOdnZ0jYpqbm4lEIkQiEerq6qioqJh07A033MDOnTvp6Ohg9erV3HrrhAdiioiIiMgxKtEr2w1Ayai2EuBbQKm75wAXAXea2YnDYm5w97zYV0dCMgXa2toIBoNkZWWRmppKSUkJ4XB4REw4HKa0tBQzo7CwkL6+Pnp7eyccu2DBgqHxr7/+OmaGiIiIiMw+iX6D5EPAN81srru/aWaLgQzgaXd3AHffY2Z7gVOAvgTnN0I0GiUzM3PoOhAIsG3btkljotHopGNvvvlm6uvrSUtL46mnnorjLEREREQkWRK6su3u+4A2BlevYXBVe8PBQhvAzM4FUoGXhg1dF9te8m0zmzvWvc3sWjPbbmbb//D6q9OV71jPmVLMZGPXrVtHd3c3l19+OTU1NdOQrYiIiIjMNMl4g+TwrSQlsWsAzCwd+CFwlbu/E2v+P8BSoABYCHx1rJu6e52757t7/sL3njQtiQYCAbq7u4eue3p6yMjImFLMVMYCXHbZZTz88MPTkq+IiIiIzCzJKLYfAVaZ2UrgBHdvBzCzBcCjwNfcvfVgsLv3+qA3gfXAuYlKtKCggEgkQldXF/39/TQ2NhIKhUbEhEIh6uvrcXdaW1tJS0sjPT19wrGRSGRo/ObNm1m6dGmipiQiIiIiCZTwQ23c/YCZbQXuJbaqbWapwCag3t0fHB5vZunu3muDezA+BfwyUbmmpKRQU1NDUVERAwMDlJWVkZOTQ21tLQDl5eUUFxfT1NREMBhk3rx5rF+/fsKxADfddBO7du1izpw5nHnmmUP3ExEREZHZxcbaWxz3h5qtATYCy9z9BTP7HIOr1s8PC7vS3TvM7EkG3yxpQAdQ7u4HJrr/isyz/fEv//CwctIJkiIiIiJyuMxsh7vnj9eflOPa3X0Tg8Xzwev7gfvHif1YovISEREREZlOOkFSRERERCROkrKyHW/HnzpP20JEREREJOm0si0iIiIiEicqtkVERERE4mRWbiN5a+9r/O6urVOKPbXqo3HNRURERETevbSyLSIiIiISJyq2J9DS0kJ2djbBYJDq6upD+t2dqqoqgsEgubm5tLe3Tzr2lltuITc3l7y8PC688EL27NmTkLmIiIiISOIltNg2s61mVjSq7TozazKzZ83seTPbaWaXDuv/gZk9F2t/yMzmJyLXgYEBKisraW5uprOzk4aGBjo7O0fENDc3E4lEiEQi1NXVUVFRMenYG264gZ07d9LR0cHq1au59dZbEzEdEREREUmCRK9sNwAlo9pKgG8Bpe6eA1wE3GlmJ8b6r3f3Fe6eC/wW+GIiEm1rayMYDJKVlUVqaiolJSWEw+ERMeFwmNLSUsyMwsJC+vr66O3tnXDsggULhsa//vrrDJ5CLyIiIiKzUaLfIPkQ8E0zm+vub5rZYiADeNpj58a7+x4z28vgEe197r4fwAar0hOAhJwvH41GyczMHLoOBAJs27Zt0phoNDrp2Jtvvpn6+nrS0tJ46qmn4jgLEREREUmmhK5su/s+oI3B1WsYXNXecLDQBjCzc4FU4KVhbeuBl4GlwHfGureZXWtm281s+x8O/HE6ch3rGVOKmWzsunXr6O7u5vLLL6empuaocxURERGRmSkZb5AcvpWkJHYNgJmlAz8ErnL3dw62u/tVDK6A/wq4lDG4e52757t7/sL5aUedZCAQoLu7e+i6p6eHjIyMKcVMZSzAZZddxsMPP3zUuYqIiIjIzJSMYvsRYJWZrQROcPd2ADNbADwKfM3dW0cPcvcBYAPw6UQkWVBQQCQSoauri/7+fhobGwmFQiNiQqEQ9fX1uDutra2kpaWRnp4+4dhIJDI0fvPmzSxdujQR0xERERGRJEj4oTbufsDMtgL3ElvVNrNUYBNQ7+4PHoyN7dM+y91/Hfv+E8ALicgzJSWFmpoaioqKGBgYoKysjJycHGprawEoLy+nuLiYpqYmgsEg8+bNY/369ROOBbjpppvYtWsXc+bM4cwzzxy6n4iIiIjMPjbW/uK4P9RsDbARWObuL5jZ54D1wPPDwq4EdgLPAAsAA54DKg6+aXI8K87I9se/8r0p5aITJEVERETkSJnZDnfPH68/Kce1u/smBovng9f3A/ePE/7XCUlKRERERGSa6QRJEREREZE4ScrKdrwdv+h92h4iIiIiIkmnlW0RERERkThRsS0iIiIiEiezchvJ23v/yN67/++UYhdVfiLO2YiIiIjIu5VWtkVERERE4kTF9jhaWlrIzs4mGAxSXV19SL+7U1VVRTAYJDc3l/b29knH3nLLLeTm5pKXl8eFF17Inj17EjIXEREREUmOhBbbZrbVzIpGtV1nZk1m9qyZPW9mO83s0jHGfsfMDiQiz4GBASorK2lubqazs5OGhgY6OztHxDQ3NxOJRIhEItTV1VFRUTHp2BtuuIGdO3fS0dHB6tWrufXWWxMxHRERERFJkkSvbDcAJaPaSoBvAaXungNcBNxpZiceDDCzfOBEEqStrY1gMEhWVhapqamUlJQQDodHxITDYUpLSzEzCgsL6evro7e3d8KxCxYsGBr/+uuvM3gCvYiIiIjMVokuth8CVpvZXAAzWwxkAE+7ewTA3fcAe4FTYjHHAbcDNyYqyWg0SmZm5tB1IBAgGo1OKWaysTfffDOZmZk88MADWtkWERERmeUSWmy7+z6gjcHVaxhc1d7g7n4wxszOBVKBl2JNXwQ2u3tvAvM8pG30KvR4MZONXbduHd3d3Vx++eXU1NRMQ7YiIiIiMlMl4w2Sw7eSlMSuATCzdOCHwFXu/o6ZZQCfBb4z2U3N7Foz225m2/cd+ONRJRgIBOju7h667unpISMjY0oxUxkLcNlll/Hwww8fVZ4iIiIiMrMlo9h+BFhlZiuBE9y9HcDMFgCPAl9z99ZY7F8BQeDXZrYbmGdmvx7rpu5e5+757p7//vlpR5VgQUEBkUiErq4u+vv7aWxsJBQKjYgJhULU19fj7rS2tpKWlkZ6evqEYyORyND4zZs3s3Tp0qPKU0RERERmtoQfauPuB8xsK3AvsVVtM0sFNgH17v7gsNhHgdMOXpvZAXcPxjvHlJQUampqKCoqYmBggLKyMnJycqitrQWgvLyc4uJimpqaCAaDzJs3j/Xr1084FuCmm25i165dzJkzhzPPPHPofiIiIiIyO9lYe4zj/lCzNcBGYJm7v2BmnwPWA88PC7vS3TtGjTvg7vMnu3/eGUv88a/+05Ry0QmSIiIiInKkzGyHu+eP15+U49rdfRNgw67vB+6fwrhJC20RERERkZlCJ0iKiIiIiMRJUla24y1lUZq2h4iIiIhI0mllW0REREQkTpLyBsl4M7PXgF3JzuNd5mTglWQn8S6i1zux9Honll7vxNLrnXh6zRMr3q/3me5+ynids3IbCbBroneFyvQzs+16zRNHr3di6fVOLL3eiaXXO/H0midWsl9vbSMREREREYkTFdsiIiIiInEyW4vtumQn8C6k1zyx9Honll7vxNLrnVh6vRNPr3liJfX1npVvkBQRERERmQlm68q2iIiIiEjSHRPFtpldZGa7zOzXZnbTGP1mZnfF+nea2crJxprZQjPbYmaR2J8nJWo+M92Rvt5mlmlmT5nZr8zseTP70rAx3zCzqJl1xL6KEzmnmewof753m9kvYq/p9mHt+vkex1H8fGcP+/ntMLP9ZnZdrE8/3+OYwuu91MyeNbM3zewrUxmrn++JHelrrt/hR+Yof8b1O/wwHcXPd/J+h7v7jP4CjgNeArKAVOA54OxRMcVAM2BAIbBtsrHAbcBNse9vAr6V7LnOhK+jfL3TgZWx798HvDjs9f4G8JVkz2+mfR3N6x3r2w2cPMZ99fMdh9d71H1eZvCzVfXzfXSv9yKgAFg3/DXU7++kvOb6HZ7A1zvWp9/hCXy9R90nYb/Dj4WV7XOBX7v7b9y9H2gEPjkq5pNAvQ9qBU40s/RJxn4SuC/2/X3Ap+I8j2PFEb/e7t7r7u0A7v4a8Cvg9EQmfww6mp/viejne2zT9XqvAl5y9/+Mf8rHtElfb3ff6+4/B946jLH6+R7fEb/m+h1+RI7mZ3wi+hkf23S93gn9HX4sFNunA93Drns49P/848VMNPZUd++FwV8wDP5NSI7u9R5iZouBvwK2DWv+Yuyf5e/VP4kNOdrX24HHzWyHmV07LEY/32Oblp9voARoGNWmn+9DTeW1PJKx+vke39G85kP0O3zKjvb11u/wwzMtP98k+Hf4sVBs2xhtoz9CZbyYqYyVkY7m9R7sNJsPPAxc5+77Y83fBc4C8oBe4B+POtPZ4Whf779295XA3wKVZnb+dCY3C03Hz3fq/7+9O1aNIoriMP7dwsagAW0UtFDwDSwsrAUtBDttTJt38B3srMQqWAmK2/sEgmhUBG1F2ICg9nos5g5Zlp1lZ+7e3Qx8PxhmMjN3yfw5OdxkdjbAXeDFzHHre7GSHmz/HqY4N3t4L6V528P7WUd9b7yHj2Gy/R24PPP1JeDHiucsGzttbw3n9dEav+cxK8mblNIpmib9PCJetidExDQi/kbEP+Apza0gFeYdEe36CHjFca7W92JFeWe3gXcRMW13WN+dVsl7yFjru1tJ5vbw/orytof3VpR3tvEePobJ9lvgWkrpSv5t5D4wmTtnAjxMjRvA73zbZdnYCbCXt/eA17UvZCQG551SSsAz4EtEPJ4dMPee13vAp3qXMColee+klM4ApJR2gFsc52p9L1bST1oPmLv9aH13WiXvIWOt726DM7eHD1KStz28v5Ke0tp8D6/15OU6F5pPB/hK8wTqo7xvH9jP2wl4ko9/BK4vG5v3nwfeAN/y+ty2r/OkLEPzBm7S3M45BN7n5U4+dpDPPaT5wbi47es8KUtB3ldpnsT+AHy2vuvmnY+dBn4Cu3OvaX0Pz/sCzV+r/gC/8vbZrrF5v/VdIXN7+MbztodvMO98bCs93P8gKUmSJFUyhreRSJIkSaPkZFuSJEmqxMm2JEmSVImTbUmSJKkSJ9uSJElSJU62JUmSpEqcbEuSJEmVONmWJEmSKvkPD4CdBNR65fYAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 864x648 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.feature_selection import RFE\n",
    "selector = RFE(rf, n_features_to_select=30, step=10,verbose=2)\n",
    "selector = selector.fit(X, y)\n",
    "selector.support_\n",
    "\n",
    "features = []\n",
    "for i in list(enumerate(selector.support_.tolist())):\n",
    "    if i[1] == True:\n",
    "        features.append(i[0])\n",
    "features\n",
    "X.columns[features]\n",
    "\n",
    "feature_imp = pd.Series(rf.feature_importances_[features],\n",
    "                        index=X_train.columns[features]).sort_values(ascending=False)\n",
    "plt.figure(figsize=(12,9))\n",
    "ax = sns.barplot(x=feature_imp, y=feature_imp.index)\n",
    "plt.title(\"Feature Importance\")\n",
    "\n",
    "\n",
    "for p in ax.patches:\n",
    "    ax.annotate(\"%.3f\" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),\n",
    "            xytext=(5, 0), textcoords='offset points', ha=\"left\", va=\"center\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "X2 = df[['V4','V10','V11','V12','V14','V17']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size = 0.15, stratify = y, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "from imblearn.over_sampling import SMOTE\n",
    "from imblearn.under_sampling import RandomUnderSampler\n",
    "from imblearn.pipeline import Pipeline\n",
    "os = SMOTE(random_state=42)\n",
    "os_data_X2, os_data_y2=os.fit_sample(X_train2, y_train2)\n",
    "os_data_X2 = pd.DataFrame(data=os_data_X2,columns= X2.columns)\n",
    "os_data_y2 = pd.DataFrame(data=os_data_y2,columns=[\"Class\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "rf2 = RandomForestClassifier().fit(os_data_X2, os_data_y2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[42606    42]\n",
      " [   16    58]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00     42648\n",
      "           1       0.58      0.78      0.67        74\n",
      "\n",
      "    accuracy                           1.00     42722\n",
      "   macro avg       0.79      0.89      0.83     42722\n",
      "weighted avg       1.00      1.00      1.00     42722\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred = rf2.predict(X_test2)\n",
    "print(confusion_matrix(y_test2, y_pred))\n",
    "print(classification_report(y_test2, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " precision-1 score for rf_model : 0.8639285714285714 std: 0.16920017428948422\n",
      "\n",
      " recall-1 score for rf_model : 0.7392857142857142 std: 0.1591591425368371\n",
      "\n",
      " f1-1 score for rf_model : 0.806133866133866 std: 0.11612843540403847\n",
      "\n"
     ]
    }
   ],
   "source": [
    "custom_scorer = {\n",
    "                 'precision-1': make_scorer(precision_score, average='weighted', labels=[1]),\n",
    "                 'recall-1': make_scorer(recall_score, average='weighted', labels = [1]),\n",
    "                 'f1-1': make_scorer(f1_score, average='weighted', labels = [1])\n",
    "                 }\n",
    "\n",
    "for i, j in custom_scorer.items():\n",
    "    scores = cross_val_score(rf2, X_test2, y_test2, cv = 10, scoring = j)\n",
    "    print(f\" {i} score for rf_model : {scores.mean()} std: {scores.std()}\\n\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ife6NlFRJQ0f"
   },
   "source": [
    "### Neural Network\n",
    "\n",
    "In the final step, you will make classification with Neural Network which is a Deep Learning algorithm. \n",
    "\n",
    "Neural networks are a series of algorithms that mimic the operations of a human brain to recognize relationships between vast amounts of data. They are used in a variety of applications in financial services, from forecasting and marketing research to fraud detection and risk assessment.\n",
    "\n",
    "A neural network contains layers of interconnected nodes. Each node is a perceptron and is similar to a multiple linear regression. The perceptron feeds the signal produced by a multiple linear regression into an activation function that may be nonlinear.\n",
    "\n",
    "In a multi-layered perceptron (MLP), perceptrons are arranged in interconnected layers. The input layer collects input patterns. The output layer has classifications or output signals to which input patterns may map. \n",
    "\n",
    "Hidden layers fine-tune the input weightings until the neural network’s margin of error is minimal. It is hypothesized that hidden layers extrapolate salient features in the input data that have predictive power regarding the outputs.\n",
    "\n",
    "You will discover **[how to create](https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5)** your deep learning neural network model in Python using **[Keras](https://keras.io/about/)**. Keras is a powerful and easy-to-use free open source Python library for developing and evaluating deep learning models.\n",
    "\n",
    "- The steps you are going to cover for this algorithm are as follows:\n",
    "\n",
    "   *i. Import Libraries*\n",
    "   \n",
    "   *ii. Define Model*\n",
    "    \n",
    "   *iii. Compile Model*\n",
    "   \n",
    "   *iv. Fit Model*\n",
    "   \n",
    "   *v. Prediction and Model Evaluating*\n",
    "   \n",
    "   *vi. Plot Precision and Recall Curve*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "i9Rl75fpMuHi"
   },
   "source": [
    "***i. Import Libraries***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {
    "executionInfo": {
     "elapsed": 1897,
     "status": "ok",
     "timestamp": 1610977899555,
     "user": {
      "displayName": "Owen l",
      "photoUrl": "",
      "userId": "01085249422681493006"
     },
     "user_tz": -180
    },
    "id": "LhEc3K9KMuHi"
   },
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization\n",
    "from tensorflow.keras.optimizers import Adam, RMSprop, SGD\n",
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "from sklearn.neural_network import MLPClassifier"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "gD6Rh1R8MuHi"
   },
   "source": [
    "***ii. Define Model***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {
    "executionInfo": {
     "elapsed": 2582,
     "status": "ok",
     "timestamp": 1610977900244,
     "user": {
      "displayName": "Owen l",
      "photoUrl": "",
      "userId": "01085249422681493006"
     },
     "user_tz": -180
    },
    "id": "4okQmpRpMuHi"
   },
   "outputs": [],
   "source": [
    "early_stop = EarlyStopping(monitor=\"val_loss\", mode=\"min\", patience = 10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "\n",
    "model.add(Dense(30,activation='relu')) \n",
    "model.add(BatchNormalization())\n",
    "model.add(Dense(15,activation='relu'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(Dense(units=1,activation='sigmoid'))\n",
    "opt = Adam(learning_rate=0.0001)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_IQN7--qMuHi"
   },
   "source": [
    "***iii. Compile Model***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {
    "executionInfo": {
     "elapsed": 2582,
     "status": "ok",
     "timestamp": 1610977900245,
     "user": {
      "displayName": "Owen l",
      "photoUrl": "",
      "userId": "01085249422681493006"
     },
     "user_tz": -180
    },
    "id": "f4W96rfHMuHi"
   },
   "outputs": [],
   "source": [
    "model.compile(loss='binary_crossentropy', optimizer=opt, metrics = [\"accuracy\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "PsM_5PhJMuHi"
   },
   "source": [
    "***iv. Fit Model***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {
    "executionInfo": {
     "elapsed": 2580,
     "status": "ok",
     "timestamp": 1610977900245,
     "user": {
      "displayName": "Owen l",
      "photoUrl": "",
      "userId": "01085249422681493006"
     },
     "user_tz": -180
    },
    "id": "cmkPKExFMuHj"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.5000 - accuracy: 0.7324 - val_loss: 0.2606 - val_accuracy: 0.9504\n",
      "Epoch 2/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.1013 - accuracy: 0.9667 - val_loss: 4.3292 - val_accuracy: 0.1457\n",
      "Epoch 3/100\n",
      "3777/3777 [==============================] - 7s 2ms/step - loss: 0.1000 - accuracy: 0.9668 - val_loss: 0.0323 - val_accuracy: 0.9989\n",
      "Epoch 4/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.0725 - accuracy: 0.9746 - val_loss: 0.0103 - val_accuracy: 0.9989\n",
      "Epoch 5/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.0827 - accuracy: 0.9712 - val_loss: 6.3686 - val_accuracy: 0.0456\n",
      "Epoch 6/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.0778 - accuracy: 0.9728 - val_loss: 0.3601 - val_accuracy: 0.8425\n",
      "Epoch 7/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.0749 - accuracy: 0.9738 - val_loss: 0.0734 - val_accuracy: 0.9987\n",
      "Epoch 8/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.0667 - accuracy: 0.9764 - val_loss: 2.5291 - val_accuracy: 0.3110\n",
      "Epoch 9/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.0686 - accuracy: 0.9760 - val_loss: 4.4584 - val_accuracy: 0.0855\n",
      "Epoch 10/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.0687 - accuracy: 0.9757 - val_loss: 0.0105 - val_accuracy: 0.9990\n",
      "Epoch 11/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.0674 - accuracy: 0.9765 - val_loss: 7.7451 - val_accuracy: 0.0485\n",
      "Epoch 12/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.0677 - accuracy: 0.9762 - val_loss: 0.0114 - val_accuracy: 0.9989\n",
      "Epoch 13/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.0646 - accuracy: 0.9774 - val_loss: 0.0212 - val_accuracy: 0.9956\n",
      "Epoch 14/100\n",
      "3777/3777 [==============================] - 8s 2ms/step - loss: 0.0624 - accuracy: 0.9778 - val_loss: 0.0529 - val_accuracy: 0.9991\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.keras.callbacks.History at 0x1fa1d083288>"
      ]
     },
     "execution_count": 129,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x=os_data_X.values, \n",
    "          y=os_data_y.values, \n",
    "          batch_size = 128,\n",
    "          epochs=100,\n",
    "          validation_data=(X_test.values, y_test.values), verbose=1, callbacks = [early_stop])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "InMeP9kgMuHj"
   },
   "source": [
    "***v. Prediction and Model Evaluating***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {
    "executionInfo": {
     "elapsed": 2579,
     "status": "ok",
     "timestamp": 1610977900246,
     "user": {
      "displayName": "Owen l",
      "photoUrl": "",
      "userId": "01085249422681493006"
     },
     "user_tz": -180
    },
    "id": "wRi_uFjIMuHj"
   },
   "outputs": [],
   "source": [
    "y_pred = model.predict_classes(X_test.values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00     42648\n",
      "           1       0.75      0.76      0.75        74\n",
      "\n",
      "    accuracy                           1.00     42722\n",
      "   macro avg       0.87      0.88      0.88     42722\n",
      "weighted avg       1.00      1.00      1.00     42722\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test.values, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'loss'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-154-74a374a4f790>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfigsize\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m6\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m6\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhistory\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhistory\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"loss\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlabel\u001b[0m\u001b[1;33m=\u001b[0m \u001b[1;34m\"train\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhistory\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhistory\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"val_loss\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlabel\u001b[0m\u001b[1;33m=\u001b[0m \u001b[1;34m\"validation\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mlegend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m;\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'loss'"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 432x432 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(6,6))\n",
    "plt.plot(model.history.history[\"loss\"], label= \"train\")\n",
    "plt.plot(model.history.history[\"val_loss\"], label= \"validation\")\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['__class__',\n",
       " '__delattr__',\n",
       " '__dict__',\n",
       " '__dir__',\n",
       " '__doc__',\n",
       " '__eq__',\n",
       " '__format__',\n",
       " '__ge__',\n",
       " '__getattribute__',\n",
       " '__gt__',\n",
       " '__hash__',\n",
       " '__init__',\n",
       " '__init_subclass__',\n",
       " '__le__',\n",
       " '__lt__',\n",
       " '__module__',\n",
       " '__ne__',\n",
       " '__new__',\n",
       " '__reduce__',\n",
       " '__reduce_ex__',\n",
       " '__repr__',\n",
       " '__setattr__',\n",
       " '__sizeof__',\n",
       " '__str__',\n",
       " '__subclasshook__',\n",
       " '__weakref__',\n",
       " '_chief_worker_only',\n",
       " '_implements_predict_batch_hooks',\n",
       " '_implements_test_batch_hooks',\n",
       " '_implements_train_batch_hooks',\n",
       " '_keras_api_names',\n",
       " '_keras_api_names_v1',\n",
       " '_supports_tf_logs',\n",
       " 'history',\n",
       " 'model',\n",
       " 'on_batch_begin',\n",
       " 'on_batch_end',\n",
       " 'on_epoch_begin',\n",
       " 'on_epoch_end',\n",
       " 'on_predict_batch_begin',\n",
       " 'on_predict_batch_end',\n",
       " 'on_predict_begin',\n",
       " 'on_predict_end',\n",
       " 'on_test_batch_begin',\n",
       " 'on_test_batch_end',\n",
       " 'on_test_begin',\n",
       " 'on_test_end',\n",
       " 'on_train_batch_begin',\n",
       " 'on_train_batch_end',\n",
       " 'on_train_begin',\n",
       " 'on_train_end',\n",
       " 'params',\n",
       " 'set_model',\n",
       " 'set_params',\n",
       " 'validation_data']"
      ]
     },
     "execution_count": 139,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dir(model.history)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'History' object is not subscriptable",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-77-a117503b8411>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfigsize\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m6\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m6\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhistory\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"loss\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlabel\u001b[0m\u001b[1;33m=\u001b[0m \u001b[1;34m\"train\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhistory\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"val_loss\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlabel\u001b[0m\u001b[1;33m=\u001b[0m \u001b[1;34m\"validation\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mlegend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m;\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: 'History' object is not subscriptable"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 432x432 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(6,6))\n",
    "plt.plot(model.history[\"loss\"], label= \"train\")\n",
    "plt.plot(model.history[\"val_loss\"], label= \"validation\")\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy : 95.07% (14.39%)\n",
      "precision-1 : 49.99% (42.01%)\n",
      "recall-1 : 49.82% (28.63%)\n",
      "f1-1 : 41.28% (27.90%)\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.wrappers.scikit_learn import KerasClassifier\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.metrics import make_scorer\n",
    "from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score\n",
    "def create_baseline():\n",
    "    # create model\n",
    "    model = Sequential()\n",
    "    model.add(Dense(30,activation='relu')) \n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Dense(15,activation='relu'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Dense(units=1,activation='sigmoid'))\n",
    "    opt = Adam(learning_rate=0.0001)\n",
    "    # Compile model\n",
    "    model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])\n",
    "    return model\n",
    "# evaluate model with standardized dataset\n",
    "early_stop = EarlyStopping(monitor=\"val_loss\", mode=\"min\", patience = 10)\n",
    "estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=128, verbose=0)\n",
    "kfold = StratifiedKFold(n_splits=10)\n",
    "custom_scorer = {'accuracy': make_scorer(accuracy_score),\n",
    "                 \n",
    "                 'precision-1': make_scorer(precision_score, average='weighted', labels=[1]),\n",
    "                 'recall-1': make_scorer(recall_score, average='weighted', labels = [1]),\n",
    "                 'f1-1': make_scorer(f1_score, average='weighted', labels = [1])\n",
    "                 }\n",
    "for i, j in custom_scorer.items():\n",
    "    results = cross_val_score(estimator, X_test.values, y_test.values, cv=kfold, n_jobs=-1, scoring = j, fit_params={'callbacks':early_stop})\n",
    "    print(i, \": %.2f%% (%.2f%%)\" % (results.mean()*100, results.std()*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy : 99.84% (0.05%)\n",
      "precision-1 : 43.09% (34.22%)\n",
      "recall-1 : 38.39% (33.00%)\n",
      "f1-1 : 38.24% (29.96%)\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.wrappers.scikit_learn import KerasClassifier\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.metrics import make_scorer\n",
    "from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score\n",
    "\n",
    "model = MLPClassifier(random_state=1, max_iter=300, learning_rate_init=0.0001,early_stopping=True).fit(os_data_X.values, \n",
    "                                                                                                           os_data_y.values)\n",
    "\n",
    "# evaluate model with standardized dataset\n",
    "kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)\n",
    "custom_scorer = {'accuracy': make_scorer(accuracy_score),\n",
    "                 \n",
    "                 'precision-1': make_scorer(precision_score, average='weighted', labels=[1]),\n",
    "                 'recall-1': make_scorer(recall_score, average='weighted', labels = [1]),\n",
    "                 'f1-1': make_scorer(f1_score, average='weighted', labels = [1])\n",
    "                 }\n",
    "for i, j in custom_scorer.items():\n",
    "    results = cross_val_score(model, X_test.values, y_test.values, cv=kfold, n_jobs=-1, scoring = j)\n",
    "    print(i, \": %.2f%% (%.2f%%)\" % (results.mean()*100, results.std()*100))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_JAEDNkjMuHj"
   },
   "source": [
    "***vi. Plot Precision and Recall Curve***"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "LpbiGnpIxVK3"
   },
   "source": [
    "## 4. Model Deployement\n",
    "You cooked the food in the kitchen and moved on to the serving stage. The question is how do you showcase your work to others? Model Deployement helps you showcase your work to the world and make better decisions with it. But, deploying a model can get a little tricky at times. Before deploying the model, many things such as data storage, preprocessing, model building and monitoring need to be studied.\n",
    "\n",
    "Deployment of machine learning models, means making your models available to your other business systems. By deploying models, other systems can send data to them and get their predictions, which are in turn populated back into the company systems. Through machine learning model deployment, can begin to take full advantage of the model you built.\n",
    "\n",
    "Data science is concerned with how to build machine learning models, which algorithm is more predictive, how to design features, and what variables to use to make the models more accurate. However, how these models are actually used is often neglected. And yet this is the most important step in the machine learning pipline. Only when a model is fully integrated with the business systems, real values ​​can be extract from its predictions.\n",
    "\n",
    "After doing the following operations in this notebook, jump to *Pycharm* and create your web app with Flask API."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "oCAYcMLEH_7P"
   },
   "source": [
    "### Save and Export the Model as .pkl\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "id": "MqluJ9yvIOex"
   },
   "outputs": [],
   "source": [
    "pickle.dump(rf2, open('rf2_model.pkl', 'wb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vaZP1N93IPQi"
   },
   "source": [
    "### Save and Export Variables as .pkl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {
    "id": "q_vA-dJWxfFH"
   },
   "outputs": [],
   "source": [
    "pickle.dump(X2.columns, open('variables.pkl', 'wb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "cm9Z__Y7MuHj"
   },
   "source": [
    "___\n",
    "\n",
    "<p style=\"text-align: center;\"><img src=\"https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV\" class=\"img-fluid\" alt=\"CLRSWY\"></p>\n",
    "\n",
    "___"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [
    "OV28RJBeMuHb",
    "hlm6gCsKMuHb",
    "zKZcwgucJQ0I",
    "4f8q5y12MuHe",
    "9wvBCEvpJQ0U",
    "_3zm70O7JQ0Z"
   ],
   "name": "Fraud Detection_Student_V2.ipynb",
   "provenance": [],
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
