# Zerodha Personal Project

## Prerequisites
- Python 3.x
- Django
- Other dependencies listed in `requirements.txt`

## Initialization
- Create a folder and open it in your terminal

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Utsav-sxn/AD-zerodha.git
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Database Setup and Data Import
- **Creating the Database Table**:
  1. Ensure that Django is installed and the project is set up.
  2. Run the command `python manage.py makemigrations` to create migrations for the `stock_data` model and set the name of the database to `zerodha_ad`.
  COnfigure the settings.py file to use the database you want to use.
  Eg- For MySQL 
  'ENGINE': 'django.db.backends.mysql',
   'NAME': 'zerodha_ad',
   'USER': <your_username>,
   'PASSWORD': <your_password>,
  3. Run the command `python manage.py migrate` to apply the migrations and create the table in the database.
- **Importing Data from CSV**:
  - Import into database.

## Running the Project
1. Run the Django development server:
   ```bash
   python manage.py runserver
   ```
2. Open your web browser and go to `http://127.0.0.1:8000/`.

## Model Prerequisites
- Ensure that required python libraries are installed. Use the command-
  ```bash
  pip install numpy
  ```
  ```bash
  pip install pandas
  ```
  ```bash
  pip install scikit-learn
  ```
  ```bash
  pip install tensorflow
  ```

- Data file requirements
  1. File: /workspaces/AD-zerodha/data/ind_market_dataset.csv
  2. Columns:
     1. Ticker: Stock Symbol
     2. Open, Close, High, Low, Volume: Numerical stock price data

- Hardware & Environment
  1. TensorFlow without GPU: The line os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ensures that the code runs on CPU only.
  2. Python version: Python 3.x

- Model Execution steps:
  1. Preprocess the dataset by executing the preprocessing.py file.
  2. Run the Stock_prediction-v1.py in a Python environment (Jupyter, VSCode, or terminal).
  3. The model will:
     1. Normalize the data.
     2. Train on stock data with adversarial augmentation.
     3. Predict the next day’s Open price for each ticker.
  5. Output displayed on the terminal will show the actual vs. predicted Open prices.
