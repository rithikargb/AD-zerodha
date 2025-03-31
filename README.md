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
