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

## Running the Project
1. Run the Django development server:
   ```bash
   python manage.py runserver
   ```
2. Open your web browser and go to `http://127.0.0.1:8000/`.

## Additional Notes
- Ensure that your database is set up correctly in the `settings.py` file.
- Migrate the database if necessary:
   ```bash
   python manage.py migrate
