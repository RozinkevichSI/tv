@echo off
cd /d "C:\Users\stani\Downloads\project_televizors"

echo Установка зависимостей...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Запуск приложения Flask...
start "" http://127.0.0.1:5000/
python app.py

pause