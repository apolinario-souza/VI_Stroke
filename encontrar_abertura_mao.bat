@echo off
call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
call conda activate base
python encontrar_abertura_mao.py
