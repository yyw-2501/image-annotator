@echo off
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+ 并勾选 "Add to PATH"。
    echo 安装后运行：pip install -r requirements.txt
    pause
    exit /b 1
)

python -c "import PySide6" >nul 2>nul
if errorlevel 1 (
    echo [提示] 缺少依赖 PySide6，正在安装主环境依赖……
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请手动运行：pip install -r requirements.txt
        pause
        exit /b 1
    )
)

python main.py
pause
