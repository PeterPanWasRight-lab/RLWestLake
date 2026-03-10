@echo off
echo 连续矩形世界强化学习环境测试脚本
echo.

echo 检查Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python。请确保已安装Python 3.8或更高版本。
    pause
    exit /b 1
)

echo 检查依赖...
python -c "import gymnasium, numpy, matplotlib" >nul 2>&1
if %errorlevel% neq 0 (
    echo 依赖未安装，正在安装...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo 错误: 依赖安装失败。
        pause
        exit /b 1
    )
    echo 依赖安装成功。
) else (
    echo 依赖已安装。
)

echo.
echo 运行测试...
python tmp.py

echo.
echo 测试完成。
pause