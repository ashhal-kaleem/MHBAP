@echo off
cd /d D:\MHBAP
python -m pytest ml\tests\ -x -q --tb=short 2>&1
echo EXIT_CODE=%ERRORLEVEL%
