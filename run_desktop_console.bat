@echo off
cd /d %~dp0
set PYTHONPATH=src
python -m qr_onboarding.cli desktop-console
