@echo off
echo Setting up CUDA compilation environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
nvcc %*