@echo off
REM Advanced batch script to dynamically load environment variables from .env file
REM Usage: load_env.bat

echo Loading environment variables from .env file...

REM Clean up any existing temp files
if exist "temp_env.bat" del temp_env.bat
if exist "temp_env.batch" del temp_env.batch

REM Check if .env file exists
if not exist ".env" (
    echo ERROR: .env file not found in current directory!
    echo Please make sure you're running this script from the project root directory.
    pause
    exit /b 1
)

echo ================================
echo Processing .env file...
echo ================================

REM Create temporary batch files
echo @echo off > temp_env.bat
echo @echo off > set_env.bat

REM Process each line in .env file
for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    set "var_name=%%a"
    set "var_value=%%b"
    
    REM Skip lines without = or starting with #
    if not "%%a"=="" (
        if not "%%a:~0,1%"=="#" (
            if not "%%b"=="" (
                REM Clean the variable name (trim spaces)
                call :trim var_name "%%a"
                
                REM Clean the value (remove quotes and inline comments)
                call :clean_value var_value "%%b"
                
                REM Add to both temp files
                call echo Set %%var_name%%=%%var_value%% >> temp_env.bat
                call echo set "%%var_name%%=%%var_value%%" >> set_env.bat
            )
        )
    )
)

echo.
echo ================================
echo Setting environment variables...
echo ================================

REM Display what's being set verbosely
for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    set "var_name=%%a"
    set "var_value=%%b"
    
    REM Skip lines without = or starting with #
    if not "%%a"=="" (
        if not "%%a:~0,1%"=="#" (
            if not "%%b"=="" (
                REM Clean the variable name (trim spaces)
                call :trim var_name "%%a"
                
                REM Clean the value (remove quotes and inline comments)
                call :clean_value var_value "%%b"
                
                REM Display in the requested format
                call echo Set %%var_name%%=%%var_value%%
            )
        )
    )
)

REM Actually set the environment variables in current session
call set_env.bat

REM Clean up temp files
del temp_env.bat
del set_env.bat

echo.
echo ================================
echo Environment variables loaded successfully
echo ================================
echo.
echo Verification:
echo - AWS_REGION1: %AWS_REGION1%
echo - BEDROCK_MODEL_ID: %BEDROCK_MODEL_ID%
echo - ATLAS_DB_NAME: %ATLAS_DB_NAME%
echo - ATLAS_URI: %ATLAS_URI%
echo - SHOW_CLOUDWATCH_LOGS: %SHOW_CLOUDWATCH_LOGS%
echo.
echo You can now use these environment variables in your current terminal session.
echo To verify any variable, use: echo %%VARIABLE_NAME%%
echo.

goto :eof

REM Function to trim leading and trailing spaces
:trim
set "result=%~2"
REM Remove leading spaces
:trim_leading
if "%result:~0,1%"==" " (
    set "result=%result:~1%"
    goto :trim_leading
)
REM Remove trailing spaces
:trim_trailing
if "%result:~-1%"==" " (
    set "result=%result:~0,-1%"
    goto :trim_trailing
)
set "%~1=%result%"
goto :eof

REM Function to clean value (remove quotes and comments)
:clean_value
set "result=%~2"
REM Remove surrounding quotes
if "%result:~0,1%"=="""" (
    if "%result:~-1%"=="""" (
        set "result=%result:~1,-1%"
    )
)
REM Remove inline comments (space followed by #)
for /f "tokens=1 delims=#" %%x in ("%result%") do set "result=%%x"
REM Trim the result
call :trim result "%result%"
set "%~1=%result%"
goto :eof