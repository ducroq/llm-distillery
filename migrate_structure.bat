@echo off
REM One-time migration script to reorganize trained models
REM Run this on GPU machine after git pull

echo === LLM Distillery Structure Migration ===
echo.
echo This will move trained models to the new structure:
echo   inference/deployed/{filter}_v{version}/ -^> filters/{filter}/v{version}/
echo.

REM Check if old structure exists
if not exist "inference\deployed" (
    echo [INFO] No inference/deployed/ directory found - nothing to migrate
    exit /b 0
)

REM Check for uplifting_v1 (the one we know exists)
if exist "inference\deployed\uplifting_v1\model" (
    echo Found: uplifting_v1
    echo.

    REM Check if filter directory exists
    if not exist "filters\uplifting\v1" (
        echo [WARNING] Filter directory not found: filters\uplifting\v1
        echo [WARNING] Run 'git pull' first to get the updated structure
        pause
        exit /b 1
    )

    REM Backup if model already exists
    if exist "filters\uplifting\v1\model" (
        echo [INFO] Model already exists in new location, backing up...
        move "filters\uplifting\v1\model" "filters\uplifting\v1\model.backup.old"
    )

    REM Move model
    echo Moving model to filters\uplifting\v1\model\...
    move "inference\deployed\uplifting_v1\model" "filters\uplifting\v1\"
    if errorlevel 1 (
        echo [ERROR] Failed to move model
        pause
        exit /b 1
    )

    REM Move training metadata if exists
    if exist "inference\deployed\uplifting_v1\training_history.json" (
        echo Moving training_history.json...
        move "inference\deployed\uplifting_v1\training_history.json" "filters\uplifting\v1\"
    )

    if exist "inference\deployed\uplifting_v1\training_metadata.json" (
        echo Moving training_metadata.json...
        move "inference\deployed\uplifting_v1\training_metadata.json" "filters\uplifting\v1\"
    )

    REM Move plots to reports if they exist
    if exist "inference\deployed\uplifting_v1\plots" (
        echo Moving plots to reports\uplifting_v1_plots\...
        if not exist "reports\uplifting_v1_plots" mkdir "reports\uplifting_v1_plots"
        move "inference\deployed\uplifting_v1\plots\*" "reports\uplifting_v1_plots\" 2>nul
    )

    REM Move report if exists
    if exist "inference\deployed\uplifting_v1\*.docx" (
        echo Moving training report...
        if not exist "reports" mkdir "reports"
        for %%f in (inference\deployed\uplifting_v1\*.docx) do (
            move "%%f" "reports\uplifting_v1_training_report.docx"
        )
    )

    echo [OK] Migration complete for uplifting_v1
    echo.
) else (
    echo [INFO] No uplifting_v1 model found in inference/deployed/
    echo.
)

REM Clean up old structure
echo.
echo Checking if old structure can be removed...

REM Count remaining files
dir /s /b "inference\deployed" 2>nul | find /c /v "" > temp_count.txt
set /p REMAINING=<temp_count.txt
del temp_count.txt

if "%REMAINING%"=="0" (
    echo [INFO] Old structure is empty
    choice /C YN /M "Remove inference\deployed\ directory?"
    if errorlevel 2 goto skip_remove
    if errorlevel 1 (
        rmdir /s /q "inference\deployed"
        echo [OK] Removed inference\deployed\
    )
) else (
    echo [INFO] Some files remain in inference\deployed\
    echo [INFO] Review manually before removing
)

:skip_remove
echo.
echo === Migration Complete ===
echo.
echo Verify your model:
echo   filters\uplifting\v1\model\
echo.
echo Reports and plots:
echo   reports\uplifting_v1_training_report.docx
echo   reports\uplifting_v1_plots\
echo.
echo New training will automatically use the correct structure.
echo.
pause
