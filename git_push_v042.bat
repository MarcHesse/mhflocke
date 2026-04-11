@echo off
cd /d D:\claude\mhflocke

git add -A
git status

echo.
echo ============================================================
echo   Ready to commit. Press any key to commit and push...
echo ============================================================
pause

git commit -m "v0.4.2: Unified codebase + Brain3D visualization + Freenove sim-to-real"
git push origin main

echo.
echo Done!
pause
