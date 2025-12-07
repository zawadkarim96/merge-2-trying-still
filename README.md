# PS Business Suites

## Release Notes
- v7.4: Fixed syntax and navigation chain. Dashboard shows only upcoming warranties (3 & 60 days). Removed Products/Orders pages, kept tables for import integrity. Customers page supports Needs (product/unit). DD-MM-YYYY display, flexible importer + Excel serial dates. Login safe rerun for all Streamlit versions.
- v7.5: Fixed selectbox errors; "Add Customer" now supports Purchase/Warranty fields (date, product, model, serial, unit) and auto-creates warranty.
- v7.6: Admins can bulk-delete customers from the Customers page with a select-all option.
- v7.7: Importer allows blank column mappings and optional skipping of blank rows. Dashboard counts active warranties by expiry date. Customer summaries merge duplicate records and report how many were combined.
- v7.8: Customer summary groups unnamed customers under "(blank)" so their contact info is still accessible.
- v7.9: Staff can only submit the current day's daily report, file weekly updates on Saturdays, and record monthly reports for the active month. Client imports better distinguish repeat purchases with different generators to avoid duplicate flags.

### Recent fixes
- `run_app.py` now bootstraps `pywebview` automatically and shows a clear message if the install fails, so the desktop window opens reliably even when the module was missing on the server.
- The sales launcher stores its database, uploads, and templates in the per-OS app data folder instead of the repository checkout to avoid permission issues on deployed servers.

## Run Without Touching the Command Line
These launchers create a dedicated virtual environment, install dependencies from `requirements.txt`, and then open the Streamlit app inside the native desktop shell (no browser required). After the first setup run the cached environment is reused automatically, so relaunching is instant. You only need Python 3.9+ installed. You can still run `streamlit run main.py` for the traditional browser experience—both approaches share the exact same database and uploads.

To launch the PS Business Suites by ZAD sales experience instead of the service CRM, set `PS_APP=sales` (or `PS_APP_SCRIPT=sales_app.py` when using the desktop launcher) before starting the application. The same unified requirements file and Procfile work for both apps on Render or Railway.

### Any Platform (single command or double-click)
Run `python run_app.py` from the repository root, or double-click the file in your file explorer. The script prepares the environment and launches the app inside a pywebview dialog titled **PS Business Suites**.

### Windows
1. Double-click `run_app.bat`.
2. Wait for the first run to finish creating the virtual environment and installing dependencies. A desktop window titled **PS Business Suites** will open automatically.
3. Subsequent launches skip the setup step—the cached environment is reused so the window opens right away.

### macOS & Linux
1. (First time only) Make the script executable: `chmod +x run_app.sh`.
2. Double-click `run_app.sh` in your file manager **or** run `./run_app.sh` from a terminal.
3. The script prepares the environment and opens the app inside a desktop window—no external browser involved. Future launches reuse the prepared environment.

## Build a One-Click Desktop App for Your Team
If your staff does not have Python installed, you can package the project into a standalone application and distribute it like a regular program.

1. On a machine with Python 3.9+ installed, open a terminal in the repository root.
2. Run `python build_executable.py`. The script creates a temporary virtual environment, installs PyInstaller, and produces a bundle inside `dist/PS Service Software/`. Subsequent builds reuse the cached environment unless `requirements.txt` changes.
3. Share the contents of `dist/PS Service Software/` with your staff. Windows users can double-click `PS Service Software.exe` (complete with the PS Service Software icon); macOS and Linux users can run the executable from Finder/File Explorer or the terminal.

### Where user data lives
Regardless of whether you launch from source, the packaged desktop app, or `streamlit run main.py`, databases, uploads, and the Excel import template are stored in a writable folder per operating system:

- **Windows:** `%APPDATA%\ps-business-suites`
- **macOS:** `~/Library/Application Support/ps-business-suites`
- **Linux:** `${XDG_DATA_HOME:-~/.local/share}/ps-business-suites`

Staff can back up or migrate the application by copying that folder. Deleting it resets the app to a clean state the next time the executable is opened.

## Troubleshooting
- If Python is not installed or not on your `PATH`, install it from [python.org](https://www.python.org/downloads/) (Windows) or via your package manager (macOS/Linux).
- To reset everything, delete the `.venv` folder and rerun the launcher to recreate a clean environment.
- After repository cleanup, run the bundled test suite with `pytest` from the repository root. All tests should pass, confirming that removing legacy PS-SALES copies did not break the active application paths.
- If the desktop launcher reports that `pywebview` is missing, rerun `python run_app.py` to auto-install it (the script now bootstraps `pywebview` when absent). You can also reinstall dependencies with `pip install -r requirements.txt` to ensure the native window backend is available.
