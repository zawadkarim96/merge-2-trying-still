# PS Business Suites by ZAD

PS Business Suites by ZAD is a Streamlit application for managing quotations, work orders,
delivery orders and follow-up notifications for sales teams. The tool focuses on
sales workflows (from quotation to payment collection) while adopting the
structure and best practices of the PS Service Software project.

## Features

- **Secure authentication** – Username/password login with SHA-256 hashing and
  role-based access (admin vs. staff).
- **Quotation management** – Create quotations, upload PDF documents, manage
  statuses (pending, accepted, declined, inform later) and schedule follow-up
  reminders automatically.
- **Work & delivery orders** – Track downstream documents, upload PDFs and
  monitor missing steps through automated notifications.
- **Company, category and district catalogue** – Maintain master data with
  multi-category assignments per company.
- **Notifications** – Built-in reminder system covering follow-ups, missing work
  orders/delivery orders and overdue payments.
- **Dashboards** – Streamlit charts for quotation trends, revenue insights and
  performance metrics.
- **Configurable grace periods** – Admins can adjust reminder lead times from
  the settings page.

## Installation

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Override the data directory by setting `PS_SALES_DATA_DIR` if the
   default home-based location is not writable (for example on Render).

4. Run the application with Streamlit:

   ```bash
   streamlit run sales_app.py
   ```

The application stores its SQLite database and uploads under the user-specific
`~/.ps_sales/` directory (or `%APPDATA%\ps_sales\` on Windows). On container
platforms such as Render, the app falls back to a `.ps_sales/` directory inside
the working directory so the service can write files. The directory is created
automatically on first launch together with default configuration settings and a
pair of seed accounts for testing: an administrator (`admin` / `admin`) and a
staff salesperson (`salesperson` / `admin`).

## Usage Overview

- **Login** with your credentials. Use the seeded admin account on first run and
  change the password immediately.
- **Navigation** is available via the sidebar. Admin users can access all
  management pages; staff users see only their own records.
- **Quotations** support inline company creation, category insights, follow-up
  reminders and PDF uploads stored safely under the uploads directory.
- **Work orders** become available after quotations are marked as accepted; the
  system warns users if work orders or delivery orders are missing after the
  configured grace periods.
- **Delivery orders** include revenue tracking and payment monitoring. Overdue
  payments automatically create notifications for both the salesperson and admin
  team.
- **Notifications** can be reviewed and marked as read. The dashboard also shows
  upcoming reminders.

## Desktop Packaging

To distribute a desktop executable (for Windows/macOS/Linux), use tools such as
`pyinstaller` or `briefcase`. Ensure that the packaging process preserves the
`~/.ps_sales/` data directory so documents and the SQLite database remain
accessible between sessions.

## License

This project inherits the licensing terms of the surrounding repository. Review
those terms before redistribution.
