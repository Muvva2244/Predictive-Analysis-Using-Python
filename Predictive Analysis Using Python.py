import pyodbc
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from urllib.parse import quote_plus

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


# ============================================================================
# 1. SQL SERVER CONNECTION - WINDOWS AUTHENTICATION
# ============================================================================

SERVER = "Saichandra-Pc"
DATABASE = "Pro"

CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)


def get_sql_connection():
    return pyodbc.connect(CONN_STR)


# SQLAlchemy connection using the SAME SQL Server connection
ENGINE_STR = (
    "mssql+pyodbc:///?odbc_connect="
    + quote_plus(CONN_STR)
)


# ============================================================================
# 2. EXTRACT LIVE DATA FROM SQL SERVER
# ============================================================================

def extract_pmo_data():
    conn = get_sql_connection()

    query_tasks = """
    SELECT
        t.task_id,
        t.project_id,
        t.title AS task_title,
        t.progress_pct,
        t.sla_days,
        t.is_milestone,

        DATEDIFF(day, t.start_date, GETDATE()) AS elapsed_days,

        DATEDIFF(day, t.start_date, t.due_date)
            AS total_planned_days,

        CASE
            WHEN t.due_date < GETDATE()
                 AND t.progress_pct < 100
            THEN 1
            ELSE 0
        END AS is_overdue

    FROM dbo.tasks t;
    """

    query_financials = """
    SELECT
        p.project_id,
        p.name AS project_name,
        p.budget,
        p.profitability_index,
        f.forecasted_cost,
        f.actual_cost,
        f.revenue,

        (
            SELECT COUNT(*)
            FROM dbo.risks r
            WHERE r.project_id = p.project_id
              AND r.is_resolved = 0
        ) AS active_risks_count,

        (
            SELECT COUNT(*)
            FROM dbo.risks r
            WHERE r.project_id = p.project_id
              AND r.severity IN ('High', 'Critical')
              AND r.is_resolved = 0
        ) AS high_risks_count

    FROM dbo.projects p

    JOIN dbo.financials f
        ON p.project_id = f.project_id;
    """

    df_tasks = pd.read_sql_query(query_tasks, conn)
    df_financials = pd.read_sql_query(query_financials, conn)

    conn.close()

    return df_tasks, df_financials


# ============================================================================
# 3. ML MODEL 1 - TASK DELAY PROBABILITY
# ============================================================================

def predict_task_delays(df_tasks):
    print("\n--- Running Task Delay Model ---")

    if df_tasks.empty:
        print("No task records found in database.")
        return df_tasks

    # Expected progress based on elapsed time
    df_tasks["expected_progress_pct"] = np.where(
        df_tasks["total_planned_days"] > 0,
        np.clip(
            (
                df_tasks["elapsed_days"]
                / df_tasks["total_planned_days"]
            ) * 100,
            0,
            100
        ),
        100
    )

    # Difference between expected and actual progress
    df_tasks["progress_lag"] = (
        df_tasks["expected_progress_pct"]
        - df_tasks["progress_pct"]
    )

    # ML features
    X = df_tasks[
        [
            "progress_pct",
            "sla_days",
            "elapsed_days",
            "progress_lag"
        ]
    ]

    y = df_tasks["is_overdue"]

    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(0)

    # If insufficient training data
    if len(df_tasks) < 2 or y.nunique() < 2:
        df_tasks["delay_probability"] = np.clip(
            df_tasks["progress_lag"] / 100.0,
            0.0,
            1.0
        )
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression()
        clf.fit(X_scaled, y)

        df_tasks["delay_probability"] = (
            clf.predict_proba(X_scaled)[:, 1]
        )

    print(
        df_tasks[
            [
                "task_id",
                "task_title",
                "progress_pct",
                "delay_probability"
            ]
        ]
    )

    return df_tasks


# ============================================================================
# 4. ML MODEL 2 - COST OVERRUN PREDICTION
# ============================================================================

def predict_cost_overruns(df_financials):
    print("\n--- Running Cost Overrun Model ---")

    if df_financials.empty:
        print("No financial records found in database.")
        return df_financials

    X = df_financials[
        [
            "budget",
            "forecasted_cost",
            "active_risks_count"
        ]
    ]

    y = df_financials["actual_cost"]

    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(0)

    if len(df_financials) < 2:
        df_financials["predicted_final_cost"] = (
            df_financials["forecasted_cost"] * 1.05
        )
    else:
        reg = LinearRegression()
        reg.fit(X, y)

        df_financials["predicted_final_cost"] = reg.predict(X)

    # Predicted overrun
    df_financials["predicted_cost_overrun"] = (
        df_financials["predicted_final_cost"]
        - df_financials["budget"]
    )

    # Risk score
    df_financials["risk_score"] = np.clip(
        (
            df_financials["high_risks_count"] * 25
        )
        +
        (
            df_financials["predicted_cost_overrun"]
            .apply(lambda x: max(0, x))
            /
            df_financials["budget"].replace(0, np.nan)
            * 50
        ),
        0,
        100
    )

    df_financials["risk_score"] = (
        df_financials["risk_score"].fillna(0)
    )

    print(
        df_financials[
            [
                "project_id",
                "project_name",
                "budget",
                "predicted_final_cost",
                "risk_score"
            ]
        ]
    )

    return df_financials


# ============================================================================
# 5. SAVE PREDICTIONS BACK TO SQL SERVER
# ============================================================================

def save_predictions_to_sql(df_tasks, df_financials):
    print("\n--- Writing Predictions Back to SQL Server ---")

    # Combine task and financial predictions
    merged_predictions = pd.merge(
        df_tasks[
            [
                "task_id",
                "project_id",
                "delay_probability"
            ]
        ],
        df_financials[
            [
                "project_id",
                "predicted_final_cost",
                "predicted_cost_overrun",
                "risk_score"
            ]
        ],
        on="project_id",
        how="outer"
    )

    # Timestamp
    merged_predictions["generated_at"] = datetime.now()

    # Create SQL connection
    conn = get_sql_connection()
    cursor = conn.cursor()

    # Clear previous predictions
    cursor.execute("TRUNCATE TABLE dbo.project_predictions;")
    conn.commit()
    conn.close()

    # SQLAlchemy engine
    engine = create_engine(ENGINE_STR)

    # Write predictions to SQL Server
    merged_predictions.to_sql(
        name="project_predictions",
        con=engine,
        schema="dbo",
        if_exists="append",
        index=False
    )

    print(
        f"Successfully saved {len(merged_predictions)} records "
        f"to dbo.project_predictions."
    )


# ============================================================================
# 6. PDF REPORT GENERATION
# ============================================================================

class NumberedCanvas(canvas.Canvas):
    """Canvas class to generate dynamic page numbers and footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#666666"))
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(792 - 36, 20, page_text)  # Landscape width = 792 pt
        self.drawString(36, 20, "Confidential - PMO Predictive Analytics Report")
        self.restoreState()


def export_to_pdf(df_tasks, df_financials, output_filename="PMO_Predictions_Report.pdf"):
    print(f"\n--- Exporting Predictions to PDF: {output_filename} ---")

    # Use landscape A4 to fit wide data tables
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=landscape(A4),
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Document Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A365D'),
        alignment=0
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#4A5568')
    )

    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=12,
        spaceAfter=6
    )

    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#2D3748')
    )

    cell_header_style = ParagraphStyle(
        'TableHeaderCell',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    elements = []

    # Title & Subtitle Header
    elements.append(Paragraph("PMO Predictive Analytics Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    elements.append(Spacer(1, 15))

    # Helper function to construct structured PDF tables from DataFrames
    def build_table(df):
        df_clean = df.copy()
        float_cols = df_clean.select_dtypes(include=['float', 'float64']).columns
        df_clean[float_cols] = df_clean[float_cols].round(2)

        headers = [
            Paragraph(str(col).replace("_", " ").title(), cell_header_style)
            for col in df_clean.columns
        ]

        rows = [headers]
        for _, row in df_clean.iterrows():
            formatted_row = [Paragraph(str(val), cell_style) for val in row.values]
            rows.append(formatted_row)

        table = Table(rows, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6CB0')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
        ]))
        return table

    # 1. Task Predictions Table
    elements.append(Paragraph("Task Predictions", section_style))
    if not df_tasks.empty:
        display_task_cols = [
            col for col in [
                "task_id", "project_id", "task_title",
                "progress_pct", "sla_days", "is_overdue", "delay_probability"
            ] if col in df_tasks.columns
        ]
        elements.append(build_table(df_tasks[display_task_cols]))
    else:
        elements.append(Paragraph("No task prediction data available.", cell_style))

    elements.append(Spacer(1, 20))

    # 2. Financial Predictions Table
    elements.append(Paragraph("Financial Predictions", section_style))
    if not df_financials.empty:
        display_fin_cols = [
            col for col in [
                "project_id", "project_name", "budget",
                "forecasted_cost", "actual_cost", "predicted_final_cost",
                "predicted_cost_overrun", "risk_score"
            ] if col in df_financials.columns
        ]
        elements.append(build_table(df_financials[display_fin_cols]))
    else:
        elements.append(Paragraph("No financial prediction data available.", cell_style))

    # Build PDF File
    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated: {output_filename}")


# ============================================================================
# 7. MAIN PROGRAM EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"Connecting to SQL Server: {SERVER} | Database: {DATABASE}")

    # Test Database Connection
    try:
        test_conn = get_sql_connection()
        cursor = test_conn.cursor()
        cursor.execute("SELECT @@SERVERNAME, DB_NAME()")
        result = cursor.fetchone()
        print("Connected successfully!")
        print(f"Server: {result[0]}")
        print(f"Database: {result[1]}")
        test_conn.close()
    except Exception as e:
        print(f"Connection Failed: {e}")
        exit(1)

    # 1. Extract Data
    tasks, financials = extract_pmo_data()
    print(f"\nTasks loaded: {len(tasks)}")
    print(f"Financial records loaded: {len(financials)}")

    # 2. Run Machine Learning Models
    tasks_pred = predict_task_delays(tasks)
    financials_pred = predict_cost_overruns(financials)

    # 3. Save Predictions to SQL Database
    save_predictions_to_sql(tasks_pred, financials_pred)

    # 4. Generate PDF Report
    export_to_pdf(tasks_pred, financials_pred, "PMO_Predictions_Report.pdf")

    # 5. Console Outputs
    print("\n========== TASK PREDICTIONS ==========")
    print(tasks_pred.to_string())

    print("\n========== FINANCIAL PREDICTIONS ==========")
    print(financials_pred.to_string())

    print("\nPredictive analytics pipeline complete.")