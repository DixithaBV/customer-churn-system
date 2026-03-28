"""
sql_queries.py
--------------
All SQL queries used in the dashboard's SQL Explorer tab.
Each query is a dict with keys:
    title       – shown in the selectbox
    description – shown below the selectbox
    sql         – the actual query string
"""

QUERIES = [
    {
        "title": "1. Overall Churn Rate",
        "description": "Total customers, how many churned, and the churn percentage.",
        "sql": """
SELECT
    COUNT(*)                                        AS total_customers,
    SUM(Churn)                                      AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)        AS churn_rate_pct
FROM customers;
""",
    },
    {
        "title": "2. Churn Rate by Contract Type",
        "description": "Which contract type has the highest churn? Month-to-month customers are most at risk.",
        "sql": """
SELECT
    Contract,
    COUNT(*)                                        AS total,
    SUM(Churn)                                      AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)        AS churn_pct,
    ROUND(AVG(MonthlyCharges), 2)                   AS avg_monthly_charge
FROM customers
GROUP BY Contract
ORDER BY churn_pct DESC;
""",
    },
    {
        "title": "3. Revenue at Risk (Churned Customers)",
        "description": "How much monthly revenue is lost to churn per internet service type?",
        "sql": """
SELECT
    InternetService,
    COUNT(*)                                        AS churned_customers,
    ROUND(SUM(MonthlyCharges), 2)                   AS monthly_revenue_lost,
    ROUND(AVG(MonthlyCharges), 2)                   AS avg_charge
FROM customers
WHERE Churn = 1
GROUP BY InternetService
ORDER BY monthly_revenue_lost DESC;
""",
    },
    {
        "title": "4. Top 10 Highest-Risk Customers",
        "description": "Customers who churned with the highest monthly charges — biggest revenue loss.",
        "sql": """
SELECT
    customerID,
    Contract,
    InternetService,
    tenure,
    MonthlyCharges,
    TotalCharges
FROM customers
WHERE Churn = 1
ORDER BY MonthlyCharges DESC
LIMIT 10;
""",
    },
    {
        "title": "5. Churn by Payment Method",
        "description": "Electronic check users churn significantly more than auto-pay users.",
        "sql": """
SELECT
    PaymentMethod,
    COUNT(*)                                        AS total,
    SUM(Churn)                                      AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)        AS churn_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_pct DESC;
""",
    },
    {
        "title": "6. Tenure Cohort Analysis",
        "description": "New customers (0-12 months) churn much more than long-term ones.",
        "sql": """
SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 12 THEN '0-12 months (New)'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months'
        ELSE '48+ months (Loyal)'
    END                                             AS tenure_cohort,
    COUNT(*)                                        AS total,
    SUM(Churn)                                      AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)        AS churn_pct,
    ROUND(AVG(MonthlyCharges), 2)                   AS avg_charge
FROM customers
GROUP BY tenure_cohort
ORDER BY churn_pct DESC;
""",
    },
    {
        "title": "7. Senior Citizen Churn Analysis",
        "description": "Are senior citizens more likely to churn? Compare with non-seniors.",
        "sql": """
SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS segment,
    COUNT(*)                                        AS total,
    SUM(Churn)                                      AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)        AS churn_pct,
    ROUND(AVG(MonthlyCharges), 2)                   AS avg_monthly
FROM customers
GROUP BY SeniorCitizen;
""",
    },
    {
        "title": "8. Support Tickets vs Churn",
        "description": "Customers who raised more support tickets churn at higher rates.",
        "sql": """
SELECT
    c.customerID,
    c.Churn,
    c.Contract,
    COUNT(t.rowid)                                  AS num_tickets,
    ROUND(AVG(t.rating), 2)                         AS avg_rating
FROM customers c
LEFT JOIN support_tickets t ON c.customerID = t.customerID
GROUP BY c.customerID, c.Churn, c.Contract
ORDER BY num_tickets DESC
LIMIT 20;
""",
    },
    {
        "title": "9. Segment Summary Table",
        "description": "Pre-aggregated churn metrics by contract type and internet service.",
        "sql": """
SELECT
    Contract,
    InternetService,
    total_customers,
    churned,
    ROUND(churned * 100.0 / total_customers, 2)    AS churn_pct,
    avg_monthly,
    avg_tenure
FROM segment_summary
ORDER BY churn_pct DESC;
""",
    },
    {
        "title": "10. Average Transaction Amount by Churn",
        "description": "Do churned customers pay more or less per transaction on average?",
        "sql": """
SELECT
    c.Churn,
    COUNT(DISTINCT c.customerID)                    AS customers,
    COUNT(t.rowid)                                  AS total_transactions,
    ROUND(AVG(t.amount), 2)                         AS avg_txn_amount,
    ROUND(SUM(CASE WHEN t.paid_on_time = 0 THEN 1 ELSE 0 END) * 100.0
          / COUNT(t.rowid), 2)                      AS late_payment_pct
FROM customers c
JOIN transactions t ON c.customerID = t.customerID
GROUP BY c.Churn;
""",
    },
]