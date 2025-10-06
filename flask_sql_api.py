from flask import Flask, jsonify, request
import pyodbc
import pandas as pd
from datetime import datetime
from flask import session
from sqlalchemy import text


app = Flask(__name__)

# Database connection string
conn_str = (
    'DRIVER={ODBC Driver 18 for SQL Server};'
    'SERVER=DESKTOP-VUKR5KV;'
    'DATABASE=Clout;'
    'Trusted_Connection=yes;'
    'TrustServerCertificate=yes;'
)

# Utility function to run SQL or stored procedures
def run_query(query, use_proc=False):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    if use_proc:
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
        data = [dict(zip(columns, row)) for row in rows]
        cursor.close()
        conn.close()
        return data
    else:
        df = pd.read_sql(query, conn)
        conn.close()
        return df.to_dict(orient="records")

def current_user_id() -> str | None:
    # Your auth layer should set this at login
    return session.get("user_id")

@app.route('/login', methods=['POST'])
def login():
    # Authenticate user, then:
    session["user_id"] = "b.723.smith@gmail.com"  # Replace with real user
    return redirect("/dashboard")  # or wherever your Dash app lives


# GET: /api/scorecard
@app.route('/api/scorecard')
def scorecard():
    try:
        raw_data = run_query("EXEC GetSubjectSentimentScores", use_proc=True)
        for row in raw_data:
            if "NormalizedSentimentScore" in row and row["NormalizedSentimentScore"] is not None:
                row["NormalizedSentimentScore"] = int(row["NormalizedSentimentScore"])
        return jsonify(raw_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET: /api/timeseries
@app.route('/api/timeseries')
def timeseries():
    try:
        query = """
        SELECT
            CAST(SentimentDate AS DATE) AS SentimentDate,
            Subject,
            CAST(ROUND((AVG(SentimentScore) + 1.0) * 5000.0, 0) AS INT) AS NormalizedSentimentScore
        FROM (
            SELECT Subject, SentimentScore, CreatedUTC AS SentimentDate FROM RedditCommentsUSSenate
            UNION ALL
            SELECT Subject, SentimentScore, CreatedUTC AS SentimentDate FROM BlueskyCommentsUSSenate
        ) AS Combined
        GROUP BY CAST(SentimentDate AS DATE), Subject
        ORDER BY SentimentDate ASC, Subject ASC
        """
        return jsonify(run_query(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET: /api/traits
@app.route('/api/traits')
def traits():
    try:
        query = """
        SELECT Subject, TraitType, TraitRank, TraitDescription, CreatedUTC
        FROM SubjectTraitSummary
        ORDER BY Subject, TraitType, CreatedUTC
        """
        return jsonify(run_query(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET: /api/bill-sentiment
@app.route('/api/bill-sentiment')
def bill_sentiment():
    try:
        query = """
        SELECT
            BillName,
            CAST(ROUND(AVG(SentimentScore), 2) AS FLOAT) AS AverageSentimentScore,
            CASE
                WHEN AVG(SentimentScore) >= 0.50 THEN 'Very Positive'
                WHEN AVG(SentimentScore) >= 0.05 THEN 'Slightly Positive'
                WHEN AVG(SentimentScore) > -0.05 AND AVG(SentimentScore) < 0.05 THEN 'Neutral'
                WHEN AVG(SentimentScore) <= -0.05 THEN 'Slightly Negative'
                WHEN AVG(SentimentScore) <= -0.50 THEN 'Very Negative'
            END AS SentimentLabel
        FROM NationalBillSentimentMentions
        WHERE CAST(ReasonSummary AS VARCHAR(MAX)) <> 'No reference to the bill found.'
        GROUP BY BillName
        ORDER BY AverageSentimentScore DESC
        """
        return jsonify(run_query(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET: /api/top-issues
@app.route('/api/top-issues')
def top_issues():
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(WeekStartDate) FROM WeeklyIdeologyTopics")
        latest_week = cursor.fetchone()[0]

        cursor.execute("""
            SELECT WeekStartDate, IdeologyLabel, Rank, Topic
            FROM WeeklyIdeologyTopics
            WHERE WeekStartDate = ?
            ORDER BY IdeologyLabel, Rank ASC
        """, (latest_week,))

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        results = {
            "WeekStartDate": str(latest_week),
            "Liberal": [],
            "Conservative": []
        }

        for row in rows:
            results[row.IdeologyLabel].append({
                "Rank": row.Rank,
                "Topic": row.Topic
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET: /api/common-ground-issues
@app.route('/api/common-ground-issues')
def common_ground_issues():
    try:
        query = """
        WITH Latest AS (
            SELECT Subject, MAX(GeneratedDate) AS LatestGeneratedDate
            FROM CommonGroundIssues
            GROUP BY Subject
        )
        SELECT c.Subject, c.IssueRank, c.Issue, c.Explanation, c.GeneratedDate
        FROM CommonGroundIssues c
        INNER JOIN Latest l ON c.Subject = l.Subject AND c.GeneratedDate = l.LatestGeneratedDate
        ORDER BY c.Subject, c.IssueRank
        """
        return jsonify(run_query(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET: /api/subject-photos
@app.route('/api/subject-photos')
def subject_photos():
    try:
        query = """
        SELECT Subject, PhotoURL, OfficeTitle, State, Party, SourceWebsite
        FROM ElectedOfficialPhotos
        ORDER BY Subject
        """
        return jsonify(run_query(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET: /api/mention-counts
@app.route('/api/mention-counts')
def mention_counts():
    try:
        start = request.args.get('start_date')
        end = request.args.get('end_date')

        where_clause = ""
        if start and end:
            where_clause = f"WHERE CreatedUTC BETWEEN '{start}' AND '{end}'"

        query = f"""
        SELECT Subject, COUNT(*) AS MentionCount FROM (
            SELECT Subject, CreatedUTC FROM RedditCommentsUSSenate {where_clause}
            UNION ALL
            SELECT Subject, CreatedUTC FROM RedditPostsUSSenate {where_clause}
            UNION ALL
            SELECT Subject, CreatedUTC FROM BlueskyCommentsUSSenate {where_clause}
            UNION ALL
            SELECT Subject, CreatedUTC FROM BlueskyPostsUSSenate {where_clause}
        ) AS AllMentions
        GROUP BY Subject
        ORDER BY MentionCount DESC
        """
        return jsonify(run_query(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET: /api/momentum
@app.route('/api/momentum')
def momentum():
    try:
        start = request.args.get('start_date')
        end = request.args.get('end_date')

        where_clause = ""
        if start and end:
            try:
                datetime.strptime(start, "%Y-%m-%d")
                datetime.strptime(end, "%Y-%m-%d")
                where_clause = f"WHERE CreatedUTC BETWEEN '{start}' AND '{end}'"
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        query = f"""
        SELECT
            Subject,
            CAST(CreatedUTC AS DATE) AS ActivityDate,
            COUNT(*) AS MentionCount,
            AVG(SentimentScore) AS AvgSentiment,
            COUNT(*) * AVG(SentimentScore) AS MomentumScore
        FROM (
            SELECT Subject, SentimentScore, CreatedUTC FROM RedditCommentsUSSenate {where_clause}
            UNION ALL
            SELECT Subject, SentimentScore, CreatedUTC FROM BlueskyCommentsUSSenate {where_clause}
        ) AS AllActivity
        GROUP BY Subject, CAST(CreatedUTC AS DATE)
        ORDER BY Subject, ActivityDate
        """
        return jsonify(run_query(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/mention-counts-daily", methods=["GET"])
def mention_counts_daily():
    subject = request.args.get("subject")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if not subject:
        return jsonify({"error": "Missing 'subject' parameter"}), 400
    if not start_date:
        start_date = "2025-01-01"
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except Exception:
        return jsonify({"error": "Invalid date format (use YYYY-MM-DD)"}), 400

    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        query = """
            SELECT
                Subject,
                CAST(CreatedUTC AS DATE) AS [Date],
                COUNT(*) AS MentionCount
            FROM (
                SELECT Subject, CreatedUTC FROM RedditCommentsUSSenate
                UNION ALL
                SELECT Subject, CreatedUTC FROM RedditPostsUSSenate
                UNION ALL
                SELECT Subject, CreatedUTC FROM BlueskyCommentsUSSenate
                UNION ALL
                SELECT Subject, CreatedUTC FROM BlueskyPostsUSSenate
            ) AS combined
            WHERE Subject = ?
              AND CreatedUTC BETWEEN ? AND ?
            GROUP BY Subject, CAST(CreatedUTC AS DATE)
            ORDER BY [Date]
        """

        cursor.execute(query, subject, start, end)
        rows = cursor.fetchall()

        data = [
            {
                "Subject": row.Subject,
                "Date": row.Date.isoformat(),
                "MentionCount": row.MentionCount
            }
            for row in rows
        ]
        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET: /api/latest-comments
@app.route('/api/latest-comments')
def latest_comments():
    """
    Returns the latest N comments (default 10) across Reddit + Bluesky,
    optionally filtered to a single Subject. Columns: Source, Comment, Timestamp, URL.
    """
    subject = request.args.get('subject')
    # sanitize and clamp the limit; TOP can't be parameterized directly in T-SQL
    try:
        limit = int(request.args.get('limit', 10))
    except ValueError:
        limit = 10
    limit = max(1, min(limit, 50))  # 1..50

    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Build query. Parameterize subject; inline TOP for SQL Server.
        query = f"""
            WITH Combined AS (
                SELECT
                    'Reddit' AS Source,
                    CAST(Body AS NVARCHAR(MAX))      AS Comment,
                    CAST(CreatedUTC AS DATETIME)     AS CreatedUTC,
                    CAST(URL AS NVARCHAR(4000))      AS URL,
                    CAST(Subject AS NVARCHAR(255))   AS Subject
                FROM dbo.RedditCommentsUSSenate
                UNION ALL
                SELECT
                    'Bluesky' AS Source,
                    CAST([Text] AS NVARCHAR(MAX))    AS Comment,
                    CAST(CreatedUTC AS DATETIME)     AS CreatedUTC,
                    CAST(URL AS NVARCHAR(4000))      AS URL,
                    CAST(Subject AS NVARCHAR(255))   AS Subject
                FROM dbo.BlueskyCommentsUSSenate
            )
            SELECT TOP {limit}
                Source, Comment, CreatedUTC, URL, Subject
            FROM Combined
            { "WHERE Subject = ?" if subject else "" }
            ORDER BY CreatedUTC DESC;
        """

        params = (subject,) if subject else ()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Format response with desired columns
        def fmt_ts(dt):
            # Try friendly "7/15/24 4:43 PM" on Windows/Linux; fall back to ISO
            try:
                return dt.strftime("%#m/%#d/%y %#I:%M %p")  # Windows
            except Exception:
                try:
                    return dt.strftime("%-m/%-d/%y %-I:%M %p")  # Unix
                except Exception:
                    return dt.isoformat(sep=" ")

        data = [
            {
                "Source": row.Source,
                "Comment": row.Comment,
                "Timestamp": fmt_ts(row.CreatedUTC),
                "URL": row.URL
            }
            for row in rows
        ]

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# App start
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)










