from flask import Flask, jsonify, request, session, redirect
import pyodbc
import pandas as pd
import os
from datetime import datetime
from sqlalchemy import text

"""
Flask API server for dashboard data.
- Includes YouTube across time series, momentum, mention counts, and latest comments.
- Adds optional subject + date filtering to /api/momentum and /api/timeseries for performance.
"""

# -----------------------------
# App + DB
# -----------------------------
flask_app = Flask(__name__)
app = flask_app
flask_app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")

# Local SQL Server connection (adjust if needed)
CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=DESKTOP-VUKR5KV;"
    "DATABASE=Clout;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

def run_query(query: str, params=None):
    """Helper: run ad-hoc SQL and return list[dict]."""
    conn = pyodbc.connect(CONN_STR)
    try:
        cur = conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()

# -----------------------------
# API: Scorecard (stored proc)
# -----------------------------
@flask_app.route("/api/scorecard")
def scorecard():
    try:
        data = run_query("EXEC GetSubjectSentimentScores")
        # Make sure the front-end sees ints
        for row in data:
            if "NormalizedSentimentScore" in row and row["NormalizedSentimentScore"] is not None:
                row["NormalizedSentimentScore"] = int(row["NormalizedSentimentScore"])
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Timeseries (subject/date aware, includes YouTube)
# -----------------------------
@flask_app.route("/api/timeseries")
def timeseries():
    try:
        subject = request.args.get("subject")
        start   = request.args.get("start_date")
        end     = request.args.get("end_date")

        date_where = ""
        if start and end:
            try:
                datetime.strptime(start, "%Y-%m-%d")
                datetime.strptime(end, "%Y-%m-%d")
                date_where = f"WHERE CreatedUTC BETWEEN '{start}' AND '{end}'"
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        subject_where = "WHERE Subject = ?" if subject else ""

        query = f"""
            SELECT
                CAST(SentimentDate AS DATE) AS SentimentDate,
                Subject,
                CAST(ROUND((AVG(SentimentScore) + 1.0) * 5000.0, 0) AS INT) AS NormalizedSentimentScore
            FROM (
                SELECT Subject, SentimentScore, CreatedUTC AS SentimentDate FROM dbo.RedditCommentsUSSenate {date_where}
                UNION ALL
                SELECT Subject, SentimentScore, CreatedUTC AS SentimentDate FROM dbo.BlueskyCommentsUSSenate {date_where}
                UNION ALL
                SELECT Subject, SentimentScore, CreatedUTC AS SentimentDate FROM dbo.YouTubeComments {date_where}
            ) AS Combined
            {subject_where}
            GROUP BY CAST(SentimentDate AS DATE), Subject
            ORDER BY SentimentDate ASC, Subject ASC;
        """
        if subject:
            return jsonify(run_query(query, params=(subject,)))
        return jsonify(run_query(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Traits
# -----------------------------
@flask_app.route("/api/traits")
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

# -----------------------------
# API: Bill sentiment
# -----------------------------
@flask_app.route("/api/bill-sentiment")
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
            ORDER BY AverageSentimentScore DESC;
        """
        return jsonify(run_query(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Weekly top issues
# -----------------------------
@flask_app.route("/api/top-issues")
def top_issues():
    try:
        conn = pyodbc.connect(CONN_STR)
        cur = conn.cursor()
        cur.execute("SELECT MAX(WeekStartDate) FROM WeeklyIdeologyTopics")
        latest_week = cur.fetchone()[0]

        cur.execute("""
            SELECT WeekStartDate, IdeologyLabel, Rank, Topic
            FROM WeeklyIdeologyTopics
            WHERE WeekStartDate = ?
            ORDER BY IdeologyLabel, Rank ASC
        """, (latest_week,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        results = {
            "WeekStartDate": str(latest_week),
            "Liberal": [],
            "Conservative": []
        }
        for r in rows:
            results[r.IdeologyLabel].append({"Rank": r.Rank, "Topic": r.Topic})
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Common ground issues
# -----------------------------
@flask_app.route("/api/common-ground-issues")
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
            INNER JOIN Latest l
                ON c.Subject = l.Subject AND c.GeneratedDate = l.LatestGeneratedDate
            ORDER BY c.Subject, c.IssueRank;
        """
        return jsonify(run_query(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Subject photos
# -----------------------------
@flask_app.route("/api/subject-photos")
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

# -----------------------------
# API: Mention counts (includes YouTube)
# -----------------------------
@flask_app.route("/api/mention-counts")
def mention_counts():
    try:
        start = request.args.get("start_date")
        end   = request.args.get("end_date")

        where_clause = ""
        if start and end:
            where_clause = f"WHERE CreatedUTC BETWEEN '{start}' AND '{end}'"

        query = f"""
            SELECT Subject, COUNT(*) AS MentionCount FROM (
                SELECT Subject, CreatedUTC FROM dbo.RedditCommentsUSSenate {where_clause}
                UNION ALL
                SELECT Subject, CreatedUTC FROM dbo.RedditPostsUSSenate {where_clause}
                UNION ALL
                SELECT Subject, CreatedUTC FROM dbo.BlueskyCommentsUSSenate {where_clause}
                UNION ALL
                SELECT Subject, CreatedUTC FROM dbo.BlueskyPostsUSSenate {where_clause}
                UNION ALL
                SELECT Subject, CreatedUTC FROM dbo.YouTubeComments {where_clause}
            ) AS AllMentions
            GROUP BY Subject
            ORDER BY MentionCount DESC;
        """
        return jsonify(run_query(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Momentum (subject/date aware, includes YouTube)
# -----------------------------
@flask_app.route("/api/momentum")
def momentum():
    try:
        subject = request.args.get("subject")
        start   = request.args.get("start_date")
        end     = request.args.get("end_date")

        where_clause = ""
        if start and end:
            try:
                datetime.strptime(start, "%Y-%m-%d")
                datetime.strptime(end, "%Y-%m-%d")
                where_clause = f"WHERE CreatedUTC BETWEEN '{start}' AND '{end}'"
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        subject_clause = "WHERE Subject = ?" if subject else ""

        query = f"""
            SELECT
                Subject,
                CAST(CreatedUTC AS DATE) AS ActivityDate,
                COUNT(*) AS MentionCount,
                AVG(SentimentScore) AS AvgSentiment
            FROM (
                SELECT Subject, SentimentScore, CreatedUTC FROM dbo.RedditCommentsUSSenate {where_clause}
                UNION ALL
                SELECT Subject, SentimentScore, CreatedUTC FROM dbo.BlueskyCommentsUSSenate {where_clause}
                UNION ALL
                SELECT Subject, SentimentScore, CreatedUTC FROM dbo.YouTubeComments {where_clause}
            ) AS AllActivity
            {subject_clause}
            GROUP BY Subject, CAST(CreatedUTC AS DATE)
            ORDER BY Subject, ActivityDate;
        """
        if subject:
            return jsonify(run_query(query, params=(subject,)))
        return jsonify(run_query(query))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Mention counts by day (subject)
# -----------------------------
@flask_app.route("/api/mention-counts-daily")
def mention_counts_daily():
    subject = request.args.get("subject")
    start   = request.args.get("start_date") or "2025-01-01"
    end     = request.args.get("end_date") or datetime.now().strftime("%Y-%m-%d")

    if not subject:
        return jsonify({"error": "Missing 'subject' parameter"}), 400

    try:
        start_dt = datetime.fromisoformat(start)
        end_dt   = datetime.fromisoformat(end)
    except Exception:
        return jsonify({"error": "Invalid date format (use YYYY-MM-DD)"}), 400

    try:
        conn = pyodbc.connect(CONN_STR)
        cur = conn.cursor()
        query = """
            SELECT
                Subject,
                CAST(CreatedUTC AS DATE) AS [Date],
                COUNT(*) AS MentionCount
            FROM (
                SELECT Subject, CreatedUTC FROM dbo.RedditCommentsUSSenate
                UNION ALL
                SELECT Subject, CreatedUTC FROM dbo.RedditPostsUSSenate
                UNION ALL
                SELECT Subject, CreatedUTC FROM dbo.BlueskyCommentsUSSenate
                UNION ALL
                SELECT Subject, CreatedUTC FROM dbo.BlueskyPostsUSSenate
                UNION ALL
                SELECT Subject, CreatedUTC FROM dbo.YouTubeComments
            ) AS combined
            WHERE Subject = ?
              AND CreatedUTC BETWEEN ? AND ?
            GROUP BY Subject, CAST(CreatedUTC AS DATE)
            ORDER BY [Date];
        """
        cur.execute(query, (subject, start_dt, end_dt))
        rows = cur.fetchall()
        conn.close()

        data = [{"Subject": r.Subject, "Date": r.Date.isoformat(), "MentionCount": r.MentionCount} for r in rows]
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Latest comments (includes YouTube video URL join)
# -----------------------------
@flask_app.route("/api/latest-comments")
def latest_comments():
    subject = request.args.get("subject")
    # clamp limit 1..50
    try:
        limit = int(request.args.get("limit", 10))
    except ValueError:
        limit = 10
    limit = max(1, min(limit, 50))

    try:
        conn = pyodbc.connect(CONN_STR)
        cur = conn.cursor()
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
                UNION ALL
                SELECT
                    'YouTube' AS Source,
                    CAST(yc.[Text] AS NVARCHAR(MAX))   AS Comment,
                    CAST(yc.CreatedUTC AS DATETIME)    AS CreatedUTC,
                    CAST(yv.URL AS NVARCHAR(4000))     AS URL,
                    CAST(yc.Subject AS NVARCHAR(255))  AS Subject
                FROM dbo.YouTubeComments yc
                LEFT JOIN dbo.YouTubeVideos yv ON yv.VideoID = yc.VideoID
            )
            SELECT TOP {limit}
                Source, Comment, CreatedUTC, URL, Subject
            FROM Combined
            { "WHERE Subject = ?" if subject else "" }
            ORDER BY CreatedUTC DESC;
        """
        params = (subject,) if subject else ()
        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()

        def fmt_ts(dt):
            try:
                return dt.strftime("%#m/%#d/%y %#I:%M %p")  # Windows
            except Exception:
                try:
                    return dt.strftime("%-m/%-d/%y %-I:%M %p")  # Unix
                except Exception:
                    return dt.isoformat(sep=" ")

        data = [
            {"Source": r.Source, "Comment": r.Comment, "Timestamp": fmt_ts(r.CreatedUTC), "URL": r.URL}
            for r in rows
        ]
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route("/api/subjects")
def subjects():
    """
    Returns a canonical list of subjects for the UI dropdown.
    Uses ElectedOfficialPhotos as the source of truth.
    """
    try:
        rows = run_query("""
            SELECT DISTINCT Subject
            FROM ElectedOfficialPhotos
            WHERE Subject IS NOT NULL
            ORDER BY Subject
        """)
        # rows like [{'Subject': 'Ted Cruz'}, ...]
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Main (only if you run this module directly)
# -----------------------------
if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=5050, debug=False)











