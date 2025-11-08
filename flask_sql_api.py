from flask import Flask, jsonify, request
import os
import urllib.parse
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

"""
Flask API server for dashboard data.
- Connection pooling via SQLAlchemy engine
- Named, parameterized SQL (sqlalchemy.text)
- Uses CreatedDate (persisted) instead of CAST(CreatedUTC AS DATE)
- Uses SubjectDailyActivity / SubjectDailySentiment where appropriate for speed
"""

# -----------------------------
# App + DB
# -----------------------------
flask_app = Flask(__name__)
app = flask_app
flask_app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")

# Local SQL Server connection
CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=DESKTOP-VUKR5KV;"
    "DATABASE=Clout;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

ENGINE = create_engine(
    "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(CONN_STR),
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800,
)

def run_query(sql: str, params=None):
    """Execute SQL and return list[dict] with connection pooling."""
    with ENGINE.begin() as conn:
        rows = conn.execute(text(sql), params or {})
        return [dict(r._mapping) for r in rows]

# -----------------------------
# Helpers
# -----------------------------
def parse_date_or_400(value: str, name: str):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        raise ValueError(f"Invalid {name} format. Use YYYY-MM-DD.")

# -----------------------------
# API: Scorecard (stored proc)
# -----------------------------

# -----------------------------
# API: Subject bundle (one call per subject)
# -----------------------------


@flask_app.route("/api/scorecard")
def scorecard():
    try:
        data = run_query("EXEC GetSubjectSentimentScores")
        for row in data:
            if "NormalizedSentimentScore" in row and row["NormalizedSentimentScore"] is not None:
                row["NormalizedSentimentScore"] = int(row["NormalizedSentimentScore"])
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Timeseries (reads rollup)
# -----------------------------
@flask_app.route("/api/timeseries")
def timeseries():
    try:
        subject = request.args.get("subject")
        start = request.args.get("start_date")
        end = request.args.get("end_date")

        params = {}
        where_parts = []

        if start and end:
            start_d = parse_date_or_400(start, "start_date")
            end_d = parse_date_or_400(end, "end_date")
            where_parts.append("CreatedDate BETWEEN :start_d AND :end_d")
            params.update({"start_d": start_d, "end_d": end_d})

        if subject:
            where_parts.append("Subject = :subject")
            params["subject"] = subject

        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        sql = f"""
            SELECT
                CreatedDate AS SentimentDate,
                Subject,
                CAST(ROUND((ISNULL(AvgSentiment, 0) + 1.0) * 5000.0, 0) AS INT) AS NormalizedSentimentScore
            FROM dbo.SubjectDailySentiment
            {where_sql}
            ORDER BY SentimentDate ASC, Subject ASC;
        """
        return jsonify(run_query(sql, params))
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Traits
# -----------------------------
@flask_app.route("/api/traits")
def traits():
    try:
        sql = """
            SELECT Subject, TraitType, TraitRank, TraitDescription, CreatedUTC
            FROM SubjectTraitSummary
            ORDER BY Subject, TraitType, CreatedUTC;
        """
        return jsonify(run_query(sql))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Bill sentiment
# -----------------------------
@flask_app.route("/api/bill-sentiment")
def bill_sentiment():
    try:
        sql = """
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
        return jsonify(run_query(sql))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Weekly top issues
# -----------------------------
@flask_app.route("/api/top-issues")
def top_issues():
    try:
        with ENGINE.begin() as conn:
            latest_week = conn.execute(text("SELECT MAX(WeekStartDate) FROM WeeklyIdeologyTopics")).scalar()
            rows = conn.execute(
                text(
                    """
                    SELECT WeekStartDate, IdeologyLabel, Rank, Topic
                    FROM WeeklyIdeologyTopics
                    WHERE WeekStartDate = :wk
                    ORDER BY IdeologyLabel, Rank ASC;
                    """
                ),
                {"wk": latest_week},
            ).fetchall()

        results = {"WeekStartDate": str(latest_week), "Liberal": [], "Conservative": []}
        for r in rows:
            results[r[1]].append({"Rank": r[2], "Topic": r[3]})
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Common ground issues
# -----------------------------
@flask_app.route("/api/common-ground-issues")
def common_ground_issues():
    try:
        sql = """
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
        return jsonify(run_query(sql))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Subject photos
# -----------------------------
@flask_app.route("/api/subject-photos")
def subject_photos():
    try:
        sql = """
            SELECT Subject, PhotoURL, OfficeTitle, State, Party, SourceWebsite
            FROM ElectedOfficialPhotos
            ORDER BY Subject;
        """
        return jsonify(run_query(sql))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Mention counts (TOTAL by subject; rollup)
# -----------------------------
@flask_app.route("/api/mention-counts")
def mention_counts():
    try:
        start = request.args.get("start_date")
        end = request.args.get("end_date")

        params = {}
        where_date = ""
        if start and end:
            start_d = parse_date_or_400(start, "start_date")
            end_d = parse_date_or_400(end, "end_date")
            where_date = "WHERE CreatedDate BETWEEN :start_d AND :end_d"
            params.update({"start_d": start_d, "end_d": end_d})

        sql = f"""
            SELECT Subject, SUM(MentionCount) AS MentionCount
            FROM dbo.SubjectDailyActivity
            {where_date}
            GROUP BY Subject
            ORDER BY MentionCount DESC;
        """
        return jsonify(run_query(sql, params))
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Momentum (reads rollup)
# -----------------------------
@flask_app.route("/api/momentum")
def momentum():
    try:
        subject = request.args.get("subject")
        start = request.args.get("start_date")
        end = request.args.get("end_date")

        params = {}
        where_parts = []

        if start and end:
            start_d = parse_date_or_400(start, "start_date")
            end_d = parse_date_or_400(end, "end_date")
            where_parts.append("CreatedDate BETWEEN :start_d AND :end_d")
            params.update({"start_d": start_d, "end_d": end_d})

        if subject:
            where_parts.append("Subject = :subject")
            params["subject"] = subject

        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        sql = f"""
            SELECT
                Subject,
                CreatedDate AS ActivityDate,
                MentionCount,
                AvgSentiment
            FROM dbo.SubjectDailySentiment
            {where_sql}
            ORDER BY Subject, ActivityDate;
        """
        return jsonify(run_query(sql, params))
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Mention counts by day (per subject; rollup)
# -----------------------------
@flask_app.route("/api/mention-counts-daily")
def mention_counts_daily():
    subject = request.args.get("subject")
    start = request.args.get("start_date") or "2025-01-01"
    end = request.args.get("end_date") or datetime.now().strftime("%Y-%m-%d")

    if not subject:
        return jsonify({"error": "Missing 'subject' parameter"}), 400

    try:
        start_d = parse_date_or_400(start, "start_date")
        end_d = parse_date_or_400(end, "end_date")
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    sql = """
        SELECT Subject, CreatedDate AS [Date], MentionCount
        FROM dbo.SubjectDailyActivity
        WHERE Subject = :subject
          AND CreatedDate BETWEEN :start_d AND :end_d
        ORDER BY [Date];
    """
    rows = run_query(sql, {"subject": subject, "start_d": start_d, "end_d": end_d})
    data = [
        {
            "Subject": r["Subject"],
            "Date": r["Date"].isoformat(),
            "MentionCount": int(r["MentionCount"]),
        }
        for r in rows
    ]
    return jsonify(data)

# -----------------------------
# API: Latest comments (includes YouTube video URL join)
# -----------------------------
@flask_app.route("/api/latest-comments")
def latest_comments():
    subject = request.args.get("subject")

    try:
        limit = int(request.args.get("limit", 10))
    except ValueError:
        limit = 10
    limit = max(1, min(limit, 50))

    params = {}
    where_subject = ""
    if subject:
        where_subject = "WHERE Subject = :subject"
        params["subject"] = subject

    sql = f"""
        WITH Combined AS (
            SELECT 'Reddit' AS Source,
                   CAST(Body AS NVARCHAR(MAX)) AS Comment,
                   CAST(CreatedUTC AS DATETIME) AS CreatedUTC,
                   CAST(URL AS NVARCHAR(4000)) AS URL,
                   CAST(Subject AS NVARCHAR(255)) AS Subject
            FROM dbo.RedditCommentsUSSenate
            UNION ALL
            SELECT 'Bluesky',
                   CAST([Text] AS NVARCHAR(MAX)),
                   CAST(CreatedUTC AS DATETIME),
                   CAST(URL AS NVARCHAR(4000)),
                   CAST(Subject AS NVARCHAR(255))
            FROM dbo.BlueskyCommentsUSSenate
            UNION ALL
            SELECT 'YouTube',
                   CAST(yc.[Text] AS NVARCHAR(MAX)),
                   CAST(yc.CreatedUTC AS DATETIME),
                   CAST(yv.URL AS NVARCHAR(4000)),
                   CAST(yc.Subject AS NVARCHAR(255))
            FROM dbo.YouTubeComments yc
            LEFT JOIN dbo.YouTubeVideos yv ON yv.VideoID = yc.VideoID
        )
        SELECT TOP {limit}
            Source, Comment, CreatedUTC, URL, Subject
        FROM Combined
        {where_subject}
        ORDER BY CreatedUTC DESC;
    """
    rows = run_query(sql, params)
    data = [
        {
            "Source": r["Source"],
            "Comment": r["Comment"],
            "CreatedUTC": r["CreatedUTC"].isoformat(),
            "URL": r["URL"],
            "Subject": r["Subject"],
        }
        for r in rows
    ]
    return jsonify(data)


# -----------------------------
# API: Constituent Asks (Top-N per subject)
# -----------------------------
@flask_app.route("/api/constituent-asks")
def constituent_asks():
    """
    Query params:
      - subjects: optional comma-separated list (e.g., ?subjects=Ted%20Cruz,Bernie%20Sanders)
      - top_n: optional, default 5 (1..10)
      - latest: 1 (default) or 0; if 0, provide week_end=YYYY-MM-DD
      - week_end: optional, used when latest=0
    """
    try:
        subjects_param = request.args.get("subjects", "").strip()
        subjects_list = [s.strip() for s in subjects_param.split(",") if s.strip()] if subjects_param else []

        try:
            top_n = int(request.args.get("top_n", 5))
        except ValueError:
            top_n = 5
        top_n = max(1, min(top_n, 10))

        latest_flag = request.args.get("latest", "1").strip()
        use_latest = (latest_flag != "0")

        in_clause = ""
        params = {"top_n": top_n}
        if subjects_list:
            bind_names = [f"s{i}" for i in range(len(subjects_list))]
            in_clause = " AND a.Subject IN (" + ", ".join(f":{b}" for b in bind_names) + ") "
            params.update({b: subjects_list[i] for i, b in enumerate(bind_names)})

        if use_latest:
            sql = f"""
                WITH Ranked AS (
                    SELECT
                        a.Subject, a.Ask, a.SupportCount, a.Confidence,
                        a.WeekStartUTC, a.WeekEndUTC,
                        ROW_NUMBER() OVER (
                            PARTITION BY a.Subject
                            ORDER BY a.SupportCount DESC, a.Confidence DESC, a.Id DESC
                        ) AS rn
                    FROM dbo.ConstituentAsksLatest7Days a WITH (NOLOCK)
                    WHERE a.WeekEndUTC = (SELECT MAX(WeekEndUTC) FROM dbo.ConstituentAsksLatest7Days)
                    {in_clause}
                )
                SELECT Subject, Ask, SupportCount, Confidence, WeekStartUTC, WeekEndUTC
                FROM Ranked
                WHERE rn <= :top_n
                ORDER BY Subject, rn;
            """
            return jsonify(run_query(sql, params))

        week_end_str = request.args.get("week_end")
        if not week_end_str:
            return jsonify({"error": "When latest=0, you must provide week_end=YYYY-MM-DD"}), 400

        we_date = parse_date_or_400(week_end_str, "week_end")
        start_dt = datetime.combine(we_date, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)

        params.update({"start_dt": start_dt, "end_dt": end_dt})
        sql = f"""
            WITH Ranked AS (
                SELECT
                    a.Subject, a.Ask, a.SupportCount, a.Confidence,
                    a.WeekStartUTC, a.WeekEndUTC,
                    ROW_NUMBER() OVER (
                        PARTITION BY a.Subject
                        ORDER BY a.SupportCount DESC, a.Confidence DESC, a.Id DESC
                    ) AS rn
                FROM dbo.ConstituentAsksLatest7Days a WITH (NOLOCK)
                WHERE a.WeekEndUTC >= :start_dt AND a.WeekEndUTC < :end_dt
                {in_clause}
            )
            SELECT Subject, Ask, SupportCount, Confidence, WeekStartUTC, WeekEndUTC
            FROM Ranked
            WHERE rn <= :top_n
            ORDER BY Subject, rn;
        """
        return jsonify(run_query(sql, params))
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Weekly Strategy
# -----------------------------
@flask_app.route("/api/weekly-strategy")
def weekly_strategy():
    """
    Query params:
      - subjects: optional comma-separated list
      - latest: 1 (default) or 0; if 0, provide week_end=YYYY-MM-DD
      - week_end: used when latest=0
    """
    try:
        subjects_param = request.args.get("subjects", "").strip()
        subjects_list = [s.strip() for s in subjects_param.split(",") if s.strip()] if subjects_param else []

        latest_flag = request.args.get("latest", "1").strip()
        use_latest = (latest_flag != "0")

        in_clause = ""
        params = {}
        if subjects_list:
            bind_names = [f"s{i}" for i in range(len(subjects_list))]
            in_clause = "WHERE s.Subject IN (" + ", ".join(f":{b}" for b in bind_names) + ")"
            params.update({b: subjects_list[i] for i, b in enumerate(bind_names)})

        if use_latest:
            sql = f"""
                SELECT
                    COALESCE(w.Subject, s.Subject) AS Subject,
                    w.StrategySummary,
                    w.StrategyStatement,
                    w.Rationale,
                    w.SupportCount,
                    w.Confidence,
                    w.ActionabilityScore,
                    w.WeekStartUTC,
                    w.WeekEndUTC
                FROM (SELECT DISTINCT Subject FROM dbo.WeeklySubjectStrategy WITH (NOLOCK)) AS s
                OUTER APPLY (
                    SELECT TOP (1) *
                    FROM dbo.WeeklySubjectStrategy w WITH (NOLOCK)
                    WHERE w.Subject = s.Subject
                    ORDER BY w.WeekEndUTC DESC, w.Id DESC
                ) AS w
                {in_clause}
                ORDER BY s.Subject;
            """
            return jsonify(run_query(sql, params))

        week_end_str = request.args.get("week_end")
        if not week_end_str:
            return jsonify({"error": "When latest=0, provide week_end=YYYY-MM-DD"}), 400

        we_date = parse_date_or_400(week_end_str, "week_end")
        start_dt = datetime.combine(we_date, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)

        params.update({"start_dt": start_dt, "end_dt": end_dt})
        in_clause_specific = ""
        if subjects_list:
            bind_names = [f"s{i}" for i in range(len(subjects_list))]
            in_clause_specific = " AND s.Subject IN (" + ", ".join(f":{b}" for b in bind_names) + ")"
            params.update({b: subjects_list[i] for i, b in enumerate(bind_names)})

        sql = f"""
            SELECT
                COALESCE(w.Subject, s.Subject) AS Subject,
                w.StrategySummary,
                w.StrategyStatement,
                w.Rationale,
                w.SupportCount,
                w.Confidence,
                w.ActionabilityScore,
                w.WeekStartUTC,
                w.WeekEndUTC
            FROM (SELECT DISTINCT Subject FROM dbo.WeeklySubjectStrategy WITH (NOLOCK)) AS s
            OUTER APPLY (
                SELECT TOP (1) *
                FROM dbo.WeeklySubjectStrategy w WITH (NOLOCK)
                WHERE w.Subject = s.Subject
                  AND w.WeekEndUTC >= :start_dt AND w.WeekEndUTC < :end_dt
                ORDER BY w.WeekEndUTC DESC, w.Id DESC
            ) AS w
            WHERE 1=1
            {in_clause_specific}
            ORDER BY s.Subject;
        """
        return jsonify(run_query(sql, params))
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: Subject bundle (one call per subject)
# -----------------------------
@flask_app.route("/api/subject-bundle")
def subject_bundle():
    try:
        subject = request.args.get("subject")
        start   = request.args.get("start_date")
        end     = request.args.get("end_date")
        if not subject:
            return jsonify({"error": "subject required"}), 400

        # optional date window
        where_dates = ""
        params = {"subject": subject}
        if start and end:
            try:
                start_d = datetime.strptime(start, "%Y-%m-%d").date()
                end_d   = datetime.strptime(end,   "%Y-%m-%d").date()
            except Exception:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
            where_dates = " AND CreatedDate BETWEEN :start_d AND :end_d "
            params.update({"start_d": start_d, "end_d": end_d})

        with ENGINE.begin() as conn:
            # Timeseries (normalized score) from rollup
            ts_rows = conn.execute(text(f"""
                SELECT CreatedDate,
                       CAST(ROUND((ISNULL(AvgSentiment,0)+1.0)*5000.0,0) AS INT) AS NormalizedSentimentScore
                FROM dbo.SubjectDailySentiment
                WHERE Subject=:subject {where_dates}
                ORDER BY CreatedDate
            """), params).fetchall()

            # Momentum (mentions + avg sentiment) from rollup
            mom_rows = conn.execute(text(f"""
                SELECT CreatedDate, MentionCount, AvgSentiment
                FROM dbo.SubjectDailySentiment
                WHERE Subject=:subject {where_dates}
                ORDER BY CreatedDate
            """), params).fetchall()

            # Latest weekly strategy
            strat = conn.execute(text("""
                SELECT TOP (1) StrategySummary, StrategyStatement, Rationale,
                               SupportCount, Confidence, ActionabilityScore,
                               WeekStartUTC, WeekEndUTC
                FROM dbo.WeeklySubjectStrategy
                WHERE Subject = :subject
                ORDER BY WeekEndUTC DESC, Id DESC
            """), {"subject": subject}).fetchone()

            # Top 5 asks (latest window)
            asks = conn.execute(text("""
                WITH Ranked AS (
                  SELECT a.Ask, a.SupportCount, a.Confidence, a.WeekStartUTC, a.WeekEndUTC,
                         ROW_NUMBER() OVER (
                           PARTITION BY a.Subject
                           ORDER BY a.SupportCount DESC, a.Confidence DESC, a.Id DESC
                         ) rn
                  FROM dbo.ConstituentAsksLatest7Days a
                  WHERE a.Subject=:subject
                    AND a.WeekEndUTC = (SELECT MAX(WeekEndUTC) FROM dbo.ConstituentAsksLatest7Days)
                )
                SELECT Ask, SupportCount, Confidence, WeekStartUTC, WeekEndUTC
                FROM Ranked WHERE rn <= 5
                ORDER BY rn
            """), {"subject": subject}).fetchall()

            # Photos (up to 3)
            photos = conn.execute(text("""
                SELECT TOP 3 PhotoURL, OfficeTitle, State, Party, SourceWebsite
                FROM dbo.ElectedOfficialPhotos
                WHERE Subject=:subject
                ORDER BY PhotoURL
            """), {"subject": subject}).fetchall()

            # Latest comments (trim to 5)
            comments = conn.execute(text("""
                WITH C AS (
                  SELECT 'Reddit'  AS Source, CAST(Body AS NVARCHAR(MAX))   Comment, CreatedUTC, URL
                  FROM dbo.RedditCommentsUSSenate WHERE Subject=:subject
                  UNION ALL
                  SELECT 'Bluesky', CAST([Text] AS NVARCHAR(MAX)), CreatedUTC, URL
                  FROM dbo.BlueskyCommentsUSSenate WHERE Subject=:subject
                  UNION ALL
                  SELECT 'YouTube', CAST(yc.[Text] AS NVARCHAR(MAX)), yc.CreatedUTC, yv.URL
                  FROM dbo.YouTubeComments yc
                  LEFT JOIN dbo.YouTubeVideos yv ON yv.VideoID = yc.VideoID
                  WHERE yc.Subject=:subject
                )
                SELECT TOP 5 Source, Comment, CreatedUTC, URL
                FROM C ORDER BY CreatedUTC DESC
            """), {"subject": subject}).fetchall()

        # Shape JSON safely
        out = {
            "timeseries": [
                {"Date": (r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0])),
                 "Score": int(r[1])}
                for r in ts_rows
            ],
            "momentum": [
                {"Date": (r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0])),
                 "MentionCount": int(r[1] or 0),
                 "AvgSentiment": float(r[2] or 0)}
                for r in mom_rows
            ],
            "strategy": (dict(strat._mapping) if strat else None),
            "asks":     [dict(a._mapping) for a in asks],
            "photos":   [dict(p._mapping) for p in photos],
            "latest_comments": [
                {"Source": c[0], "Comment": c[1],
                 "CreatedUTC": (c[2].isoformat() if c[2] and hasattr(c[2], "isoformat") else str(c[2])),
                 "URL": c[3]}
                for c in comments
            ],
        }
        return jsonify(out)
    except Exception as e:
        # Return the error so we see exactly what's failing
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

# -----------------------------
# API: Subjects list for UI dropdown
# -----------------------------
@flask_app.route("/api/subjects")
def subjects():
    try:
        with ENGINE.begin() as conn:
            rows = conn.execute(text("""
                WITH AllSubjects AS (
                    SELECT DISTINCT CAST(Subject AS NVARCHAR(255)) AS Subject
                    FROM dbo.ElectedOfficialPhotos WITH (NOLOCK)
                    WHERE Subject IS NOT NULL AND LTRIM(RTRIM(Subject)) <> ''

                    UNION

                    SELECT DISTINCT CAST(Subject AS NVARCHAR(255)) AS Subject
                    FROM dbo.WeeklySubjectStrategy WITH (NOLOCK)
                    WHERE Subject IS NOT NULL AND LTRIM(RTRIM(Subject)) <> ''

                    UNION

                    SELECT DISTINCT CAST(Subject AS NVARCHAR(255)) AS Subject
                    FROM dbo.ConstituentAsksLatest7Days WITH (NOLOCK)
                    WHERE Subject IS NOT NULL AND LTRIM(RTRIM(Subject)) <> ''
                )
                SELECT Subject
                FROM AllSubjects
                WHERE Subject IS NOT NULL AND LTRIM(RTRIM(Subject)) <> ''
                ORDER BY Subject;
            """)).fetchall()

        # jsonify list-of-dicts: [{ "Subject": "Ted Cruz" }, ...]
        return jsonify([{"Subject": r[0]} for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=5050, debug=False)














