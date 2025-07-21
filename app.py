import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# === Load Scorecard Data ===
scorecard_url = "https://raw.githubusercontent.com/ProblematicGoose/Project-Clout/refs/heads/main/normalized_sentiment.csv"
df_scorecard = pd.read_csv(scorecard_url)

# === Load Time Series Data ===
timeseries_url = "https://raw.githubusercontent.com/ProblematicGoose/Project-Clout/refs/heads/main/normalized_sentiment_by_date.csv"
df_timeseries = pd.read_csv(timeseries_url)

# === Create Scorecard ===
if not df_scorecard.empty:
    subject = df_scorecard['Subject'].iloc[0]
    score = df_scorecard['NormalizedSentimentScore'].iloc[0]

    if score < 400:
        color = 'crimson'
    elif score > 600:
        color = 'green'
    else:
        color = 'orange'

    scorecard = html.Div([
        html.H1(subject),
        html.H2("Sentiment Score"),
        html.H3(f"{score:,}", style={"color": color})
    ], className="card")
else:
    scorecard = html.Div("No data available", style={"textAlign": "center", "fontSize": "24px", "color": "gray", "paddingTop": "100px"})

# === Create Line Chart ===
df_timeseries['SentimentDate'] = pd.to_datetime(df_timeseries['SentimentDate'])
df_timeseries.sort_values(by=['Subject', 'SentimentDate'], inplace=True)

fig = px.line(
    df_timeseries,
    x='SentimentDate',
    y='NormalizedSentimentScore',
    color='Subject',
    markers=True,
    title='Sentiment Over Time',
    labels={
        'SentimentDate': 'Date',
        'NormalizedSentimentScore': 'Sentiment Score'
    }
)

fig.update_layout(
    paper_bgcolor='#1e1e2f',
    plot_bgcolor='#1e1e2f',
    font_color='white',
    title_font_size=20,
    xaxis=dict(showgrid=True, gridcolor='gray'),
    yaxis=dict(showgrid=True, gridcolor='gray', range=[0, 1000])
)

line_chart = html.Div([
    html.H1("Average Normalized Sentiment Over Time", style={"textAlign": "center"}),
    dcc.Graph(figure=fig)
], className="card")

# === Final App Layout ===
app = dash.Dash(__name__)
app.layout = html.Div([
    scorecard,
    line_chart
], style={"backgroundColor": "#1e1e2f", "padding": "20px"})

if __name__ == '__main__':
    app.run_server(debug=True)


