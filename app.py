import dash
from dash import html, dcc
import pandas as pd
import requests
import plotly.express as px

# === API Base URL ===
API_BASE = "https://82a49d5841a9.ngrok-free.app"

# === Get Scorecard Data ===
scorecard_resp = requests.get(f"{API_BASE}/api/scorecard")
df_scorecard = pd.DataFrame(scorecard_resp.json())

# === Get Time Series Data ===
timeseries_resp = requests.get(f"{API_BASE}/api/timeseries")
df_timeseries = pd.DataFrame(timeseries_resp.json())

# === Get Trait Data ===
traits_resp = requests.get(f"{API_BASE}/api/traits")
df_traits = pd.DataFrame(traits_resp.json())

# === Scorecard Component ===
if not df_scorecard.empty:
    subject = df_scorecard['Subject'].iloc[0]
    score = df_scorecard['NormalizedSentimentScore'].iloc[0]

    color = 'orange'
    if score < 400:
        color = 'crimson'
    elif score > 600:
        color = 'green'

    scorecard = html.Div([
        html.H1(subject),
        html.H2("Sentiment Score"),
        html.H3(f"{score:,}", style={"color": color})
    ], className="card")
else:
    scorecard = html.Div("No scorecard data available", className="card")

# === Time Series Line Chart ===
df_timeseries['SentimentDate'] = pd.to_datetime(df_timeseries['SentimentDate'])
df_timeseries.sort_values(by=['Subject', 'SentimentDate'], inplace=True)

fig = px.line(
    df_timeseries,
    x='SentimentDate',
    y='NormalizedSentimentScore',
    color='Subject',
    markers=True,
    title='Sentiment Over Time'
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

# === Trait Summary Component ===
if not df_traits.empty:
    subject = df_traits['Subject'].iloc[0]
    pos_traits = df_traits[df_traits['TraitType'] == 'Positive'].sort_values('TraitRank')['TraitDescription'].tolist()
    neg_traits = df_traits[df_traits['TraitType'] == 'Negative'].sort_values('TraitRank')['TraitDescription'].tolist()

    def make_list(traits, color):
        return html.Ul([
            html.Li(f"{i+1}. {trait}", style={"fontSize": "20px", "color": color}) for i, trait in enumerate(traits)
        ], style={"margin": "0", "paddingLeft": "20px"})

    trait_block = html.Div([
        html.H1("Top Traits Summary", style={"textAlign": "center"}),
        html.H2("People like it when I...", style={"color": "green", "fontWeight": "bold"}),
        make_list(pos_traits, "lightgreen"),
        html.H2("People don’t like it when I...", style={"color": "crimson", "marginTop": "40px", "fontWeight": "bold"}),
        make_list(neg_traits, "lightcoral")
    ], className="card")
else:
    trait_block = html.Div("No trait data available.", className="card")

# === Final Layout ===
app = dash.Dash(__name__)
app.layout = html.Div([
    scorecard,
    line_chart,
    trait_block
], style={"backgroundColor": "#1e1e2f", "padding": "20px"})

if __name__ == '__main__':
    app.run_server(debug=True)




