import dash
from dash import dcc, html, Input, Output
import pandas as pd
import requests
import plotly.express as px

# 🔁 Replace with your live ngrok URL
API_BASE = "https://239cbef66c54.ngrok-free.app"

# Preload subjects
all_traits = requests.get(f"{API_BASE}/api/traits").json()
subjects = sorted(list({row["Subject"] for row in all_traits}))

app = dash.Dash(__name__)
app.title = "Sentiment Dashboard"

app.layout = html.Div([
    html.Div([
        html.Label("Select a Subject:", style={"color": "white", "fontSize": "18px", "marginRight": "10px"}),
        dcc.Dropdown(
            id='subject-dropdown',
            options=[{"label": s, "value": s} for s in subjects],
            value=subjects[0],
            style={"width": "300px", "color": "#000"}
        )
    ], style={"padding": "20px"}),

    html.Div(id='scorecard-output', className="card"),
    html.Div(id='linechart-output', className="card"),
    html.Div(id='traits-output', className="card")

], style={"backgroundColor": "#1e1e2f", "padding": "20px"})


@app.callback(
    Output('scorecard-output', 'children'),
    Output('linechart-output', 'children'),
    Output('traits-output', 'children'),
    Input('subject-dropdown', 'value')
)
def update_visuals(subject):
    # === Scorecard ===
    score_resp = requests.get(f"{API_BASE}/api/scorecard")
    df_score = pd.DataFrame(score_resp.json())
    df_subject = df_score[df_score['Subject'] == subject]

    if not df_subject.empty:
        score = df_subject['NormalizedSentimentScore'].iloc[0]
        color = 'orange'
        if score < 400:
            color = 'crimson'
        elif score > 600:
            color = 'green'
        scorecard = html.Div([
            html.H1(subject),
            html.H2("Sentiment Score"),
            html.H3(f"{score:,}", style={"color": color})
        ])
    else:
        scorecard = html.Div("No data for selected subject")

    # === Line Chart ===
    ts_resp = requests.get(f"{API_BASE}/api/timeseries")
    df_ts = pd.DataFrame(ts_resp.json())
    df_ts['SentimentDate'] = pd.to_datetime(df_ts['SentimentDate'])
    df_ts = df_ts[df_ts['Subject'] == subject].sort_values('SentimentDate')

    fig = px.line(
        df_ts,
        x='SentimentDate',
        y='NormalizedSentimentScore',
        title='Sentiment Over Time',
        markers=True
    )
    fig.update_layout(
        paper_bgcolor='#1e1e2f',
        plot_bgcolor='#1e1e2f',
        font_color='white',
        title_font_size=20,
        xaxis=dict(showgrid=True, gridcolor='gray'),
        yaxis=dict(showgrid=True, gridcolor='gray', range=[0, 1000])
    )
    linechart = html.Div([
        html.H1("Average Normalized Sentiment Over Time", style={"textAlign": "center"}),
        dcc.Graph(figure=fig)
    ])

    # === Traits ===
    traits_resp = requests.get(f"{API_BASE}/api/traits")
    df_traits = pd.DataFrame(traits_resp.json())
    df_traits = df_traits[df_traits['Subject'] == subject]

    pos_traits = df_traits[df_traits['TraitType'] == 'Positive'].sort_values('TraitRank')['TraitDescription'].tolist()
    neg_traits = df_traits[df_traits['TraitType'] == 'Negative'].sort_values('TraitRank')['TraitDescription'].tolist()

    def make_list(traits, color):
        return html.Div([
            html.Div(
                f"{i+1}. {trait}",
                style={"fontSize": "20px", "color": color, "textAlign": "left", "marginBottom": "6px"}
            )
            for i, trait in enumerate(traits)
        ])

    traits = html.Div([
        html.H1("Top Traits Summary", style={"textAlign": "center"}),
        html.H2("People like it when I...", style={"color": "green", "fontWeight": "bold", "textAlign": "left"}),
        make_list(pos_traits, "lightgreen"),
        html.H2("People don’t like it when I...", style={"color": "crimson", "marginTop": "40px", "fontWeight": "bold", "textAlign": "left"}),
        make_list(neg_traits, "lightcoral")
    ])

    return scorecard, linechart, traits


if __name__ == '__main__':
    app.run_server(debug=True)

