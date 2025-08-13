import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import urllib.request
import json

# URLs for live data
SCORECARD_URL = "https://4c2f4bd11503.ngrok-free.app/api/scorecard"
TIMESERIES_URL = "https://4c2f4bd11503.ngrok-free.app/api/timeseries"
TRAITS_URL = "https://4c2f4bd11503.ngrok-free.app/api/traits"
BILL_SENTIMENT_URL = "https://4c2f4bd11503.ngrok-free.app/api/bill-sentiment"
TOP_ISSUES_URL = "https://4c2f4bd11503.ngrok-free.app/api/top-issues"

app = dash.Dash(__name__)
server = app.server

with urllib.request.urlopen(SCORECARD_URL) as url:
    subjects_data = json.load(url)
subjects = sorted({entry["Subject"].strip() for entry in subjects_data if entry["Subject"] is not None})

app.layout = html.Div([
    html.Div([
        html.H1("Sentiment Dashboard", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='subject-dropdown',
            options=[{'label': subj, 'value': subj} for subj in subjects],
            value=next(iter(subjects)),
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'marginBottom': '40px'}),

    html.Div(id='scorecard-div', className='card'),
    html.Div([dcc.Graph(id='timeseries-graph')], className='card'),
    html.Div(id='traits-div', className='card'),
    html.Div(id='bill-sentiment-table', className='card', style={'marginTop': '40px'}),
    html.Div(id='top-issues-div', className='card', style={'marginTop': '40px'})
])

@app.callback(
    Output('scorecard-div', 'children'),
    Output('timeseries-graph', 'figure'),
    Output('traits-div', 'children'),
    Output('bill-sentiment-table', 'children'),
    Output('top-issues-div', 'children'),
    Input('subject-dropdown', 'value')
)
def update_dashboard(selected_subject):
    with urllib.request.urlopen(SCORECARD_URL) as url:
        scorecard_data = json.load(url)
    df_scorecard = pd.DataFrame(scorecard_data)
    score_row = df_scorecard[df_scorecard['Subject'] == selected_subject]
    score = int(score_row['NormalizedSentimentScore'].iloc[0]) if not score_row.empty else 5000
    color = 'green' if score > 6000 else 'crimson' if score < 4000 else 'orange'

    scorecard_display = html.Div([
        html.H1(selected_subject, style={'fontSize': '50px', 'fontWeight': 'bold'}),
        html.Div("Sentiment Score", style={'fontSize': '30px', 'color': 'gray'}),
        html.Div(f"{score:,}", style={'fontSize': '80px', 'color': color})
    ])

    with urllib.request.urlopen(TIMESERIES_URL) as url:
        timeseries_data = json.load(url)
    df_timeseries = pd.DataFrame(timeseries_data)
    df_timeseries['SentimentDate'] = pd.to_datetime(df_timeseries['SentimentDate'])
    df_filtered = df_timeseries[df_timeseries['Subject'] == selected_subject]

    timeseries_fig = go.Figure()
    timeseries_fig.add_trace(go.Scatter(
        x=df_filtered['SentimentDate'],
        y=df_filtered['NormalizedSentimentScore'],
        mode='lines+markers',
        name=selected_subject,
        line=dict(color='royalblue')
    ))
    timeseries_fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Normalized Sentiment Score (0–10,000)",
        yaxis=dict(range=[0, 10000]),
        template='plotly_white'
    )

    with urllib.request.urlopen(TRAITS_URL) as url:
        traits_data = json.load(url)
    df_traits = pd.DataFrame(traits_data)
    df_traits = df_traits[df_traits['Subject'] == selected_subject]
    positive = df_traits[df_traits['TraitType'] == 'Positive'].sort_values('TraitRank')['TraitDescription'].tolist()
    negative = df_traits[df_traits['TraitType'] == 'Negative'].sort_values('TraitRank')['TraitDescription'].tolist()

    trait_display = html.Div([
        html.H2("People like it when I...", style={'color': 'green'}),
        html.Ul([html.Li(trait, style={'textAlign': 'left'}) for trait in positive], style={'listStyleType': 'none'}),
        html.H2("People don't like it when I...", style={'color': 'crimson', 'marginTop': '30px'}),
        html.Ul([html.Li(trait, style={'textAlign': 'left'}) for trait in negative], style={'listStyleType': 'none'})
    ])

    with urllib.request.urlopen(BILL_SENTIMENT_URL) as url:
        bill_data = json.load(url)
    df_bills = pd.DataFrame(bill_data)
    bill_table = html.Div([
        html.H2("Public Sentiment Toward National Bills", style={'textAlign': 'center'}),
        html.Table([
            html.Thead([html.Tr([html.Th("Bill"), html.Th("Score"), html.Th("Public Sentiment")])]),
            html.Tbody([
                html.Tr([
                    html.Td(row['BillName']),
                    html.Td(round(row['AverageSentimentScore'], 2), style={'paddingRight': '30px'}),
                    html.Td(row['SentimentLabel'])
                ]) for _, row in df_bills.iterrows()
            ])
        ], style={'width': '80%', 'margin': '0 auto', 'borderCollapse': 'collapse', 'textAlign': 'left'})
    ])

    # === TOP ISSUES BY IDEOLOGY ===
    try:
        with urllib.request.urlopen(TOP_ISSUES_URL) as url:
            issues_data = json.load(url)
        week = issues_data['WeekStartDate']
        conservative_issues = issues_data['Conservative']
        liberal_issues = issues_data['Liberal']

        issues_display = html.Div([
            html.H2(f"Top Issues for the Week of {week}", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.H3("Conservative Topics", style={'color': 'crimson'}),
                    html.Ul([html.Li(f"{item['Rank']}. {item['Topic']}") for item in conservative_issues])
                ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                html.Div([
                    html.H3("Liberal Topics", style={'color': 'blue'}),
                    html.Ul([html.Li(f"{item['Rank']}. {item['Topic']}") for item in liberal_issues])
                ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '5%'})
            ])
        ])
    except Exception as e:
        issues_display = html.Div([html.H3("Failed to load top issues")])

    return scorecard_display, timeseries_fig, trait_display, bill_table, issues_display

if __name__ == '__main__':
    app.run_server(debug=False)


