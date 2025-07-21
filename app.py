import dash
from dash import html
import pandas as pd

# Sample DataFrame (you can replace with actual data loading logic later)
df = pd.DataFrame({
    'Subject': ['Ted Cruz'],
    'NormalizedSentimentScore': [475]
})

# Determine layout based on data
if df.empty:
    content = html.Div(
        "No data available",
        style={
            "textAlign": "center",
            "fontSize": "24px",
            "color": "gray",
            "paddingTop": "100px"
        }
    )
else:
    subject = df['Subject'].iloc[0]
    score = df['NormalizedSentimentScore'].iloc[0]

    if score < 400:
        color = 'crimson'
    elif score > 600:
        color = 'green'
    else:
        color = 'orange'

    content = html.Div([
        html.Div(subject, style={
            'fontSize': '48px',
            'fontWeight': 'bold',
            'textAlign': 'center'
        }),
        html.Div("Sentiment Score", style={
            'fontSize': '28px',
            'color': 'gray',
            'textAlign': 'center'
        }),
        html.Div(f"{score:,}", style={
            'fontSize': '72px',
            'color': color,
            'textAlign': 'center'
        })
    ], style={"paddingTop": "50px"})

app = dash.Dash(__name__)
app.layout = html.Div(content)

if __name__ == '__main__':
    app.run(debug=True)
