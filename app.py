import dash
from dash import html
import pandas as pd

# Load CSV from GitHub (replace with your actual raw GitHub link)
csv_url = "https://raw.githubusercontent.com/ProblematicGoose/Project-Clout/refs/heads/main/normalized_sentiment.csv"
df = pd.read_csv(csv_url)

# Create Dash app
app = dash.Dash(__name__)

# Handle empty or valid dataset
if df.empty:
    app.layout = html.Div(
        "No data available",
        style={
            "textAlign": "center",
            "fontSize": "24px",
            "color": "gray",
            "paddingTop": "100px"
        }
    )
else:
    # Use first row of data
    subject = df['Subject'].iloc[0]
    score = df['NormalizedSentimentScore'].iloc[0]  # <-- Update if your column name is different

    # Choose color based on score
    if score < 400:
        color = 'crimson'
    elif score > 600:
        color = 'green'
    else:
        color = 'orange'

    # Apply layout using 'card' class styled in style.css
    app.layout = html.Div(
        html.Div([
            html.H1(subject),
            html.H2("Sentiment Score"),
            html.H3(f"{score:,}", style={"color": color})
        ], className="card")
    )

# Run locally (ignored by Render)
if __name__ == '__main__':
    app.run_server(debug=True)


