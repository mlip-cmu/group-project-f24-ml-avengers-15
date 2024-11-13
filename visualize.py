from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px

app = Flask(__name__, template_folder='templates')

# Load data from CSV (or other source)
def load_data(file_path='results/dummy_results.csv'):
    return pd.read_csv(file_path)

@app.route('/')
def index():
    data = load_data()
    fig = px.histogram(data, x='performance', color='model', barmode='overlay', title='Model Performance Comparison')
    fig_html = fig.to_html(full_html=False)
    return render_template('visualize.html', plot=fig_html)

if __name__ == "__main__":
    app.run(debug=True)
