# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import plotly.express as px

# app = Flask(__name__, template_folder='templates')

# # Load data from CSV (or other source)
# def load_data(file_path='results/dummy_results.csv'):
#     return pd.read_csv(file_path)

# @app.route('/')
# def index():
#     data = load_data()
#     fig = px.histogram(data, x='performance', color='model', barmode='overlay', title='Model Performance Comparison')
#     fig_html = fig.to_html(full_html=False)
#     return render_template('visualize.html', plot=fig_html)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from scipy import stats

app = Flask(__name__, template_folder='templates')

# Load the performance metrics data
performance_data = pd.DataFrame({
    'Model': ['Model A', 'Model B'],
    'Accuracy': [0.8, 0.7],
    'F1 Score': [0.7, 0.6]
})

# Define the route for the home page
@app.route('/')
def home():
    return render_template('visualize.html', performance_data=performance_data)

# Define the route for the model proportion adjustment
@app.route('/adjust_proportion', methods=['POST'])
def adjust_proportion():
    model_a_proportion = float(request.form['model_a_proportion'])
    model_b_proportion = 1 - model_a_proportion
    
    # Update the performance metrics data with the new proportions
    performance_data['Proportion'] = [model_a_proportion, model_b_proportion]
    
    return render_template('visualize.html', performance_data=performance_data)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)