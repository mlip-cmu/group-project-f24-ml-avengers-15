from flask import Blueprint, jsonify, request
from .experiment_manager import ExperimentManager

experiment_api = Blueprint('experiment_api', __name__, url_prefix='/experiments')
experiment_manager = ExperimentManager()

@experiment_api.route('/', methods=['POST'])
def create_experiment():
    data = request.json
    print("Received experiment creation request:", data)  # Debug print
    try:
        experiment = experiment_manager.create_experiment(
            name=data['name'],
            model_a_id=data['model_a_id'],
            model_b_id=data['model_b_id'],
            traffic_split=data.get('traffic_split', 0.5)
        )
        return jsonify(experiment.to_dict())
    except Exception as e:
        print("Error creating experiment:", str(e))  # Debug print
        return jsonify({"error": str(e)}), 400

@experiment_api.route('/<experiment_name>', methods=['DELETE'])
def end_experiment(experiment_name):
    try:
        experiment = experiment_manager.end_experiment(experiment_name)
        return jsonify(experiment.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@experiment_api.route('/<experiment_name>/results', methods=['GET'])
def get_experiment_results(experiment_name):
    try:
        results = experiment_manager.get_experiment_results(experiment_name)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@experiment_api.route('/<experiment_name>/summary', methods=['GET'])
def get_experiment_summary(experiment_name):
    try:
        summary = experiment_manager.get_experiment_summary(experiment_name)
        print(f"Sending summary for {experiment_name}: {summary}")  # Debug print
        return jsonify(summary)
    except Exception as e:
        print(f"Error getting summary for {experiment_name}: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 404

@experiment_api.route('/', methods=['GET'])
def list_experiments():
    experiments = {name: exp.to_dict() for name, exp in experiment_manager.active_experiments.items()}
    return jsonify(experiments)
