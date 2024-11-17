from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy import stats
import json
import time
from datetime import datetime
import os

@dataclass
class ModelPerformance:
    accuracy: float
    latency: float
    timestamp: float

@dataclass
class Experiment:
    name: str
    model_a_id: str
    model_b_id: str
    traffic_split: float  # Percentage of traffic to model B (0-1)
    start_time: float
    end_time: Optional[float]
    model_a_performance: List[ModelPerformance]
    model_b_performance: List[ModelPerformance]
    
    def to_dict(self):
        return {
            "name": self.name,
            "model_a_id": self.model_a_id,
            "model_b_id": self.model_b_id,
            "traffic_split": self.traffic_split,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "model_a_performance": [vars(p) for p in self.model_a_performance],
            "model_b_performance": [vars(p) for p in self.model_b_performance]
        }

class ExperimentManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, storage_path="experiments/data"):
        if not self._initialized:
            self.storage_path = storage_path
            self.active_experiments: Dict[str, Experiment] = {}
            os.makedirs(storage_path, exist_ok=True)
            self._load_experiments()
            self._initialized = True

    def _load_experiments(self):
        """Load existing experiments from storage"""
        try:
            with open(f"{self.storage_path}/experiments.json", "r") as f:
                data = json.load(f)
                for exp_data in data:
                    exp = Experiment(
                        name=exp_data["name"],
                        model_a_id=exp_data["model_a_id"],
                        model_b_id=exp_data["model_b_id"],
                        traffic_split=exp_data["traffic_split"],
                        start_time=exp_data["start_time"],
                        end_time=exp_data["end_time"],
                        model_a_performance=[ModelPerformance(**p) for p in exp_data["model_a_performance"]],
                        model_b_performance=[ModelPerformance(**p) for p in exp_data["model_b_performance"]]
                    )
                    if not exp.end_time:  # Only load active experiments
                        self.active_experiments[exp.name] = exp
        except FileNotFoundError:
            pass

    def _save_experiments(self):
        """Save experiments to storage"""
        #print(f"Saving experiments: {list(self.active_experiments.keys())}")  # Debug print
        #print(f"Storage path: {self.storage_path}")  # Debug print
        with open(f"{self.storage_path}/experiments.json", "w") as f:
            data = [exp.to_dict() for exp in self.active_experiments.values()]
            #print(f"Saving data: {data}")  # Debug print
            json.dump(data, f)
            
    def create_experiment(self, name: str, model_a_id: str, model_b_id: str, traffic_split: float = 0.5):
        """Create a new A/B test experiment"""
        if name in self.active_experiments:
            raise ValueError(f"Experiment {name} already exists")
        
        experiment = Experiment(
            name=name,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            traffic_split=traffic_split,
            start_time=time.time(),
            end_time=None,
            model_a_performance=[],
            model_b_performance=[]
        )
        self.active_experiments[name] = experiment
        self._save_experiments()
        return experiment

    def end_experiment(self, name: str):
        """End an active experiment"""
        if name not in self.active_experiments:
            raise ValueError(f"Experiment {name} not found")
        
        experiment = self.active_experiments[name]
        experiment.end_time = time.time()
        del self.active_experiments[name]
        self._save_experiments()
        return experiment

    def record_performance(self, experiment_name: str, model_id: str, accuracy: float, latency: float):
        """Record a performance measurement for a model in an experiment"""
        #print(f"Recording performance for experiment {experiment_name}, model {model_id}")  
        #print(f"Current active experiments: {list(self.active_experiments.keys())}")  
        
        if experiment_name not in self.active_experiments:
            #print(f"Experiment {experiment_name} not found in active experiments")  
            return
        
        experiment = self.active_experiments[experiment_name]
        performance = ModelPerformance(accuracy=accuracy, latency=latency, timestamp=time.time())
        
        if model_id == experiment.model_a_id:
            #print(f"Adding performance data to model A")  
            experiment.model_a_performance.append(performance)
        elif model_id == experiment.model_b_id:
            #print(f"Adding performance data to model B")  
            experiment.model_b_performance.append(performance)
        # else:
            #print(f"Model {model_id} not found in experiment {experiment_name}")  
        
        self._save_experiments()

    def get_experiment_results(self, name: str):
        """Calculate statistical results for an experiment"""
        if name not in self.active_experiments:
            raise ValueError(f"Experiment {name} does not exist")
        
        experiment = self.active_experiments[name]
        model_a_perf = experiment.model_a_performance
        model_b_perf = experiment.model_b_performance
        
        #print(f"Model A samples: {len(model_a_perf)}, Model B samples: {len(model_b_perf)}")
        
        # Calculate statistics for both metrics
        results = {}
        for metric in ['accuracy', 'latency']:
            # Access ModelPerformance dataclass fields
            model_a_values = [float(getattr(p, metric)) for p in model_a_perf]
            model_b_values = [float(getattr(p, metric)) for p in model_b_perf]
            
            # Initialize default values
            default_result = {
                'model_a_mean': None,
                'model_b_mean': None,
                'model_a_std': None,
                'model_b_std': None,
                'model_a_ci': (None, None),
                'model_b_ci': (None, None),
                'difference': None,
                'effect_size': None,
                'p_value': None,
                'significant': False,
                'sample_size_a': len(model_a_values),
                'sample_size_b': len(model_b_values),
                'model_a_performance': model_a_values,
                'model_b_performance': model_b_values
            }
            
            # Check if we have enough samples for meaningful statistics
            MIN_SAMPLES = 2
            if len(model_a_values) < MIN_SAMPLES or len(model_b_values) < MIN_SAMPLES:
                results[metric] = default_result
                continue
            
            try:
                # Calculate basic statistics
                model_a_mean = float(np.mean(model_a_values))
                model_b_mean = float(np.mean(model_b_values))
                model_a_std = float(np.std(model_a_values, ddof=1))  # Using ddof=1 for sample standard deviation
                model_b_std = float(np.std(model_b_values, ddof=1))
                
                # Calculate confidence intervals
                model_a_ci = tuple(float(x) for x in stats.t.interval(0.95, len(model_a_values)-1, 
                                          loc=model_a_mean, 
                                          scale=model_a_std/np.sqrt(len(model_a_values))))
                
                model_b_ci = tuple(float(x) for x in stats.t.interval(0.95, len(model_b_values)-1, 
                                          loc=model_b_mean, 
                                          scale=model_b_std/np.sqrt(len(model_b_values))))
                
                # Calculate t-test and effect size
                t_stat, p_value = stats.ttest_ind(model_a_values, model_b_values)
                p_value = float(p_value)
                
                # Calculate Cohen's d effect size
                pooled_std = np.sqrt((model_a_std**2 + model_b_std**2) / 2)
                effect_size = (model_b_mean - model_a_mean) / pooled_std if pooled_std > 0 else 0.0
                
                results[metric] = {
                    'model_a_mean': model_a_mean,
                    'model_b_mean': model_b_mean,
                    'model_a_std': model_a_std,
                    'model_b_std': model_b_std,
                    'model_a_ci': model_a_ci,
                    'model_b_ci': model_b_ci,
                    'difference': model_b_mean - model_a_mean,
                    'effect_size': effect_size,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'sample_size_a': len(model_a_values),
                    'sample_size_b': len(model_b_values),
                    'model_a_performance': model_a_values,
                    'model_b_performance': model_b_values
                }
            except (ValueError, RuntimeWarning, TypeError) as e:
                #print(f"Error calculating statistics for {metric}: {str(e)}")
                results[metric] = default_result
        
        return results

    def get_experiment_summary(self, name: str):
        """Get a summary of the experiment for visualization"""
        if name not in self.active_experiments:
            raise ValueError(f"Experiment {name} does not exist")
        
        experiment = self.active_experiments[name]
        #print(f"Getting summary for experiment {name}")
        
        # Get statistical results if we have data
        results = self.get_experiment_results(name) if (experiment.model_a_performance or experiment.model_b_performance) else {
            "accuracy": {
                "model_a_mean": 0,
                "model_b_mean": 0,
                "model_a_std": 0,
                "model_b_std": 0,
                "model_a_ci": [0, 0],
                "model_b_ci": [0, 0],
                "difference": 0,
                "effect_size": None,
                "p_value": 1.0,
                "significant": False,
                "sample_size_a": 0,
                "sample_size_b": 0,
                "model_a_performance": [],
                "model_b_performance": []
            },
            "latency": {
                "model_a_mean": 0,
                "model_b_mean": 0,
                "model_a_std": 0,
                "model_b_std": 0,
                "model_a_ci": [0, 0],
                "model_b_ci": [0, 0],
                "difference": 0,
                "effect_size": None,
                "p_value": 1.0,
                "significant": False,
                "sample_size_a": 0,
                "sample_size_b": 0,
                "model_a_performance": [],
                "model_b_performance": []
            }
        }
        
        summary = {
            "name": experiment.name,
            "model_a_id": experiment.model_a_id,
            "model_b_id": experiment.model_b_id,
            "traffic_split": experiment.traffic_split,
            "start_time": experiment.start_time,
            "end_time": experiment.end_time,
            "results": results
        }
        
        #print(f"Sending summary for {name}: {summary}")
        return summary

    def get_active_experiments(self):
        """Get all active experiments"""
        return self.active_experiments

    def delete_experiment(self, name: str):
        """Delete an experiment completely"""
        if name not in self.active_experiments:
            raise ValueError(f"Experiment {name} does not exist")
        
        # End the experiment if it hasn't been ended
        if self.active_experiments[name].end_time is None:
            self.end_experiment(name)
        
        # Remove from active experiments
        del self.active_experiments[name]
        
        # Save the updated state
        self._save_experiments()
        
        # Clean up any experiment-specific files if they exist
        experiment_file = f"{self.storage_path}/{name}.json"
        if os.path.exists(experiment_file):
            os.remove(experiment_file)
