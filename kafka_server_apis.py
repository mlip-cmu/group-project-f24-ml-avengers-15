import os
import requests
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class KafkaServerApi:
    def __init__(self):
        self.server_ip = os.getenv('SERVER_IP')
        self.base_url = f"http://{self.server_ip}:8080"
        self.ssh_user = os.getenv('SSH_USER')
        self.ssh_password = os.getenv('SSH_PASSWORD')
        self.kafka_port = int(os.getenv('KAFKA_PORT', 9092))  
        self.local_port = int(os.getenv('LOCAL_PORT', 9092)) 
        self._start_ssh_tunnel()

    def _start_ssh_tunnel(self):
        """Sets up SSH tunnel for Kafka communication."""
        ssh_command = [
            'ssh', '-L', f'{self.local_port}:localhost:{self.kafka_port}',
            f'{self.ssh_user}@{self.server_ip}', '-NT'
        ]
        # Start the SSH tunnel in the background
        self.ssh_tunnel = subprocess.Popen(ssh_command)
        print(f"SSH tunnel established: {self.local_port} -> {self.server_ip}:{self.kafka_port}")

    def get_movie_info(self, movie_id):
        """Fetches movie information from the /movie/<movieid> endpoint."""
        try:
            url = f"{self.base_url}/movie/{movie_id}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching movie info for movie ID {movie_id}: {e}")
            return None

    def get_user_info(self, user_id):
        """Fetches user information from the /user/<userid> endpoint."""
        try:
            url = f"{self.base_url}/user/{user_id}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching user info for user ID {user_id}: {e}")
            return None

    def close(self):
        """Explicitly close the SSH tunnel."""
        if hasattr(self, 'ssh_tunnel'):
            self.ssh_tunnel.terminate()
            print("SSH tunnel terminated.")

    def __del__(self):
        self.close()

# Example usage:
if __name__ == "__main__":
    client = KafkaServerApi()  # No need to pass parameters, as they're loaded from environment variables

    # Fetch movie and user information
    movie_id = "georgia+rule+2007"
    user_id = 32775

    movie_info = client.get_movie_info(movie_id)
    user_info = client.get_user_info(user_id)

    print("Movie Info:", movie_info)
    print("User Info:", user_info)
