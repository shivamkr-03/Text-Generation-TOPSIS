import os

def create_folders():
    os.makedirs("results", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
