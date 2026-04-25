from flask import Flask
from flask_cors import CORS
from routes import register_routes

app = Flask(__name__)


# [ONLY FOR DEVELOPMENT] Allow CORS for the frontend running on localhost:5173. Later change to use build version of frontend.
CORS(app, origins=["http://localhost:5173"]) 

register_routes(app)

if __name__ == "__main__":
    app.run(debug=True)