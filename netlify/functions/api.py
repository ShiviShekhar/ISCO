import sys
import os

# Add the 'src' directory to the Python path so we can import our app
# This allows the serverless function to find the 'src.app' module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

# Import the Flask app instance
from app import app

# Import the serverless wsgi handler
import serverless_wsgi

def handler(event, context):
    """This function is the entry point for the Netlify serverless function."""
    # The serverless_wsgi.handle function translates the serverless event
    # into a standard WSGI request that Flask can understand.
    return serverless_wsgi.handle(app, event, context)
