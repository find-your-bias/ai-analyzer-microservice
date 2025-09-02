import os
import json
import psycopg2
import boto3
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # This will enable CORS for all routes

# Configure Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        host="db",
        database="postgres",
        user="postgres",
        password="postgres"
    )
    return conn

@app.route('/', methods=['GET'])
def analyze_votes():
    """
    Analyzes the voting data by sending it to AWS Bedrock and returns the analysis.
    """
    print("result reached ai-analyzer")
    try:
        # 1. Fetch data from the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT tweet, vote FROM votes WHERE created_at > NOW() - INTERVAL '10 minutes'")
        votes_data = cur.fetchall()
        cur.close()
        conn.close()

        if not votes_data:
            return jsonify({"analysis": "Not enough data to perform analysis."})

        # 2. Format the data for the prompt
        formatted_votes = "\n".join([f"- Tweet: \"{row[0]}\", Vote: {'Agree' if row[1] == 'a' else 'Disagree'}" for row in votes_data])
        
        prompt = f"""
Human: Here is a list of votes from users reacting to various tweets. 

Voting Data:
{formatted_votes}

Based on this data, please provide a concise analysis of the voting audience. Answer the following:
1.  **Overall Bias**: What is the likely political bias of this audience (e.g., Left-leaning, Right-leaning, Centrist, etc.)?
2.  **Echo Chamber Strength**: How strong is the echo chamber? Is the audience open to different viewpoints or do they vote predictably along party lines?
3.  **Key Themes**: What are the key topics or themes that the audience strongly agrees or disagrees with?

Provide the analysis as a single block of text.

Assistant:
"""

        # 3. Call the Bedrock API (Claude model)
        body = json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 500,
            "temperature": 0.7,
        })

        modelId = 'anthropic.claude-v2'
        accept = 'application/json'
        contentType = 'application/json'

        response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        
        analysis = response_body.get('completion')

        return jsonify({"analysis": analysis})

    except Exception as e:
        # Log the exception for debugging
        app.logger.error(f"An error occurred: {e}")
        # Check for specific Boto3 exceptions if needed
        if "AccessDeniedException" in str(e):
             return jsonify({"error": "AWS credentials are not configured correctly or lack permissions for Bedrock."}), 403
        return jsonify({"error": "An internal error occurred while analyzing the votes."}), 500

@app.route("/health", methods=['GET'])
def health_check():
    return jsonify(status="ok"), 200

if __name__ == '__main__':
    # The app runs on port 5001 to avoid conflicts with other services
    app.run(host='0.0.0.0', port=5001, debug=True)
