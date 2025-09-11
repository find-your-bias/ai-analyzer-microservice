import os
import psycopg2
from flask import Flask, jsonify
from flask_cors import CORS
from langchain_aws.llms import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)
CORS(app) # This will enable CORS for all routes

# 1. Configure LangChain with Bedrock
# No need to manually create a boto3 client. LangChain handles it.
llm = BedrockLLM(
    model_id="anthropic.claude-v2",
    model_kwargs={"max_tokens_to_sample": 500, "temperature": 0.7}
)

# 2. Create a Prompt Template
# The template now has an {input} variable for the formatted votes.
prompt_template = PromptTemplate(
    input_variables=["formatted_votes"],
    template="""
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
)

# 3. Create an LLMChain
# This chain combines the prompt template and the LLM.
chain = LLMChain(llm=llm, prompt=prompt_template)

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
    Analyzes the voting data by sending it to AWS Bedrock using LangChain and returns the analysis.
    """
    print("result reached ai-analyzer")
    try:
        # Fetch data from the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT tweet, vote FROM votes WHERE created_at > NOW() - INTERVAL '10 minutes'")
        votes_data = cur.fetchall()
        cur.close()
        conn.close()

        if not votes_data:
            return jsonify({"analysis": "Not enough data to perform analysis."})

        # Format the data for the prompt
        formatted_votes = "\n".join([f"- Tweet: \"{row[0]}\", Vote: {'Agree' if row[1] == 'a' else 'Disagree'}" for row in votes_data])
        
        # 4. Run the LangChain chain
        # The chain handles formatting the prompt and calling the model.
        result = chain.invoke({"formatted_votes": formatted_votes})
        analysis = result.get('text')

        return jsonify({"analysis": analysis})

    except Exception as e:
        # Log the exception for debugging
        app.logger.error(f"An error occurred: {e}")
        # Generic error handling is more suitable now
        if "AccessDeniedException" in str(e):
             return jsonify({"error": "AWS credentials are not configured correctly or lack permissions for Bedrock.  "}), 403
        return jsonify({"error": "An internal error occurred while analyzing the votes."}), 500

@app.route("/health", methods=['GET'])
def health_check():
    return jsonify(status="ok"), 200

if __name__ == '__main__':
    # The app runs on port 5001 to avoid conflicts with other services *test*
    app.run(host='0.0.0.0', port=5001, debug=True)
