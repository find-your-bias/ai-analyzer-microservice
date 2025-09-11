import os
import psycopg2
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

# 1. Configure LangChain with Bedrock
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"max_tokens": 500, "temperature": 0.7}
)

# 2. Initial Analysis Prompt (for the first turn)
initial_analysis_template = PromptTemplate(
    input_variables=["formatted_votes"],
    template="""
Human: Here is a list of votes from users reacting to various tweets.

Voting Data:
{formatted_votes}

Based on this data, please provide a concise analysis of the voting audience. Answer the following:
1. **Overall Bias**: What is the likely political bias of this audience (e.g., Left-leaning, Right-leaning, Centrist, etc.)?
2. **Echo Chamber Strength**: How strong is the echo chamber? Is the audience open to different viewpoints or do they vote predictably along party lines?
3. **Key Themes**: What are the key topics or themes that the audience strongly agrees or disagrees with?

Provide the analysis as a single block of text. This is the start of our conversation. I will ask follow-up questions after this.

Assistant:
"""
)

# 3. Conversational Prompt (for follow-up questions)
# This prompt now includes a {history} variable managed by LangChain.
conversational_template = """
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context.

Current conversation:
{history}
Human: {input}
Assistant:"""
CONVERSATION_PROMPT = PromptTemplate(input_variables=["history", "input"], template=conversational_template)


def get_db_connection():
    return psycopg2.connect(
        host="db",
        database="postgres",
        user="postgres",
        password="postgres"
    )

@app.route('/', methods=['GET'])
def analyze_votes():
    """
    Performs the initial analysis of the vote data.
    """
    print("result reached ai-analyzer for initial analysis")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT tweet, vote FROM votes WHERE created_at > NOW() - INTERVAL '10 minutes'")
        votes_data = cur.fetchall()
        cur.close()
        conn.close()

        if not votes_data:
            return jsonify({"analysis": "Not enough data to perform analysis."})

        formatted_votes = "\n".join(
            [f'- Tweet: "{row[0]}", Vote: {"Agree" if row[1] == "a" else "Disagree"}'
             for row in votes_data]
        )
        
        # Use a simple chain for the initial, non-conversational analysis
        initial_chain = LLMChain(llm=llm, prompt=initial_analysis_template)
        result = initial_chain.invoke({"formatted_votes": formatted_votes})
        analysis = result.get("text") or getattr(result, "content", "")

        return jsonify({"analysis": analysis})

    except Exception as e:
        app.logger.error(f"An error occurred during initial analysis: {e}")
        if "AccessDeniedException" in str(e):
            return jsonify({"error": "AWS credentials are not configured correctly or lack permissions for Bedrock."}), 403
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles follow-up questions in the conversation.
    """
    data = request.get_json()
    user_input = data.get("question")
    history = data.get("history", [])

    if not user_input:
        return jsonify({"error": "No user input provided."}), 400

    try:
        # Create a new conversation chain for each request, seeding it with history
        memory = ConversationBufferMemory(human_prefix="Human", ai_prefix="Assistant")
        for message in history:
            if message["role"] == "user":
                memory.chat_memory.add_user_message(message["content"])
            elif message["role"] == "assistant":
                memory.chat_memory.add_ai_message(message["content"])
        
        conversation = ConversationChain(
            llm=llm,
            prompt=CONVERSATION_PROMPT,
            memory=memory,
            verbose=True
        )
        
        response = conversation.predict(input=user_input)
        return jsonify({"response": response})

    except Exception as e:
        app.logger.error(f"An error occurred during chat: {e}")
        return jsonify({"error": "An error occurred while processing the chat."}), 500


@app.route("/health", methods=['GET'])
def health_check():
    return jsonify(status="ok"), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
