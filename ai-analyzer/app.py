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

Provide the analysis with answers of each of the above mentioned 3 topics. Keep the analysis as concise as possible. At the end of your analysis ask a follow-up question acting as a devil's advocate. This is the start of our conversation. I will ask follow-up questions after this.

Assistant:
"""
)

# 3. Conversational Prompt (for follow-up questions)
# This prompt is designed to make the AI act as a "devil's advocate".
conversational_template = """
The following is a debate between a human and an AI. The AI's role is to act as a "devil's advocate."
The initial analysis of an audience's bias is provided below. Based on that analysis, the AI must take the *opposite* stance during the conversation.
The AI should respectfully challenge the user's assumptions and present well-reasoned counter-arguments to the audience's likely point of view. The goal is to help the user explore alternative perspectives. The AI should consider basic empathy and not make arguments that are harmful to any community.

Initial Analysis Summary (The audience's likely bias):
{initial_analysis}

Current conversation:
{history}
Human: {input}
Assistant:"""
CONVERSATION_PROMPT = PromptTemplate(input_variables=["initial_analysis", "history", "input"], template=conversational_template)


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
        # The initial analysis is the first message from the assistant in the history
        initial_analysis = ""
        if history and history[0]["role"] == "assistant":
            initial_analysis = history[0]["content"]

        # Create a new conversation chain for each request, seeding it with history
        # The `initial_analysis` is passed as a partial variable to the prompt template.
        memory = ConversationBufferMemory(memory_key="history", human_prefix="Human", ai_prefix="Assistant")
        # We skip the first message because it's the initial analysis, not a conversational turn
        for message in history[1:]:
            if message["role"] == "user":
                memory.chat_memory.add_user_message(message["content"])
            elif message["role"] == "assistant":
                memory.chat_memory.add_ai_message(message["content"])

        # The ConversationChain expects the prompt to have 'history' and 'input' as variables.
        # We can pre-fill the 'initial_analysis' variable in the prompt.
        prompt = CONVERSATION_PROMPT.partial(initial_analysis=initial_analysis)

        conversation = ConversationChain(
            llm=llm,
            prompt=prompt,
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
