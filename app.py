from flask import Flask, request, jsonify, render_template
from agent import build_agent, get_weak_topics

app = Flask(__name__)

# Build agent once at startup — not on every request
print("Loading agent...")
agent = build_agent()
print("Agent ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content
    weak_topics = get_weak_topics()
    return jsonify({"answer": answer, "weak_topics": weak_topics})


if __name__ == "__main__":
    app.run(debug=True, port=5000)