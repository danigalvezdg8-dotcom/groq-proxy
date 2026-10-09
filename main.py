from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

@app.route("/judge", methods=["POST"])
def judge():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    topic = data.get("topic", "unknown topic")
    players = data.get("players", [])
    messages = data.get("messages", [])

    transcript = ""
    for msg in messages:
        transcript += f"{msg['name']}: {msg['text']}\n"

    player_list = ", ".join(players)

    prompt = f"""You are an impartial debate judge. A debate just ended on the topic: "{topic}"

Participants: {player_list}

Transcript:
{transcript}

Respond ONLY with a JSON object (no extra text):
{{
  "winner": "<name of the winner>",
  "reason": "<1-2 sentences explaining why they won>",
  "feedback": {{
    {', '.join([f'"{p}": "<1 sentence of feedback for {p}>"' for p in players])}
  }}
}}"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "temperature": 0.3},
        timeout=15
    )

    if response.status_code != 200:
        return jsonify({"error": "Groq error"}), 500

    import json, re
    content = response.json()["choices"][0]["message"]["content"].strip()
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        return jsonify({"error": "Parse error"}), 500

    return jsonify(json.loads(match.group()))

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
