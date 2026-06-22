from flask import Flask, request, render_template
from utils import predict
import os



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    if request.method == "POST":
        data = {
            "age": float(request.form["age"]),
            "gaming_hours": float(request.form["gaming_hours"]),
            "study_hours": float(request.form["study_hours"]),
            "sleep_hours": float(request.form["sleep_hours"]),
            "attendance": float(request.form["attendance"]),
            "social_activity": float(request.form["social_activity"]),
            "device_usage": float(request.form["device_usage"]),
            "reaction_time_ms": float(request.form["reaction_time_ms"]),
            "addiction_score": float(request.form["addiction_score"]),
            "gender": request.form["gender"],
            "gaming_genre": request.form["gaming_genre"],
            "stress_level": request.form["stress_level"]
        }
        result, probability = predict(data)
    return render_template("index.html", result=result, probability=probability)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)