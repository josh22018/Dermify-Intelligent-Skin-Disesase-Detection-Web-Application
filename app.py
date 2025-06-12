from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from groq import Groq
import io
import re
import base64

app = Flask(__name__)

MODEL_NAME = "kar1hik/computer-vision-project"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

client = Groq(api_key="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")  # Replace with your actual API key

def predict_skin_disease(image):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

def format_groq_response(raw):
    # Replace section headings
    raw = re.sub(r"\*\*Description:\*\*", "<h4>Description</h4>", raw)
    raw = re.sub(r"\*\*Causes:\*\*", "<h4>Causes</h4>", raw)
    raw = re.sub(r"\*\*Precautions:\*\*", "<h4>Precautions</h4>", raw)
    raw = re.sub(r"\*\*Risks:\*\*", "<h4>Risks</h4>", raw)
    raw = re.sub(r"\*\*Treatment Options:\*\*", "<h4>Treatment Options</h4>", raw)
    raw = re.sub(r"\*\*.*?:\*\*", lambda m: f"<h4>{m.group()[2:-2]}</h4>", raw)

    # Replace bullet points
    raw = raw.replace("* ", "<li>").replace("\n", "</li>\n")

    # Detect and convert raw links to HTML anchors
    url_pattern = r"(https?://[^\s]+)"
    raw = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', raw)

    return f"<ul>{raw}</ul>"


def get_disease_info(disease_name):
    prompt = f"Provide a detailed explanation about the skin disease '{disease_name}', including description, causes, precautions, risks, and treatment options in small crisp points, and give me a link for more information related to the disease '{disease_name}'"
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    raw = chat_completion.choices[0].message.content
    return format_groq_response(raw)

def chatbot_response(disease_name, user_query):
    prompt = f"The detected skin disease is '{disease_name}'. {user_query}"
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    file = request.files["image"]
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))
    disease = predict_skin_disease(image)
    info = get_disease_info(disease)
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{encoded_image}"
    return render_template("result.html", disease=disease, info=info, image_url=image_url)

@app.route("/chat", methods=["POST"])
def chat():
    disease = request.form["disease"]
    query = request.form["query"]
    response = chatbot_response(disease, query)
    return jsonify(response=response)

if __name__ == "__main__":
    app.run(debug=True)
