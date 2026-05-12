from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import datetime
from src.detector import predict_authenticity, load_model, load_datasets
from src.fetcher import fetch_news, scrape_article_from_url
from src.config import APP_NAME, APP_VERSION

app = Flask(__name__)
CORS(app) # Enable CORS for frontend access

# Load model globally
pipeline = load_model()

@app.route('/api/status', methods=['GET'])
def get_status():
    metrics = {}
    try:
        metrics_path = os.path.join('outputs', 'metrics.json')
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
    except:
        pass

    return jsonify({
        "app": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "model_loaded": pipeline is not None,
        "metrics": metrics
    })

@app.route('/api/news', methods=['GET'])
def get_news():
    # Accept refresh as a string (JS sends Date.now() which is too big for int)
    refresh = request.args.get('refresh', '0')
    try:
        news = fetch_news(str(refresh))
        return jsonify({
            "status": "success",
            "count": len(news),
            "results": news
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"status": "error", "message": "No text provided"}), 400
    
    # Pass live_news=None to skip live cross-reference (too slow for real-time)
    verdict, score, reasons, h_score, parties = predict_authenticity(text, pipeline, live_news=None)
    
    return jsonify({
        "verdict": verdict,
        "confidence": float(score),
        "reasons": reasons,
        "heuristic_score": float(h_score),
        "parties": parties,
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/api/scrape', methods=['POST'])
def scrape():
    data = request.json
    url = data.get('url', '')
    if not url:
        return jsonify({"status": "error", "message": "No URL provided"}), 400
    
    title, text = scrape_article_from_url(url)
    return jsonify({
        "title": title,
        "text": text
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting TruthLens API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
