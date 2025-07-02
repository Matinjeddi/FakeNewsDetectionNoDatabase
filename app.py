from flask import Flask, request, render_template, redirect, url_for, session
from utils import *
import atexit
import signal 
import sys

app = Flask(__name__)
app.secret_key = 'CaKaBaSa'

predictions = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/news', methods=['GET', 'POST'])
def news():
    per_page = 10
    page = int(request.args.get('page', 1))
    
    # Get search parameters from either POST or GET
    if request.method == 'POST':
        query = request.form.get('query')
        sort_by = request.form.get('sort_by')
        # Store in session for pagination
        session['news_query'] = query
        session['news_sort_by'] = sort_by
        # redirect to the news page with the new query and sort_by
        return redirect(url_for('news', page=1, query=query, sort_by=sort_by))

    query = session.get('news_query', 'world')
    sort_by = session.get('news_sort_by', 'relevancy')
    
    # Get news articles
    news_articles = get_news(query, sort_by)
    
    # Calculate pagination
    total = len(news_articles)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_articles = news_articles[start_idx:end_idx]
    
    # Create pagination object
    pagination = {
        'has_next': end_idx < total,
        'has_prev': page > 1,
        'next_num': page + 1,
        'prev_num': page - 1,
        'pages': (total + per_page - 1) // per_page,
        'current_page': page
    }
    
    return render_template('news.html', 
                         news_articles=paginated_articles,
                         pagination=pagination,
                         page=page,
                         current_query=query,
                         current_sort=sort_by)

@app.route('/fetch_article', methods=['POST'])
def fetch_article():
    url = request.form.get('article_url')
    article_text = ''
    news_title = request.form.get('news_title', '')
    if url:
        article_text = scrape_article(url)
        
        session['prefill_news'] = article_text
        session['prefill_url'] = url
        session['news_title'] = news_title
    return redirect(url_for('predict'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prefill_news = session.pop('prefill_news', '')
    prefill_url = session.pop('prefill_url', '')
    news_title = session.pop('news_title', '')
    if request.method == 'POST':
        news = request.form['news']
        url = request.form.get('news_url', '')
        news_preprocessed = preprocess_text(news)
        news_title = request.form.get('news_title', '')
        prediction = predict_news(news_preprocessed)
        confidence = f"{predict_confidence(news_preprocessed):.2f}%"
        predictions.append({
            'date': datetime.now(),
            'url': url,
            'prediction': prediction,
            'confidence': confidence,
            'news_title': news_title
        })
        return render_template('predict.html', prediction=prediction, confidence=confidence, news_title=news_title)
    return render_template('predict.html', prefill_news=prefill_news, prefill_url=prefill_url, news_title=news_title)

@app.route('/statistics')
def statistics():
    per_page = 10
    page = int(request.args.get("page", 1))
    total_items = len(predictions)
    total_pages = (total_items + per_page - 1) // per_page
    has_next = page < total_pages
    has_prev = page > 1
    start = (page - 1) * per_page
    end = start + per_page
    page_predictions = predictions[start:end]
    mean_confidence = calculate_mean_confidence(page_predictions)

    # Sortera på datum, nyast först
    sorted_predictions = sorted(predictions, key=lambda x: parse_date(x["date"]), reverse=True)

    # Skapa sidindelning
    start = (page - 1) * per_page
    end = start + per_page
    page_predictions = sorted_predictions[start:end]

    mean_confidence = calculate_mean_confidence(page_predictions)
    # Count predictions
    true_count = sum(1 for p in predictions if p["prediction"].lower() == 'real news')
    false_count = sum(1 for p in predictions if p["prediction"].lower() == 'fake news')
    return render_template(
        'statistics.html',
        predictions=predictions,
        mean_confidence=mean_confidence,
        true_count=true_count,
        false_count=false_count,
        pagination=page_predictions,
        page=page,
        has_next=has_next,
        has_prev=has_prev,
        per_page=per_page
    )

@app.route('/about')
def about():
    return render_template('about.html')

def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    sys.exit(0)

if __name__ == '__main__':
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function
    atexit.register(lambda: print('Cleaning up...'))
    
    app.run(debug=True)