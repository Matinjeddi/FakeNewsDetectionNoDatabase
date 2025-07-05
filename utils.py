import urllib.robotparser
from urllib.parse import urlparse
import re
from transformers import pipeline
import newsapi
from datetime import datetime
import pytz
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

load_dotenv()

news_api = newsapi.NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))

model = pipeline('text-classification', model='matinjeddi/fake-news-roberta-base')
tokenizer = model.tokenizer

def get_news(query, sort_by):
    all_articles = news_api.get_everything(q=query, language='en', sort_by=sort_by)
    articles = all_articles['articles']

    # Convert UTC dates to local timezone
    local_tz = pytz.timezone('Europe/Stockholm')  # or your preferred timezone
    for article in articles:
        if article.get('publishedAt'):
            # Parse the UTC date
            utc_date = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
            local_date = utc_date.replace(tzinfo=pytz.UTC).astimezone(local_tz)
            # Update the article's date
            article['publishedAt'] = local_date.strftime('%Y-%m-%d %H:%M:%S')
    
    
    return articles

def predict_news(news):
    result = model(news, truncation=True, padding=True, max_length=512)
    predictions = result[0]['label']
    if predictions == "LABEL_1" or predictions.lower() == "real":
        return 'Real News'
    elif predictions == "LABEL_0" or predictions.lower() == "fake":
        return 'Fake News'
    
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and digits
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def predict_confidence(news):
    result = model(news, truncation=True, padding=True, max_length=512)
    confidence = result[0]['score'] * 100
    return confidence

def parse_date(d):
    if isinstance(d, datetime):
        return d
    return datetime.strptime(d, "%Y-%m-%d")

def calculate_mean_confidence(predictions_subset):
    if not predictions_subset:
        return 0

    return sum(float(p["confidence"].replace('%', '')) for p in predictions_subset) / len(predictions_subset)


def is_scraping_allowed(url, user_agent='*'):
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)

    try:
        rp.read()
        can_fetch = rp.can_fetch(user_agent, url)
        print(f"Scraping allowed: {can_fetch}")
        return can_fetch
    except Exception as e:
        return False

def is_paywall(url):
    # Common paywall indicators
    paywall_indicators = [
        'subscribe',
        'sign in',
        'log in',
        'premium',
        'membership',
        'paywall',
        'subscription',
        'limited access',
        'free trial',
        'unlock',
        'register to read',
        'join now',
        'become a member',
        'digital subscription',
        'subscribe to read',
        'subscribe to continue',
        'subscribe to view',
        'subscribe to access',
        'subscribe to unlock',
        'subscribe to read more',
        'subscribe to continue reading',
        'subscribe to view more',
        'subscribe to access more',
        'subscribe to unlock more',
        'subscribe to read the full article',
        'subscribe to continue reading the full article',
        'subscribe to view the full article',
        'subscribe to access the full article',
        'subscribe to unlock the full article'
    ]
    
    # Check URL for paywall indicators
    url_lower = url.lower()
    if any(indicator in url_lower for indicator in paywall_indicators):
        print("Paywall detected")
        return False
    
    # Check for common paywall domains
    paywall_domains = [
        'wsj.com',  # Wall Street Journal
        'nytimes.com',  # New York Times
        'ft.com',  # Financial Times
        'bloomberg.com',  # Bloomberg
        'washingtonpost.com',  # Washington Post
        'thetimes.co.uk',  # The Times
        'theguardian.com',  # The Guardian (premium content)
        'telegraph.co.uk',  # The Telegraph
        'economist.com',  # The Economist
        'newyorker.com',  # The New Yorker
        'spectator.co.uk',  # The Spectator
        'prospectmagazine.co.uk',  # Prospect Magazine
        'foreignpolicy.com',  # Foreign Policy
        'foreignaffairs.com',  # Foreign Affairs
        'nature.com',  # Nature
        'science.org',  # Science
        'jstor.org',  # JSTOR
        'sciencedirect.com',  # ScienceDirect
        'springer.com',  # Springer
        'wiley.com',  # Wiley
        'tandfonline.com',  # Taylor & Francis
        'sage.com',  # SAGE
        'emerald.com',  # Emerald
        'ieee.org',  # IEEE
        'acm.org',  # ACM
        'sciencedaily.com',  # Science Daily
        'phys.org',  # Phys.org
        'medicalnewstoday.com',  # Medical News Today
        'healthline.com',  # Healthline
        'webmd.com',  # WebMD
        'mayoclinic.org',  # Mayo Clinic
        'harvard.edu',  # Harvard
        'mit.edu',  # MIT
        'stanford.edu',  # Stanford
        'oxford.ac.uk',  # Oxford
        'cambridge.org',  # Cambridge
        'princeton.edu',  # Princeton
        'yale.edu',  # Yale
        'berkeley.edu',  # Berkeley
        'ucla.edu',  # UCLA
        'caltech.edu',  # Caltech
        'cornell.edu',  # Cornell
        'columbia.edu',  # Columbia
        'upenn.edu',  # University of Pennsylvania
        'uchicago.edu',  # University of Chicago
        'umich.edu',  # University of Michigan
        'utexas.edu',  # University of Texas
        'wisc.edu',  # University of Wisconsin
        'illinois.edu',  # University of Illinois
        'purdue.edu',  # Purdue
        'gatech.edu',  # Georgia Tech
        'cmu.edu',  # Carnegie Mellon
        'jhu.edu',  # Johns Hopkins
        'duke.edu',  # Duke
        'northwestern.edu',  # Northwestern
        'vanderbilt.edu',  # Vanderbilt
        'rice.edu',  # Rice
        'brown.edu',  # Brown
        'dartmouth.edu',  # Dartmouth
    ]

    domain = urlparse(url).netloc.lower()
    if any(paywall_domain in domain for paywall_domain in paywall_domains):
        print("Paywall detected")
        return False
    print("No paywall detected")
    return True


def scrape_article(url):
        try:
            if not is_scraping_allowed(url) or not is_paywall(url):
                article_text = 'Scraping is not allowed for this URL or it is a paywall.'
            else:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements including author information
                for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'button', 'input', 'select', 'iframe', 
                                            'meta', 'link', 'author', 'span', 'div'], 
                                           class_=re.compile(r'author|byline|writer|reporter|contributor|credit', re.I)):
                    element.decompose()
                
                # Try to find the main article content
                article = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile(r'article|content|post|entry', re.I))
                
                if article:
                    # Get only paragraphs from the article
                    paragraphs = article.find_all('p')
                    
                    # Filter and clean paragraphs
                    cleaned_paragraphs = []
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        # Skip empty paragraphs, very short ones, and author information
                        if len(text) > 20 and not re.search(r'by\s+\w+\s+\w+|\w+\s+\w+\s+reports|\w+\s+\w+\s+writes', text, re.I):
                            # Clean the text
                            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
                            text = text.strip()
                            cleaned_paragraphs.append(text)
                    
                    # Join paragraphs with double newlines
                    article_text = '\n\n'.join(cleaned_paragraphs)

                # If no content was found, return a message
                if not article_text:
                    article_text = 'Could not extract article content. The article might be behind a paywall or the content structure is not recognized.'
                
        except Exception as e:
            print(f'Error fetching article: {e}')
            article_text = 'Could not fetch article text. Please check the URL and try again.'
        return article_text
