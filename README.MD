Web-based prediction of news articles with fine-tuned BERT model and flask, html, css, javascript.

In notebooks/ are all files that were used to pre-process and train the models. 

To run the app directly:

#1 In the terminal run 'py -m venv venv' then './venv/Scripts/activate', and finally 'pip install -r requirements.txt'

#3 Retrieve a API key from http://www.newsapi.org/ and add it to NEWS_API_KEY = your_api_key in an .env file

#4 Start a docker server and create a database called fake_newsd on that server

#5 Run 'py app.py' in the terminal

To use notebooks:

#1 Download https://zenodo.org/records/4561253, https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news and https://zenodo.org/records/2607278 and save them under datasets/ in this repo.

#2 Run pre_processing_WEL.ipynb

#3 run pre_processing_fake_or_real_news

#4 run pre_processing_FA-KES.ipynb

To train the models yourself: 

If you have a GPU, run Fine_tuning_BERT_models.ipynb directly, otherwise you can use Google Colab.

To analyze the models:

Run analyzing_bert_models.ipynb and analyzing_roberta_FA-KES.ipynb