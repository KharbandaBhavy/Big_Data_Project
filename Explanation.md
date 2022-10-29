# Big Data Project 
---
## Fake News Classification using Native languages.

### Introduction to the Project

Fake News is the misinformation disseminated among the public by mainstream sources like media outlets and social media. It is generally misleading to shape beliefs of the masses to one's favour. There are Different approaches to identifying fake news were examined, such as content-based classification, social context-based classification, image-based classification, sentiment-based classification, and hybrid context-based classification. This project aims to propose a model for fake news classification based on news titles, following the content-based classification approach. The model uses a **BERT Model** and further **Logistic Regression**. Training and evaluation of the model were done on the FakeNewsNet dataset. 

### Our Approach to the problem
In this project, we have used two different approaches to establish a model for detecting fake news in native languages.

>The jupyter notebook file `1.ipynb` and `2.ipynb` shows the first approach of this project. 

`1.ipynb` file takes a predefined and validated Hindi dataset for Fake news detection. We applied Data preprocessing on the dataset, like data cleaning, data stemming, removal of stop words, and tokkenization everything performed on Hindi language directly. Further, this data is saved in a csv file as our final dataset to train the model upon it. 
`2.ipynb` file contains the training of the model. In this project, we propose a **BERT**-based deep learning approach by combining different parallel blocks of the single-layer CNNs with the Bidirectional Encoder Representations from Transformers (BERT). We utilize BERT as a sentence encoder, which can accurately get the context representation of a sentence. However, a deep neural network with bidirectional training approach can be an optimal and accurate solution for the detection of fake news. Our proposed method improves the performance of fake news detection with the powerful ability to capture semantic and long-distance dependencies in sentences.
Although, our model is representing a very low accuracy of **51%**, it might be because of the dataset that is in Hindi Language.

---

>The jupyter notebook file `3.ipynb` and `4.ipynb` shows the second approach of this project. 

This second approach is the unique element of our project, presently, we have validated datasets only for few languages like English and Hindi, but what if in future we developed the datasets in native languages and we require a model for fake news classification in that particular language. 
In our second approach, we built a transformer, using **Google Translate API** to convert the dataset from Native language to a standard language and then apply the standard classification model on the same.
`3.ipynb` file takes up the Hindi dataset(any language dataset for future) feeds it into a transformer and entirely translates it to English language. Same procedure of Data pre-processing is then applied to translated dataset(Data cleaning removal of stopwords and null values, data stemming etc). 
Further a standard Logistic regression model is trained on this dataset for the classification on fake news. This model provides us with the accuracy score of **97.69%** (training) and **97.36%** (testing).
`4.ipynb` file is a beta testing model for this approach which takes up the pre-processed Hindi dataset and perform the vector tokenization, using the tfidf approach, and gives the accuracy of **93.99%** (training) and **86.18%** testing but here we are not sure whether it is performing the tokwnization precisely or not, although it improves the accuracy with a large value. 

### Future Scope and Research
Our project, lead us to explore another area of comparison between BERT Model and Vanilla Logistic Regression when applied on different languages.
As when we applied BERT Model on Hindi dataset we got a very low point of accuracy while if we apply the same model on a standard English dataset, we got a high pointer of accuracy. Similarly, Logistic regression when directly applied on English lannguage gave a good accuracy as compared to Hindi dataset(tfidf vectorization). 
Although, BERT Model is recommended to use in such situations of Fake news classification as it can predict better as it understands the sequence of sentence better than the Logistic reression approach. But our model, invalidates this theory for languages other than English. 

---
### Dataset Explanation 
>Contents of Data : Approach 1

* `fake_news.json` and `valid_news.json` contains the raw Hindi dataset in json format with full detail (url, short_description, long_description, full_title). 
* `fake_news.csv` and `true_news.csv` contains the cleaned dataset that is to be used for model training. It contains the final description of the news and its id.
* `Hindi_stopwords.txt` This text file contains the stop words of hindi language. 
* `Final_dataset.csv` This file is the combination of `fake_news.csv` and `true_news.csv` with their labels. 

>Contents of Data : Approach 2

* `fake_news.json` and `valid_news.json` contains the raw Hindi dataset in json format with full detail (url, short_description, long_description, full_title).
* `fake_news_raw.csv` and `true_news_raw.csv` These files are just csv format of same json file above.
* `fake_news.csv` and `true_news.csv` contains the cleaned dataset that is to be used for model training. It contains the final description of the news and its id.
* `fake_news.txt` and `true_news.txt` contains the text versions of clean hindi dataset. 
* `Translated_fake_news.txt` and `Translated_true_news.txt` contains translated dataset to hindi language.
* `clean_fake_news.txt` and `clean_true_news.txt` contains the pre-processed English dataset, these are the final datasets for model training and processing. 
* `Final_dataset.csv` This file is the combination of `clean_fake_news.csv` and `clean_true_news.csv` with their labels. 


