import time
import re
import ollama # Replaced poe with ollama
from datetime import datetime
import pytz
from pymongo import MongoClient
from classification_zeroshot import NewsClassifier
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

class NewsReportGenerator:
    def __init__(self):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client["twitter_data"]
        self.tweet_filtered = self.db["tweet_filtered"]
        self.topic_status_change = self.db['topic_status_change']
        self.news_report = self.db['news_report']

        # Initialize ChromaDB client
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            print("NewsReportGenerator: ChromaDB Initialized persistent client at ./chroma_db")
        except Exception as e:
            print(f"NewsReportGenerator: Failed to initialize persistent ChromaDB: {e}. Falling back to in-memory.")
            self.chroma_client = chromadb.Client() # In-memory fallback
        self.rag_collection = self.chroma_client.get_or_create_collection(name="tweet_rag_embeddings")
        print(f"NewsReportGenerator: ChromaDB Collection 'tweet_rag_embeddings' loaded/created. Total items: {self.rag_collection.count()}")

        # Initialize Sentence Transformer model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("NewsReportGenerator: SentenceTransformer 'all-MiniLM-L6-v2' loaded.")
        except Exception as e:
            print(f"NewsReportGenerator: Failed to load SentenceTransformer 'all-MiniLM-L6-v2': {e}")
            self.embedding_model = None


    @staticmethod
    def clean_tweets(tweet):
        # Remove URLs
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        #tweet = re.sub(r'\W+|\d+', ' ', tweet)
        
        # Remove emojis
        tweet = tweet.encode('ascii', 'ignore').decode('ascii')
        
        # Convert to lowercase
        tweet = tweet.lower()
        return tweet
    
    @staticmethod
    def extract_title(md_string):
        # Regular expression to match Markdown title patterns
        title_pattern = r"^(?:\#{1,6}|\*{1,3}|\_{1,3})\s+(.+)$"
        lines = md_string.split("\n")

        for line in lines:
            match = re.match(title_pattern, line)
            if match:
                return match.group(1)

        # Return None if no title is found
        return None

    @staticmethod
    def parse_datetime(datetime_str):
        # a function to parse the datetime string from the tweet
        return datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=pytz.UTC)
    
    def get_api_response(self, tweets):
    def get_api_response(self, tweets, retrieved_rag_tweets=None):
        # a function to get the api response from Ollama Llama 3
        intro = (
            "Generate a comprehensive news report based on the 'Original Tweets' provided below, focusing on the common theme or event they describe. "
            "Provide a descriptive title for the report. Include relevant information from each tweet, establish connections between them, and create a cohesive narrative. "
            "Your news report should cover key details such as location, time, impact, and any additional relevant information. "
            "Organize the report into sections: Introduction, Background, Key Developments, Reactions and Responses, and Conclusion, using proper markdown format. "
            "Ensure the report is clear, concise, and captures the essence of the tweets in a coherent manner. Do not directly reference 'Tweet X' or 'Context Document Y' in your report. "
            "Use the 'Additional Retrieved Context' to enrich the report, provide more depth, and verify information. "
            "Crucially: If the 'Additional Retrieved Context' presents information that directly conflicts with or disputes claims made in the 'Original Tweets', "
            "clearly state that the claim is disputed or that there is conflicting information. Do not simply ignore or dismiss one set of information. Instead, present the differing perspectives within the narrative."
        )
        
        original_tweets_section = "Original Tweets:\n"
        for index, tweet in enumerate(tweets):
            original_tweets_section += f"Tweet {index + 1}: {tweet}\n"
        
        rag_context_section = "\nAdditional Retrieved Context:\n"
        if retrieved_rag_tweets:
            for i, rag_tweet_text in enumerate(retrieved_rag_tweets):
                rag_context_section += f"Context Document {i + 1}: {rag_tweet_text}\n"
        else:
            rag_context_section += "No additional context retrieved.\n"
        
        prompt = '\n'.join([intro, original_tweets_section, rag_context_section])

        try:
            response = ollama.generate(
                model='llama3:8b', 
                prompt=prompt,
                stream=False
            )
            generated_text = response['response']
            return generated_text
        except Exception as e:
            print(f"Error interacting with Ollama: {e}")
            # Return a placeholder or raise the exception, depending on desired error handling
            return "Error: Could not generate report via Ollama."
    
    def gen_topic_report(self, topic_label, topic_status):
        # a function to generate a news report for a certian cluster topic     
        matching_documents = list(self.tweet_filtered.find({"topic_label": topic_label})) # Convert cursor to list
        cleaned_tweets = []
        tweet_dates = []
        
        if not matching_documents:
            print(f"No documents found for topic_label: {topic_label}. Skipping report generation.")
            return None

        for tweet in matching_documents:
            tweet_text = tweet['text']
            cleaned_tweet = self.clean_tweets(tweet_text)
            cleaned_tweets.append(cleaned_tweet)
            tweet_dates.append(tweet['tweet_created_at'])

        # RAG Retrieval from ChromaDB
        retrieved_rag_tweets = []
        if self.embedding_model and cleaned_tweets:
            query_text = " ".join(cleaned_tweets[:5]) # Use first 5 cleaned original tweets as query
            try:
                query_embedding = self.embedding_model.encode(query_text).tolist()
                rag_results = self.rag_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5,
                    include=["documents"] 
                )
                if rag_results and rag_results['documents'] and rag_results['documents'][0]:
                    retrieved_rag_tweets = [doc for doc in rag_results['documents'][0]]
                print(f"Retrieved {len(retrieved_rag_tweets)} documents from RAG for topic {topic_label}.")
            except Exception as e:
                print(f"Error during RAG retrieval for topic {topic_label}: {e}")
        elif not self.embedding_model:
            print("Skipping RAG retrieval as embedding model is not loaded.")


        size = topic_status['size']
        created_at = datetime.utcnow()
        
        topic_recent_date = max(tweet_dates) if tweet_dates else datetime.utcnow()
        topic_oldest_date = min(tweet_dates) if tweet_dates else datetime.utcnow()

        summary = self.get_api_response(cleaned_tweets, retrieved_rag_tweets=retrieved_rag_tweets)
        title = self.extract_title(summary)

        # classify news summary category
        # Ensure NewsClassifier is initialized only if summary is generated
        categories = ["Uncategorized"]
        priority = "Not Set"

        if summary:
            try:
                news_category_classifier = NewsClassifier(task = 'news_category')
                cat_output = news_category_classifier.classify([summary])
                categories = news_category_classifier.process_output(cat_output)[0]

                news_priority_classifier = NewsClassifier(task = 'news_priority')
                prior_output = news_priority_classifier.classify([summary])
                priority = news_priority_classifier.process_output(prior_output)[0]
            except Exception as e:
                print(f"Error during summary classification for topic {topic_label}: {e}")
        else:
            print(f"Skipping summary classification for topic {topic_label} as summary is empty.")


        topic_report = { 'topic_label': topic_label,
                        'title': title if title else f"Report for Topic {topic_label}", 
                        'summary': summary,
                        'size': size,
                        'categories':categories,
                        'priority': priority,
                        'created_at': created_at,
                        'topic_recent_date': topic_recent_date, 
                        'topic_oldest_date': topic_oldest_date
                        }
        print(f'News report generated for topic: {topic_label}\n')
        return topic_report

    def run_report_generation(self):
        # Retieve the topic_labels and corresponding size and timestamps from the database
        status_docs = self.topic_status_change.find()
        status_array = []
        for doc in status_docs:
            topic_label = doc["topic_label"]
            size = doc["size"][-1]
            timestamp = doc["time"][-1]
            topic_status = {'topic_label': topic_label, 'size': size, 'timestamp': timestamp}
            status_array.append(topic_status)

        for topic_status in status_array:
            # check if the topic is already summarized
            topic_report = self.news_report.find_one({"topic_label": topic_status['topic_label']})

            if topic_report:
                # check if size increased by at least 10 new tweets
                if topic_report['size'] <= topic_status['size'] + 10:
                    continue
                else:
                    # generate new modified summary
                    topic_report = self.gen_topic_report(topic_status['topic_label'], topic_status)
                    insert_result = self.news_report.insert_one(topic_report)

            elif topic_status['size'] >= 10:
                # check if the cluster size is at least 10
                # generate new summary
                topic_report = self.gen_topic_report(topic_status['topic_label'], topic_status)
                insert_result = self.news_report.insert_one(topic_report)
            else: 
                continue
            print(f'Generated Summary for topic: {topic_status["topic_label"]}\n')
        return
    
    def main(self):
        self.run_report_generation()
        print('Reports Generation Done Successfully')


if __name__ == "__main__":
    generator = NewsReportGenerator()
    generator.main()