import os
import re
import time
from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer, ClassTfidfTransformer
from river import stream, cluster
from umap import UMAP
from datetime import datetime


class RiverCluster:
    def __init__(self, model):
        """
        Initialize the RiverCluster class.

        Args:
            model: A clustering model from the river library.
        """
        self.model = model

    def partial_fit(self, umap_embeddings):
        """
        Partially fit the clustering model with UMAP embeddings.

        Args:
            umap_embeddings: UMAP embeddings of the data.

        Returns:
            self: The updated instance of the RiverCluster class.
        """
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            self.model = self.model.learn_one(umap_embedding)

        labels = []
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            label = self.model.predict_one(umap_embedding)
            labels.append(label)

        self.labels_ = labels
        return self


class TopicModeling:
    def __init__(self, file_path):
        """
        Initialize the TopicModeling class.

        Args:
            file_path: The path to the file for saving the model.
        """
        self.file_path = file_path

    def clean_text(self, text):
        """
        Clean the text by removing URLs, mentions, and hashtags.

        Args:
            text: The input text.

        Returns:
            The cleaned text.
        """
        text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)
        text = re.sub("@[A-Za-z0-9_]+", "", text)
        text = re.sub("#[A-Za-z0-9_]+", "", text)
        return text

    def save_model(self):
        """
        Save the topic modeling model.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BERTopicModel(TopicModeling):
    def __init__(self, file_path):
        super().__init__(file_path)

        if os.path.isfile(file_path):
            self.model = BERTopic.load(file_path)
            print ('Clustering model loaded from file.')
        else:
            cluster_model = RiverCluster(cluster.DBSTREAM(fading_factor = 0.05))
            vectorizer_model = OnlineCountVectorizer(stop_words="english")
            ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

            self.model = BERTopic(
                hdbscan_model=cluster_model,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
                language='english',
                nr_topics='auto',
                verbose=True
            )
            print ('Clustering model initialized.')

    def online_topic_modeling(self, tweet_dict):
        """
        Perform online topic modeling on a dictionary of tweets.

        Args:
            tweet_dict: A list of dictionaries containing tweet information.

        Returns:
            A tuple containing the updated tweet dictionary with assigned topic labels and a dictionary of topic statistics.
        """
        textlist = [self.clean_text(d['text']) for d in tweet_dict]
        self.model.partial_fit(textlist)

        topic_keywords = {k: v.split('_')[1:] for k, v in self.model.topic_labels_.items()}
        freq_df = self.model.get_topic_freq().loc[self.model.get_topic_freq().Topic != -1, :]
        topics = sorted(freq_df.Topic.to_list())
        all_topics = sorted(list(self.model.get_topics().keys()))
        indices = [all_topics.index(topic) for topic in topics]

        embeddings = [self.model.topic_embeddings_[i] for i in indices]
        embeddings = UMAP(n_neighbors=2, n_components=2, metric='cosine', random_state=42).fit_transform(embeddings)
        topic_coordinates = {i: embeddings[i].tolist() for i in indices}

        topic_sizes = dict(sorted(self.model.topic_sizes_.items()))

        #current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        current_time = datetime.utcnow()

        topic_list = [{'topic_label': key, 'size': topic_sizes[key], 'coordinates': topic_coordinates[key], 'keywords': topic_keywords[key]} for key in topic_keywords]
        topic_stats = {'time': current_time, 'topics': topic_list}

        # Store original texts before popping them
        original_tweet_texts = [d['text'] for d in tweet_dict]

        # Generate embeddings for original texts
        # Ensure the embedding model is available (it should be after _fit or _partial_fit)
        if hasattr(self.model, 'embedding_model') and self.model.embedding_model is not None:
            tweet_embeddings = self.model.embedding_model.encode(original_tweet_texts)
        else:
            # Fallback or error handling if embedding_model is not found
            # This might happen if the model was loaded without its embedding component
            # or if it's a very custom BERTopic setup.
            # For now, we'll create zero vectors as placeholders, but ideally, this should be an error.
            print("Warning: BERTopic embedding model not found. Using zero vectors for embeddings.")
            # Assuming "all-MiniLM-L6-v2" as default, its dimension is 384
            # If a different model is used, this dimension might be incorrect.
            default_embedding_dim = 384 
            try:
                if self.model.embedding_model is not None : # Check if it exists but was not used above
                    # Attempt to get embedding dimension from the model if possible
                    # This is a best-effort guess; specific model APIs vary.
                    if hasattr(self.model.embedding_model, 'get_sentence_embedding_dimension'):
                        default_embedding_dim = self.model.embedding_model.get_sentence_embedding_dimension()
                    elif hasattr(self.model.embedding_model, 'model') and hasattr(self.model.embedding_model.model, 'get_sentence_embedding_dimension'): # For SentenceTransformer directly
                        default_embedding_dim = self.model.embedding_model.model.get_sentence_embedding_dimension()

            except Exception:
                pass # Stick to default 384 if introspection fails
            
            tweet_embeddings = [([0.0] * default_embedding_dim) for _ in original_tweet_texts]


        processed_tweets_data = []
        for i, d_input in enumerate(tweet_dict):
            # Assign topic label
            # self.model.topics_ contains the topic for each document processed in the last partial_fit call
            # The order should correspond to the input order of textlist
            topic_label = self.model.topics_[i] if i < len(self.model.topics_) else -2 # -2 for error/unknown

            processed_tweets_data.append({
                "tweet_id": d_input['_id'], 
                "text": original_tweet_texts[i],
                "topic_label": topic_label,
                "embedding": tweet_embeddings[i] 
            })
            # d_input.pop('text', None) # Remove original text from dict that might be saved to mongo via tweet_id_label if it's the same dict

        # The original tweet_dict (which is tweet_id_label in tweet_cluster.py) needs topic_label for MongoDB update
        # Ensure the original tweet_dict items are updated if they are the same objects
        # that will be used later for MongoDB updates.
        # The current structure of online_topic_modeling modifies tweet_dict items by reference for topic_label.
        # Let's make sure this happens correctly.
        for i, d_original in enumerate(tweet_dict):
            d_original['topic_label'] = self.model.topics_[i] if i < len(self.model.topics_) else -2
            # We no longer pop 'text' here as 'processed_tweets_data' holds the full data for RAG
            # and 'tweet_dict' (tweet_id_label) is used for MongoDB updates which might still need 'text' or other fields.
            # If 'text' is not needed for MongoDB update, it can be popped.
            # For safety, let's assume it might be needed for now or that popping it is handled by the caller if necessary.


        return processed_tweets_data, topic_stats

    def save_model(self):
        """
        Save the BERTopic model.
        """
        self.model.save(self.file_path)
