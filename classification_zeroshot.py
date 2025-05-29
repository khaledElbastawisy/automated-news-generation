from transformers import pipeline

class NewsClassifier:
    """
    A class to classify news articles into categories using zero-shot classification.
    """

    def __init__(self, model_name="MoritzLaurer/xtremedistil-l6-h256-mnli-fever-anli-ling-binary", task=None, device=0):
        self._validate_task(task)
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=device)
        self.task = task
        self.candidate_labels = self._get_candidate_labels(task)

    @staticmethod
    def _validate_task(task):
        if task not in ["news_category", "news_priority"]:
            raise ValueError("Invalid task parameter. Expected 'news_category' or 'news_priority'.")

    @staticmethod
    def _get_candidate_labels(task):
        if task == "news_category":
            return [
                "Health", "Science", "Environmental", "Political", "Entertainment", "Technology", "Crime", "Celebrity",
                "Economic", "Sports", "Weather", "Education", "Fashion", "Business", "Religion", "Military", "Space",
                "Automotive", "Gaming", "Social Media",
            ]
        elif task == "news_priority":
            return ["Critical", "High Priority", "Medium Priority", "Low Priority"]

    def classify(self, sequences, multi_label=False):
        if self.task == "news_category":
            multi_label = True
        return self.classifier(sequences, self.candidate_labels, multi_label=multi_label)

    def process_output(self, output, threshold=0.5):
        if self.task == "news_category":
            return self._process_output_category(output, threshold)
        elif self.task == "news_priority":
            return self._process_output_priority(output)

    @staticmethod
    def _process_output_category(output, threshold):
        processed_output = []
        for item in output:
            top_categories = []
            for i, (label, score) in enumerate(zip(item['labels'], item['scores'])):
                if i == 0:
                    top_categories.append(label)
                elif len(top_categories) < 3 and score >= threshold:
                    top_categories.append(label)

                if i == 2:
                    processed_output.append(top_categories)
                    break
        return processed_output

    @staticmethod
    def _process_output_priority(output):
        return [item["labels"][0] for item in output]

def main():
    # Example for news_category
    news_classifier_category = NewsClassifier(task="news_category")
    sequences_category = [
        "Scientists make a breakthrough in cancer treatment.",
        "The stock market hit an all-time high today.",
        "A new movie starring a famous actor was released."
    ]
    output_category = news_classifier_category.classify(sequences_category)
    processed_output_category = news_classifier_category.process_output(output_category)
    print("Processed Categories:", processed_output_category)

    # Example for news_priority
    news_classifier_priority = NewsClassifier(task="news_priority")
    sequences_priority = [
        "A minor power outage affected a small neighborhood.",
        "A multi-car pile-up has closed the northbound M1 motorway.",
        "The local library will host a book reading next week."
    ]
    output_priority = news_classifier_priority.classify(sequences_priority)
    processed_output_priority = news_classifier_priority.process_output(output_priority)
    print("Processed Priorities:", processed_output_priority)

if __name__ == "__main__":
    main()