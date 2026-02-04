from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Sample dataset: reviews and labels

reviews = [
    ("I loved this movie, it was fantastic!", "positive"),
    ("Amazing storyline and great acting!", "positive"),
    ("I hated this movie, it was boring.", "negative"),
    ("Terrible plot, I wasted my time.", "negative"),
    ("It was okay, not the best but enjoyable.", "positive"),
    ("Not good, I did not like it.", "negative")
]


# Split dataset into texts and labels

texts = [review[0] for review in reviews]
labels = [review[1] for review in reviews]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)




vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)




y_pred = classifier.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy on test data: {accuracy*100:.2f}%")


print("\n Welcome to your Movie Review Sentiment Analyzer!")
while True:
    review_input = input("\nEnter a movie review (or 'exit' to quit): ")
    if review_input.lower() == "exit":
        print("Goodbye! Thanks for testing the Sentiment Analyzer ")
        break
    review_vector = vectorizer.transform([review_input])
    prediction = classifier.predict(review_vector)[0]
    print(f"Sentiment prediction: {prediction.upper()}")