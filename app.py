from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import uuid  # For generating unique filenames
import traceback

app = Flask(__name__, static_url_path='/static')

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Dictionary mapping product IDs to dataset files
product_datasets = {
    'product1': 'iphone11_reviews_sentiments.csv',
    'product2': 'alexa_data.csv',
    'product3': 'Mamaearth.csv',
    'product4': 'macbook.csv',
    'product5': 'lg_refrigerator_reviews.csv',
    'product6': 'maybelline.csv',
    'product7': 'tetley-tea.csv',
    'product8': 'himalaya.csv' # Add more mappings as needed
}

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/electronics')
def electronics():
    return render_template('electronics.html')

@app.route('/beauty')
def beauty():
    return render_template('beauty.html')

@app.route('/home_products')
def home_products():
    return render_template('home_products.html')

@app.route('/run_code', methods=['POST'])
def run_code():
    data = request.json
    product_id = data['product_id']
    product_name = data['product_name']
    product_image = data['product_image']

    # Get the dataset file for the given product ID
    dataset_file = product_datasets.get(product_id)
    if not dataset_file or not os.path.exists(dataset_file):
        return jsonify({"message": "Dataset not found."})

    # Load the dataset
    df = pd.read_csv(dataset_file)

    # Preprocess the text data
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = text.lower()  # Convert to lowercase
        text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
        return text

    # Ensure all entries in 'review_text' are strings
    df['review_text'] = df['review_text'].astype(str)
    df['review'] = df['review_text'].apply(preprocess_text)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    # Vectorize text data
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test_vectorized)

    # Evaluate the model
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:\n", report)

    # Extract precision values for expected labels
    precision_values = {}
    for label in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:  # Adjust labels as per your dataset
        if label in report:
            precision = report[label].get('precision', None)
            if precision is not None:
                precision_values[label] = precision

    print("Extracted Precision Values:", precision_values)

    # Ensure we have precision values for plotting
    if not precision_values:
        print("No valid precision values found. Cannot plot pie chart.")
        return jsonify({"message": "No valid precision values found."})
    else:
        # Plotting the pie chart
        labels = list(precision_values.keys())
        sizes = list(precision_values.values())

        colors = ['gold', 'lightcoral', 'lightskyblue']

        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title(f'Precision Values for Sentiment Classes - {product_name}')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Generate a unique filename for the plot
        unique_filename = f'{uuid.uuid4()}.png'
        image_path = os.path.join('static', unique_filename)
        
        # Ensure static directory exists
        if not os.path.exists('static'):
            os.makedirs('static')
        
        plt.savefig(image_path)
        plt.close()

        print(f"Image saved at: {image_path}")

        # Use url_for to create the URL for the image
        image_url = url_for('static', filename=unique_filename)

        # Render the template with the image
        response = render_template('result.html', product_name=product_name, product_image=product_image, product_id=product_id, image_file=image_url)
        
        # Return the response first
        return response

@app.route('/static/<path:filename>')
def static_files(filename):
    # Serve the file from the static directory
    return send_from_directory('static', filename)

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000)  # Replace with a valid port number
    except SystemExit as e:
        print(f"SystemExit exception occurred: {e}")
        traceback.print_exc()
 