# AI-tools-repoPart 1: Theoretical Understanding (Brief Answers)
Q1: TensorFlow vs. PyTorch

TensorFlow is a robust framework for production-grade deployment, offering static computation graphs (in earlier versions) and strong support for distributed training. PyTorch uses dynamic computation graphs, making it more intuitive for research and debugging. Choose TensorFlow for scalable, production-ready systems (e.g., mobile apps); choose PyTorch for rapid prototyping or complex research models.

Q2: Jupyter Notebooks Use Cases

Exploratory Data Analysis (EDA): Jupyter allows interactive visualization and code execution to explore datasets (e.g., plotting distributions in the Iris dataset).
Model Prototyping: It supports iterative development of ML models, enabling real-time tweaking of parameters and visualization of results (e.g., loss curves).
Q3: spaCy vs. Python String Operations

spaCy enhances NLP tasks with pre-trained models for tokenization, POS tagging, and NER, offering higher accuracy and efficiency compared to manual string operations (e.g., regex). It handles complex linguistic structures, such as sentence boundaries, with minimal code.

Comparative Analysis: Scikit-learn vs. TensorFlow

Target Applications: Scikit-learn excels in classical ML (e.g., decision trees, SVMs) and is ideal for smaller datasets. TensorFlow is designed for deep learning (e.g., CNNs, RNNs) and handles large-scale data.
Ease of Use: Scikit-learn is beginner-friendly with simple APIs (e.g., fit()/predict()). TensorFlow has a steeper learning curve due to its complexity.
Community Support: Both have strong communities, but TensorFlow’s is larger due to Google’s backing, with extensive tutorials and production tools.
Part 2: Practical Implementation
Task 1: Classical ML with Scikit-learn (Iris Dataset)
iris_decision_tree.py
python
Edit in files
•
Show inline
Explanation (for report): The script loads the Iris dataset, checks for missing values (none expected), splits the data, trains a decision tree classifier, and evaluates it using accuracy, precision, and recall. The code is commented for clarity, and results will be included in the report PDF with screenshots.

Task 2: Deep Learning with TensorFlow (MNIST Dataset)
mnist_cnn.py
python
Edit in files
•
Show inline
Explanation (for report): The script builds a CNN using TensorFlow to classify MNIST digits, achieving >95% test accuracy. It normalizes data, defines a CNN architecture with two convolutional layers, trains for 10 epochs, and visualizes predictions on five test images. Screenshots of the accuracy plot and sample predictions will be included in the report.

Task 3: NLP with spaCy (Amazon Reviews)
amazon_reviews_nlp.py
python
Edit in files
•
Show inline
Explanation (for report): The script uses spaCy for NER to extract product names/brands and TextBlob for rule-based sentiment analysis. Sample reviews are processed to demonstrate functionality, with outputs showing extracted entities and sentiment scores. In practice, a larger Amazon Reviews dataset would be used, and results will be included in the report.

Part 3: Ethics & Optimization
Ethical Considerations (for report):

MNIST Model Bias: The MNIST dataset is balanced, but biases could arise if the model overfits to certain digit styles (e.g., specific handwriting). TensorFlow Fairness Indicators can evaluate performance across subgroups (e.g., digits) to identify disparities.
Amazon Reviews Bias: Sentiment analysis may misinterpret sarcasm or cultural nuances. spaCy’s rule-based systems can be customized with domain-specific rules to improve accuracy.
Mitigation: Use fairness metrics (e.g., equal opportunity) and diverse training data to reduce biases.
Troubleshooting Challenge (Buggy Code Fix):


