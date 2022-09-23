import pandas as pd
import cohere
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', None)

# Get the SST2 training and test sets
df = pd.read_csv('./dataset/SPAM text message 20170820 - Data.csv')
df.head()

num_examples = 500
df_sample = df.sample(num_examples)

# Split into training and testing sets
sentences_train, sentences_test, labels_train, labels_test = train_test_split(
            list(df_sample.values[:, 1]), list(df_sample.values[:, 0]), test_size=0.25, random_state=0)


# ADD YOUR API KEY HERE
api_key = "RIa4jKwZOinD3oKb2ABPFt9PPtMsiMXH7NVd3lng"

# Create and retrieve a Cohere API key from os.cohere.ai
co = cohere.Client(api_key)

# Embed the training set
embeddings_train = co.embed(texts=sentences_train,
                             model="small",
                             truncate="LEFT").embeddings
# Embed the testing set
embeddings_test = co.embed(texts=sentences_test,
                             model="small",
                             truncate="LEFT").embeddings

print(f"Review text: {sentences_train[0]}")
print(f"Embedding vector: {embeddings_train[0][:10]}")

# import SVM classifier code
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


svm_classifier = make_pipeline(StandardScaler(), SVC(class_weight='balanced'))

# fit the support vector machine
svm_classifier.fit(embeddings_train, labels_train)

# get the score from the test set, and print it out to screen!
score = svm_classifier.score(embeddings_test, labels_test)
print(f"Validation accuracy on Large is {100*score}%!")


# results:
# Review text: How long has it been since you screamed, princess?
# Embedding vector: [-1.8313706, -3.7380753, 0.03112517, -0.62801707, -0.9700512, 2.455242, 2.2599323, 0.39281273, 1.0931463, 0.6162281]
# Validation accuracy on Large is 100.0%!