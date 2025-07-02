import pandas

from sklearn import dummy
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import re
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words


#read in file
df = pandas.read_csv('friends_transcripts.tsv', sep='\t')

filtered_df = df[df['speaker'].isin(["Joey Tribbiani", "Chandler Bing","Ross Geller","Phoebe Buffay","Rachel Green","Monica Geller"])]

# Now count unique scenes per speaker

new = []
for text in filtered_df['transcript']:
    string_text = str(text)
    doc = nlp(string_text)
    ents = doc.ents
    text_label = [(ent.text, ent.label_) for ent in ents]
    #print(text_label)
    
    for t, l in text_label:
        #if l == 'PERSON' or l == 'GPE':
       string_text = string_text.replace(t, " <ENT>")
    #print(text)
    lemmatized_sentence = " ".join([token.lemma_.lower() for token in doc])

    tokens = lemmatized_sentence.split()
    #print(tokens)
    #cleaned = ""
    cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens if re.sub(r'[^\w\s]', '', token)]
    no_stops_tokens = " ".join([token if token not in stopwords else "<S>" for token in cleaned_tokens])
    #print(no_stops_tokens)
    new.append(no_stops_tokens)
#print(len(new))

filtered_df["custom_tokenized_text"] = new


#print(df.franchise.value_counts())
vectorizer = feature_extraction.text.CountVectorizer(min_df=2, max_df=.45)
D = vectorizer.fit_transform(filtered_df.custom_tokenized_text)

encoder = preprocessing.LabelEncoder()
Y = encoder.fit_transform(filtered_df.speaker)

#split data into training and other data
seed = 14218
D_train, D_other, Y_train, Y_other = model_selection.train_test_split(
    D, Y, test_size=.2, random_state=seed,
)

#split remaining data into test and dev data
D_dev, D_test, Y_dev, Y_test = model_selection.train_test_split(
    D_other, Y_other, test_size=.5, random_state=seed,
)

print(f"Training size:\t\t{len(Y_train):,}")
print(f"Development size:\t{len(Y_dev):,}")
print(f"Testing size:\t\t{len(Y_test):,}")
#baseline model
dumb = dummy.DummyClassifier()
_ = dumb.fit(D_train, Y_train)

dev_acc = metrics.accuracy_score(Y_dev, dumb.predict(D_dev))
print(f"Development baseline accuracy:\t{dev_acc:.2f}")
test_acc = metrics.accuracy_score(Y_test, dumb.predict(D_test))
print(f"Testing baseline accuracy:\t{test_acc:.2f}")

#Fitting
logreg = linear_model.LogisticRegression(solver="liblinear", penalty="l1")
_ = logreg.fit(D_train, Y_train)

# Dev accuracy.
dev_acc = metrics.accuracy_score(Y_dev, logreg.predict(D_dev))
print(f"Development LR accuracy:\t{dev_acc:.4f}")

"""
for C in [.1, .2, .5, 1., 2., 5., 10., 20., 50.]:
    logreg = linear_model.LogisticRegression(
        solver="liblinear", penalty="l1", C=C
    )
    logreg.fit(D_train, Y_train)
    dev_acc = metrics.accuracy_score(Y_dev, logreg.predict(D_dev))
    print(f"C: {C}\tdevelopment LR accuracy:\t{dev_acc:.4f}")

    #logreg = linear_model.LogisticRegression(
    #solver="liblinear", penalty="l1", C=1.0
"""
logreg = linear_model.LogisticRegression(
solver="liblinear", penalty="l1", C=2.0)

logreg.fit(D_train, Y_train)
test_acc = metrics.accuracy_score(Y_test, logreg.predict(D_test))
print(f"Testing LR accuracy:\t{test_acc:.4f}")

probs = logreg.predict_proba(D_test)


for i in range(6):  # show top 5
    top_classes = np.argsort(probs[i])[::-1][:3]
    print(f"Model confidence about horoscope number {encoder.classes_[i]}:")
    for cls in top_classes:
        print(f"  {encoder.classes_[cls]} ({probs[i][cls]:.3f})")

"""
for i in range(len(encoder.classes_)):
    if encoder.classes_[i] in ("Joey Tribbiani", "Chandler Bing","Ross Geller","Phoebe Buffay","Rachel Green","Monica Geller"):
        print(f"{i}: Speaker: " + encoder.classes_[i])

vocab = np.array(vectorizer.get_feature_names_out()) # to retrieve vocab items
coefs = logreg.coef_  # shape: (n_classes, n_features)


101: Speaker: Chandler Bing
308: Speaker: Joey Tribbiani
420: Speaker: Monica Geller
487: Speaker: Phoebe Buffay
504: Speaker: Rachel Green
524: Speaker: Ross Geller

#cast = (101,308,420,487,504,524)

for i in range(len(coefs)):
    class_idx = i
    top_positive = np.argsort(coefs[class_idx])[-10:][::-1]
    top_negative = np.argsort(coefs[class_idx])[:10]

    print(f"Top tokens for {encoder.classes_[class_idx]}")

    for idx in top_positive:
        print(f"{vocab[idx]:<10} ({coefs[class_idx, idx]:.4f})")

    for idx in top_negative:
        print(f"{vocab[idx]:<10} ({coefs[class_idx, idx]:.4f})")


# Get the class names for the selected classes
Y_pred = logreg.predict(D_test)

micro_precision = metrics.precision_score(Y_test, Y_pred, average='micro')
macro_precision = metrics.precision_score(Y_test, Y_pred, average='macro')

micro_recall = metrics.recall_score(Y_test, Y_pred, average='micro')
macro_recall = metrics.recall_score(Y_test, Y_pred, average='macro')

micro_f1 = metrics.f1_score(Y_test, Y_pred, average='micro')
macro_f1 = metrics.f1_score(Y_test, Y_pred, average='macro')

print(f"Micro Precision:\t{micro_precision:.4f}")
print(f"Macro Precision:\t{macro_precision:.4f}")
print(f"Micro Recall:\t{micro_recall:.4f}")
print(f"Macro Recall:\t{macro_recall:.4f}")
print(f"Micro F1 Score:\t{micro_f1:.4f}")
print(f"Macro F1 Score:\t{macro_f1:.4f}")

# calculate the confusion matrix
conf_matrix = metrics.confusion_matrix(Y_test, Y_pred)

print("Confusion Matrix:")
print(conf_matrix)

# visualizing confusion matrix
plt.figure(figsize=(4, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
"""
