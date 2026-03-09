import numpy as np
import pandas as pd
import emoji
from sklearn.model_selection import train_test_split
import tensorflow as tf
Adam=tf.keras.optimizers.Adam
layers=tf.keras.layers
Sequential=tf.keras.models.Sequential
l2=tf.keras.regularizers.l2
Dense=layers.Dense
Dropout=layers.Dropout
LSTM=layers.LSTM
GRU=layers.GRU
Bidirectional=layers.Bidirectional
SimpleRNN=layers.SimpleRNN
Embedding=layers.Embedding
pad_sequences=tf.keras.preprocessing.sequence.pad_sequences
Tokenizer=tf.keras.preprocessing.text.Tokenizer
to_categorical=tf.keras.utils.to_categorical

# data = pd.read_csv('emoji_data.csv', header = None)
# print(data)

# X = data[0].values
# Y = data[1].values

df = pd.read_csv('spam.csv')
df_spam = df[df['Category']=='spam']
df_spam.shape

df_ham = df[df['Category']=='ham']
df_ham.shape

df_ham_downsampled = df_ham.sample(df_spam.shape[0])
df_ham_downsampled.shape

df_balanced = pd.concat([df_ham_downsampled, df_spam])
df_balanced.shape

df_balanced['spam']=df_balanced['Category'].apply(lambda x: 1 if x=='spam' else 0)
# X_train, X_test, y_train, y_test = train_test_split(df_balanced['Message'],df_balanced['spam'], stratify=df_balanced['spam'])

emoji_dict = {
    0: ":thumbs_up:",
    1: ":thumbs_down:"
    # 2: ":grinning_face_with_big_eyes:",
    # 3: ":disappointed_face:",
    # 4: ":fork_and_knife_with_plate:"
}
def label_to_emoji(label):
    return emoji.emojize(emoji_dict[label])
# print(label_to_emoji(1))

file = open('glove.6B.100d.txt', 'r', encoding = 'utf8')
# file = open('glove.6B.300d.txt', 'r', encoding = 'utf8')
content = file.readlines()
file.close()
embeddings = {}
for line in content:
    line = line.split()
    embeddings[line[0]] = np.array(line[1:], dtype = float)

def get_maxlen(data):
    maxlen = 0
    for sent in data:
        maxlen = max(maxlen, len(sent))
    return maxlen

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Message'])
word2index = tokenizer.word_index
print(word2index)
print(len(word2index))
Xtokens = tokenizer.texts_to_sequences(df_balanced['Message'])
print(len(Xtokens))
maxlen = get_maxlen(Xtokens)
print(maxlen)
maxlen=80
Xtokens= pad_sequences(Xtokens, maxlen = maxlen,  padding = 'post', truncating = 'post')
X_train, X_test, y_train, y_test = train_test_split(Xtokens, df_balanced['spam'], stratify=df_balanced['spam'])
print(X_train.shape)
Y_train = to_categorical(np.array(y_train,dtype=int))
Y_test = to_categorical(np.array(y_test,dtype=int))

embed_size = 100
embedding_matrix = np.zeros((len(word2index)+1, embed_size))
for word, i in word2index.items():
    try:
        embed_vector = embeddings[word]
        embedding_matrix[i] = embed_vector
    except KeyError:
        continue

# model = Sequential([
#     Embedding(input_dim = len(word2index) + 1,
#               output_dim = embed_size,
#               input_length = maxlen, 
#               weights = [embedding_matrix],
#               trainable = False
#              ),
#     GRU(units = 16, return_sequences = True),
#     GRU(units = 4),
#     Dense(5, activation = 'softmax')
# ])
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# model.fit(Xtrain, Ytrain, epochs = 180)
# model.evaluate(Xtrain, Ytrain)
# model.save('RNNemojiGRU.h5')

model = Sequential([
    Embedding(input_dim = len(word2index) + 1,
              output_dim = embed_size,
              input_length = maxlen, 
              weights = [embedding_matrix],
              trainable = False
             ),
    # Dropout(0.1),
    Bidirectional(GRU(units = 50, return_sequences = True, kernel_regularizer=l2(0.001))),#, kernel_regularizer=l2(0.0001)
    GRU(units = 10), #, recurrent_dropout=0.1
    Dense(2, activation = 'softmax')
])
METRICS = [
      tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]
model.compile(optimizer = Adam(0.0002), loss = 'categorical_crossentropy', metrics = METRICS)
model.fit(X_train, Y_train, epochs = 200)
model.evaluate(X_test, Y_test)
model.save('RNNSpamBi(GRU).h5')

# model=tf.keras.models.load_model("RNNemojiGRU.h5")
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.experimental_enable_resource_variables = True
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,  # Use TensorFlow Lite built-in ops.
#     tf.lite.OpsSet.SELECT_TF_OPS,    # Enable TensorFlow ops.
# ]
# converter._experimental_lower_tensor_list_ops = False
# tflite_quant_model = converter.convert()
# with open("RNNemojiGRUquant.tflite", "wb") as f:
#     f.write(tflite_quant_model)

# # Load the TFLite model
# interpreter = tf.lite.Interpreter(model_path="RNNemojiGRUquant.tflite")
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

model=tf.keras.models.load_model('RNNSpamBi(GRU).h5')
test = ["good morning kindly check your email",
        "Call Germany for only 1 pence per minute! Call from a fixed line via access number 0844 861 85 85. No prepayment.",
        "Valentines Day Special! Win over £1000 in our quiz and take your partner on the trip of a lifetime! Send GO to 83600 now. 150p/msg rcvd. CustCare:08718720201.",
        "i will eat you for free, contact me on :911",
        "I take it the post has come then! You must have 1000s of texts now! Happy reading. My one from wiv hello caroline at the end is my favourite. Bless him",
        "Hi there, 2nights ur lucky night! Uve been invited 2 XCHAT, the Uks wildest chat! Txt CHAT to 86688 now! 150p/MsgrcvdHG/Suite342/2Lands/Row/W1J6HL LDN 18yrs",
        "Twinks, bears, scallies, skins and jocks are calling now. Don't miss the weekend's fun. Call 08712466669 at 10p/min. 2 stop texts call 08712460324(nat rate)"]
test_seq = tokenizer.texts_to_sequences(test)
Xtest = pad_sequences(test_seq, maxlen = maxlen, padding = 'post', truncating = 'post')

# print(Xtest)
y_pred = model.predict(Xtest)
print(y_pred)
y_pred = np.argmax(y_pred, axis = 1)

# Prepare input data (replace with your actual input data)
# input_data = np.array(Xtest, dtype=np.float32)  # Replace with your input shape
# interpreter.set_tensor(input_details[0]['index'], input_data)
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)
# y_pred = np.argmax(output_data, axis = 1)

for i in range(len(test)):
    # input_data = np.array(Xtest[i].reshape(1,10), dtype=np.float32)
    # interpreter.set_tensor(input_details[0]['index'], input_data)
    # interpreter.invoke()
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    # y_pred = np.argmax(output_data, axis = 1)
    print(test[i], label_to_emoji(y_pred[i]))