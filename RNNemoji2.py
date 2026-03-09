import numpy as np
import pandas as pd
import emoji
import tensorflow as tf
Adam=tf.keras.optimizers.Adam
layers=tf.keras.layers
Sequential=tf.keras.models.Sequential
GlobalAveragePooling1D=layers.GlobalAveragePooling1D
Flatten=layers.Flatten
BatchNormalization=layers.BatchNormalization
l2=tf.keras.regularizers.l2
Dense=layers.Dense
Model=tf.keras.Model 
Input=layers.Input
Dropout=layers.Dropout
LSTM=layers.LSTM
GRU=layers.GRU
Bidirectional=layers.Bidirectional
SimpleRNN=layers.SimpleRNN
Embedding=layers.Embedding
pad_sequences=tf.keras.preprocessing.sequence.pad_sequences
Tokenizer=tf.keras.preprocessing.text.Tokenizer
to_categorical=tf.keras.utils.to_categorical

df = pd.read_csv('spam.csv')
data = pd.read_csv('emoji_data.csv')
# data1 = data[data['label']==1]
# print(data1.shape)
# data0 = data[data['label']==0]#.sample(data1.shape[0])
# print(data0.shape)
# data2 = data[data['label']==2]#.sample(data1.shape[0])
# print(data2.shape)
# data3 = data[data['label']==3]#.sample(data1.shape[0])
# print(data3.shape)
# data4 = data[data['label']==4]#.sample(data1.shape[0])
# print(data4.shape)
# data= pd.concat([data0,data1,data2,data3,data4])
data = data.sample(frac=1).reset_index(drop=True)
# print(data)
X = data['sent'].values
Y = data['label'].values
data2 = data.sample(frac=1).reset_index(drop=True)
X2 = data2['sent'].values
Y2 = data2['label'].values

emoji_dict = {
    0: ":red_heart:",
    1: ":thumbs_up:",
    2: ":grinning_face_with_big_eyes:",
    3: ":disappointed_face:",
    4: ":fork_and_knife_with_plate:"
}
def label_to_emoji(label):
    return emoji.emojize(emoji_dict[label])

def get_maxlen(data):
    maxlen = 0
    for sent in data:
        maxlen = max(maxlen, len(sent))
    return maxlen

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Message'])
word2index = tokenizer.word_index
# print(word2index)
print(len(word2index))
Xtokens = tokenizer.texts_to_sequences(X)
Xtokens2 = tokenizer.texts_to_sequences(X2)
# print(Xtokens)
maxlen = get_maxlen(Xtokens)
print(maxlen)
maxlen=80
X= pad_sequences(Xtokens, maxlen = maxlen,  padding = 'post', truncating = 'post')
Y = to_categorical(np.array(Y,dtype=int))
X2= pad_sequences(Xtokens2, maxlen = maxlen,  padding = 'post', truncating = 'post')
Y2 = to_categorical(np.array(Y2,dtype=int))

model=tf.keras.models.load_model("RNNSpamBi(GRU).h5")
extractor_model = Model(inputs=model.layers[0].input, outputs=model.layers[1].output)
intermediate_outputs = extractor_model.predict(X)
print(intermediate_outputs.shape)
intermediate_outputs2 = extractor_model.predict(X2)

sequence_length = intermediate_outputs.shape[1]  # Length of the sequence=80
units = intermediate_outputs.shape[2]  # Number of units in the GRU layer=100
input_shape = (sequence_length, units)
# print(input_shape)
input_layer = Input(shape= input_shape)
x= GRU(units = 30, return_sequences = True)(input_layer)
x=Flatten()(x)
x = BatchNormalization()(x)
# Dropout(0.1)
# x = GlobalAveragePooling1D()(input_layer)
# print(x.shape)
x = Dense(300, activation='relu', kernel_regularizer=l2(0.0001))(x) 
x = BatchNormalization()(x)
# x = Dense(50, activation='relu', kernel_regularizer=l2(0.001))(x) 
# x = Dense(50, activation='relu')(x)
output_layer = Dense(5, activation='softmax')(x)

dense_model = Model(inputs=input_layer, outputs=output_layer)
dense_model.compile(optimizer= Adam(0.0002), loss='categorical_crossentropy', metrics=['accuracy'])
dense_model.fit(intermediate_outputs, Y, epochs=200)#, batch_size=32, validation_split=0.2)
dense_model.evaluate(intermediate_outputs2,Y2)
# dense_model.save('RNNemojiDense.h5')

# dense_model=tf.keras.models.load_model('RNNemojiDense.h5')
while True:
    message = input("Enter a message: ")
    test = [message]
    test_seq = tokenizer.texts_to_sequences(test)
    Xtest = pad_sequences(test_seq, maxlen = maxlen, padding = 'post', truncating = 'post')
    # print(Xtest)
    y_pred = extractor_model.predict(Xtest)
    y_pred=dense_model.predict(y_pred)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis = 1)
    print(test[0], label_to_emoji(y_pred[0]))