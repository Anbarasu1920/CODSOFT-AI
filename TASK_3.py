import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, add
from tensorflow.keras.utils import to_categorical
import numpy as np

def extract_features(image_path):
    model = VGG16(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def load_captions(filename):
    captions = {}
    with open(filename, 'r') as file:
        for line in file:
            tokens = line.strip().split(',')
            img_id, img_caption = tokens[0], tokens[1]
            if img_id not in captions:
                captions[img_id] = []
            captions[img_id].append(img_caption)
    return captions

def preprocess_captions(captions):
    tokenizer = Tokenizer()
    all_captions = [cap for key in captions for cap in captions[key]]
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(cap.split()) for cap in all_captions)
    return tokenizer, max_length, vocab_size

def create_sequences(tokenizer, max_length, captions, features):
    X1, X2, y = [], [], []
    for key, caps in captions.items():
        feature = features[key][0]
        for cap in caps:
            seq = tokenizer.texts_to_sequences([cap])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=len(tokenizer.word_index) + 1)[0]
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dense(256, activation='relu')(inputs1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)
    decoder1 = add([fe1, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

captions = load_captions('captions.txt')
tokenizer, max_length, vocab_size = preprocess_captions(captions)
features = {img_id: extract_features(f'images/{img_id}.jpg') for img_id in captions.keys()}
X1, X2, y = create_sequences(tokenizer, max_length, captions, features)
model = define_model(vocab_size, max_length)
model.fit([X1, X2], y, epochs=20, verbose=2)

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

image_path = 'test.jpg'
photo = extract_features(image_path)
caption = generate_caption(model, tokenizer, photo, max_length)
print(caption)
