import numpy as np
from numpy import array
import string
import pickle
import os
import glob
from pickle import load
from time import time
from keras.preprocessing import sequence
from keras.layers import LSTM, Embedding, Dense, Dropout
from nltk.translate.bleu_score import corpus_bleu
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

##Some parts of this code were written by referencing (https://github.com/hlamba28/Automatic-Image-Captioning.git)
def load_doc(filename):
	#open file as read only and read text from the file
	file=open(filename,'r')
	desc=file.read() 
	file.close()
	return desc

def caption_to_desc(desc):
	#convert descriptions to dictionary
	description=dict()

	for line in desc.split('\n'):
		token=line.split()
		if len(line)<2:
			continue
		imageId,imageDesc=token[0],token[1:]
		imageId=imageId.split('.')[0]
		imageDesc=' '.join(imageDesc)
		if imageId not in description:
			description[imageId]=list()
		description[imageId].append(imageDesc)
	return description


def clean(description):
	#create translation table for punctuation
	trantab=str.maketrans('','',string.punctuation)
	for key,desc_list in description.items():
		for i in range(len(desc_list)):
			desc=desc_list[i]
			#split into list of words
			desc=desc.split()
			#convert into lower case
			desc=[j.lower() for j in desc]
			#remove punctuation 
			desc=[j.translate(trantab) for j in desc]
			#length of word is more than 1
			desc=[w for w in desc if len(w)>1]
			#remove words with numbers in them
			desc=[w for w in desc if w.isalpha()]
			#join to form the sentence again
			desc_list[i]=' '.join(desc)
	return(description)

#create vocabulary of words of descriptions
def to_vocab(description):
	
	vocab=set()
	for key in description.keys():
		[vocab.update(w.split())for w in description[key]]
	return vocab

def save_file(description,out_filename):

	sentence=list()

	for key,desc_list in description.items():
		for des in desc_list:
			sentence.append(key+' '+des)
	desc='\n'.join(sentence)
	file=open(out_filename,'w')
	file.write(desc)
	file.close()
    
descriptions,vocab=caption_to_desc('trial_CaptionToToken.txt')
print(descriptions)
print (vocab)

filename = 'results_20130124.token.txt'
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = caption_to_desc(doc)
print('Loaded: %d ' % len(descriptions))
# clean descriptions
clean(descriptions)
# summarize vocabulary
vocabulary = to_vocab(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
# save to file
save_file(descriptions, 'descriptions.txt')

def load_imgs(filename):

    doc =load_doc(filename)
    dataset=list()
    
    for line in doc.split('\n'):

        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

filename = 'train.txt'
train = load_imgs(filename)
print('Dataset: %d' %len(train))

##train images
images = 'flickr30k-images/'
## Create a list of all image names in the directory
img = glob.glob(images + '*.jpg')

train_images_file = 'train.txt'
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

train_img = []
for i in img: 
    if i[len(images):(len(i)-4)] in train_images: 
       train_img.append(i)

filename = 'val.txt'
val = load_imgs(filename)
print('Dataset: %d' %len(val))

# ##train images
images = 'flickr30k-images/'
## Create a list of all image names in the directory
img = glob.glob(images + '*.jpg')


val_images_file = 'val.txt'
val_images = set(open(val_images_file, 'r').read().strip().split('\n'))

val_img = []
for i in img: 
    if i[len(images):(len(i)-4)] in val_images: 
       val_img.append(i)

test_images_file = 'test.txt'
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

test_img = []
for i in img: 
    if i[len(images):(len(i)-4)] in test_images: 
       test_img.append(i)

test=load_imgs(test_images_file)
print('Dataset: %d' %len(test))

def clean_desc_load(filename,dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        token = line.split()
        image_id, image_desc = token[0],token[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            
            desc = 'startseq ' + ' '.join(image_desc)+ ' endseq'
            descriptions[image_id].append(desc)
    return descriptions

descriptions_train = clean_desc_load('descriptions.txt',train)
descriptions_val = clean_desc_load('descriptions.txt',val)
descriptions_test = clean_desc_load('descriptions.txt',test)


def preprocess(path):
    image1 = image.load_img(path, target_size=(299, 299))
    x = image.img_to_array(image1)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

model = InceptionV3(weights='imagenet')

## Removing softmax layer
model_new = Model(model.input, model.layers[-2].output)

def encode(image):
    image = preprocess(image) 
    fea_vec = model_new.predict(image) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) 
    return fea_vec 

start = time()
encoding_train = {}
for img in train_img:
    encoding_train[img[len(images):]] = encode(img)
print("Time taken in seconds =", time()-start)
print(len(encoding_train))
start = time()
encoding_val = {}
for img in val_img:
    encoding_val[img[len(images):]] = encode(img)
print("Time taken in seconds =", time()-start)
print(len(encoding_val))
start = time()
encoding_test = {}
for img in test_img:
    encoding_test[img[len(images):]] = encode(img)
print("Time taken in seconds =", time()-start)
print(len(encoding_test))

with open("encoded_train_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)

with open("encoded_val_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_val, encoded_pickle)
    
with open("encoded_test_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)


train_captions = []
for key,val in descriptions_train.items():
   for caption in val:
       train_captions.append(caption)

# print(len(train_captions))

val_captions = []
for key,val in descriptions_val.items():
   for caption in val:
       val_captions.append(caption)

print(len(val_captions))

## Words which occur in descriptions more than 10 times
threshold = 10
word_count = {}
no_sentence = 0
for sentence in train_captions:
    no_sentence = no_sentence + 1
    for w in sentence.split(' '):
        word_count[w] = word_count.get(w, 0) + 1

vocab = [w for w in word_count if word_count[w] >= threshold]
print('preprocessed words %d -> %d' % (len(word_count), len(vocab)))

idxtoword = {}
wordtoidx = {}

idx = 1
for w in vocab:
    wordtoidx[w] = idx
    idxtoword[idx] = w
    idx = idx + 1
    
vocab_size = len(idxtoword) + 1 # for zero padding
print(vocab_size)

# Dictionary to list of descriptions
def to_list(descrip):
	desc = list()
	for key in descrip.keys():
		[desc.append(d) for d in descrip[key]]
	return desc

# Finding maximum length off captions for descriptions
def maximum_length(descrip):
	list1 = to_list(descrip)
	return max(len(x.split()) for x in list1)

#  Maximum length of the captions in training descriptions
max_length = maximum_length(descriptions_train)
# print('Description Length: %d' % max_length)

def data_gen(descriptions,photos,wordtoidx,max_length,photos_per_batch):
	img_feature,partial_caption,output_word=list(),list(),list()
	n=0
	while 1:
		for key,desc_all in descriptions.items():
			n+=1
			photo=photos[key+'.jpg']
			for desc in desc_all:
				#encode to sequence
				seq=[wordtoidx[word] for word in desc.split(' ') if word in wordtoidx ]

				#split the sequence for generating matrix
				for i in range(1,len(seq)):
					in_seq,out_seq=seq[:i],seq[i]	
					in_seq=pad_sequences([in_seq],maxlen=max_length)[0]
					out_seq=to_categorical([out_seq],num_classes=vocab_size)[0]
					img_feature.append(photo)
					partial_caption.append(in_seq)
					output_word.append(out_seq)
			if n==photos_per_batch:
				yield[[array(img_feature),array(partial_caption)],array(output_word)]
				# print(len(img_feature))
				# print(len(partial_caption))
				# print(len(output_word))
				img_feature,partial_caption,output_word=list(),list(),list()
				n=0

#Glove word embeddings
path_glove=os.getcwd()+'/glove.6B'
embedding={}

f=open('glove.6B.200d.txt', encoding="utf-8")
# create dictionary of embedding coefficients for each word
for line in f:
	value=line.split()
	word=value[0]
	coeff=np.asarray(value[1:],dtype='float32')
	embedding[word]=coeff
f.close()

# print('Found %s word vectors.' % len(embedding))
embed_dimension=200
embed_matrix=np.zeros((vocab_size,embed_dimension))

for word,idx in wordtoidx.items():
	embed_vec=embedding.get(word)
	if embed_vec is not None:
		embed_matrix[idx]=embed_vec


print('Size of embedding matrix for all words in vocab:',embed_matrix.shape)

train_features = load(open("encoded_train_images.pkl", "rb"))

#RNN Model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embed_dimension, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.summary()

model.layers[2].set_weights([embed_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy',optimizer ='adam')


model.summary()

#Train RNN model for  20 epochs
epochs = 20
number_pictures_in_batch = 32
steps = len(descriptions_train)//number_pictures_in_batch

for i in range(epochs):
	generator = data_gen(descriptions_train,train_features,wordtoidx,max_length,number_pictures_in_batch)
	model.fit_generator(generator,epochs=1, steps_per_epoch=steps,verbose=1)
	model.save('model_new_' + str(i) + '.h5')

#Training RNN model for further 10 epochs by reducing the learning rate
model.optimizer.lr=0.0001
epochs = 10
number_pictures_in_batch = 64
steps = len(descriptions_train)//number_pictures_in_batch
for i in range(epochs):
	generator = data_gen(descriptions_train,train_features,wordtoidx,max_length,number_pictures_in_batch)
	model.fit_generator(generator,epochs=1, steps_per_epoch=steps,verbose=1)
	model.save('model_new1_' + str(i) + '.h5')

model.save_weights('model_weights.hdf5')
model.load_weights('./model_weights_30.hdf5')

with open("./encoded_val_images.pkl","rb") as encoded_pickle:
    encoding_val =load(encoded_pickle)

def greedy_search(img):
    ip_txt = 'startseq'
    for i in range(max_length):
        seq = [wordtoidx[w] for w  in ip_txt.split() if w in wordtoidx]
        seq = pad_sequences([seq], maxlen=max_length)
        ypred = model.predict([img,seq] ,verbose=0)
        ypred = np.argmax(ypred)
        word = idxtoword[ypred]
        ip_txt = ip_txt + ' ' + word
        if word == 'endseq':
            break
    final = ip_txt.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search(image, beam_index = 3):
    start = [wordtoidx["startseq"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image, np.array(par_caps)])
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            
            #Finding the top n predictions 
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Selecting top n images
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idxtoword[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    final_caption = ' '.join(final_caption[1:])
    return final_caption

def eval_model(model,descriptions,encoding,word_count,max_length):
    actual,pred1,pred2,pred3,pred4=list(),list(),list(),list(),list()
    for key,desc_list in descriptions.items():
        print("Key: ", key)
        img=encoding[key+'.jpg'].reshape(1,2048)
        yhat1=greedy_search(img)
        yhat2=beam_search(img,beam_index=3)
        yhat3=beam_search(img,beam_index=5)
        yhat4=beam_search(img,beam_index=7)
        actual1=[d.split() for d in desc_list]
        actual.append(actual1)
        pred1.append(yhat1.split())
        pred2.append(yhat2.split())
        pred3.append(yhat3.split())
        pred4.append(yhat4.split())
        
    print('BLEU-1',corpus_bleu(actual,pred1,weights=(1.0,0,0,0)))
    print('BLEU-2',corpus_bleu(actual,pred1,weights=(0.5,0.5,0,0)))
    print('BLEU-3',corpus_bleu(actual,pred1,weights=(0.3,0.3,0.3,0,)))
    print('BLEU-4    ',corpus_bleu(actual,pred1,weights=(0.25,0.25,0.25,0.25)))
    
    print('BLEU-1',corpus_bleu(actual,pred2,weights=(1.0,0,0,0)))
    print('BLEU-2',corpus_bleu(actual,pred2,weights=(0.5,0.5,0,0)))
    print('BLEU-3',corpus_bleu(actual,pred2,weights=(0.3,0.3,0.3,0,)))
    print('BLEU-4    ',corpus_bleu(actual,pred2,weights=(0.25,0.25,0.25,0.25)))
    
    print('BLEU-1',corpus_bleu(actual,pred3,weights=(1.0,0,0,0)))
    print('BLEU-2',corpus_bleu(actual,pred3,weights=(0.5,0.5,0,0)))
    print('BLEU-3',corpus_bleu(actual,pred3,weights=(0.3,0.3,0.3,0,)))
    print('BLEU-4    ',corpus_bleu(actual,pred3,weights=(0.25,0.25,0.25,0.25)))
    
    print('BLEU-1',corpus_bleu(actual,pred4,weights=(1.0,0,0,0)))
    print('BLEU-2',corpus_bleu(actual,pred4,weights=(0.5,0.5,0,0)))
    print('BLEU-3',corpus_bleu(actual,pred4,weights=(0.3,0.3,0.3,0,)))
    print('BLEU-4    ',corpus_bleu(actual,pred4,weights=(0.25,0.25,0.25,0.25)))

eval_model(model,descriptions_test,encoding_test,word_count,max_length)
