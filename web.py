import numpy as np
import flask
import pickle
from flask import Flask, render_template, request, Markup, url_for
from nltk import word_tokenize
import stopword
from nltk.stem import PorterStemmer
import time
from shutil import copyfile
from difflib import SequenceMatcher
from selenium import webdriver
from nltk.stem import WordNetLemmatizer
import os
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageSequence
import tensorflow as tf


os.environ['KMP_DUPLICATE_LIB_OK']='True'
global graph
graph = tf.get_default_graph()


wordnet_lemmatizer = WordNetLemmatizer()
# # CONSTANTS
SIGN_PATH = "/Users/user/Desktop/wchhack"
DOWNLOAD_WAIT = 10
SIMILIARITY_RATIO = 0.9

contractions={"what's":" what is",
"can't": "can not",
"he's": "he is",
"it's": "it is",
"doesn't": "does not",
"don't": "do not",
"let's": "let us",
"you're": "you are",
"+": "plus",
"-": "minus",
"/": "divide",
"*": "multiply",
"i'm":"i am"}


labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                   'Z':25,'space':26,'del':27,'nothing':28}








app = Flask(__name__)
im = Image.open("static/demo.gif")
ch = 'a'
index = 1
for frame in ImageSequence.Iterator(im):
    frame.save("static/%s.png" % ch)
    ch = chr(ord(ch) + 1)

size = 64,64


global model
model = load_model('sign_detection.h5')


@app.route('/')
# @app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route("/audiotoalexa/", methods=["GET", "POST"])
def move_forward():
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say Something")
        audio = r.listen(source)

    text = r.recognize_google(audio)
    text=str(text)
    print("Google thinks you said:\n" ,text )
    words = process_text(text)
        # print (words)
        # Download words that have not been downloaded in previous sessions.
    real_words = []
    for w in words:
        real_name = find_in_db(w)
        if real_name:
            print(w + " is already in db as " + real_name)
            real_words.append(real_name)
        else:
            real_words.append(download_word_sign(w))
    words = real_words
    # Concatenate videos and save output video to folder
    merge_signs(words)

    import cv2
    cap = SIGN_PATH + "/static/out.mp4"

    # while(1):
    #    ret, frame = cap.read()
    #    # cv2.imshow('frame',frame)
    #    if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
    #        cap.release()
    #        cv2.destroyAllWindows()
    #        break
    #    cv2.imshow('frame',frame)
    #    cv2.waitKey(10)
    # cap.release()
    # cv2.destroyAllWindows()
    # time.sleep(1)
    return flask.render_template('forward.html', message=text)


# Get words
def download_word_sign(word):
    browser = webdriver.Chrome("/Applications/Google Chrome.app")
    browser.get("http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi")
    first_letter = word[0]
    letters = browser.find_elements_by_xpath('//a[@class="sideNavBarUnselectedText"]')
    for letter in letters:
        if first_letter == str(letter.text).strip().lower():
            letter.click()
            break

    # Show drop down menu ( Spinner )
    spinner = browser.find_elements_by_xpath("//option")
    best_score = -1.
    closest_word_item = None
    for item in spinner:
        item_text = item.text
        # if stem == str(item_text).lower()[:len(stem)]:
        s = similar(word, str(item_text).lower())
        if s > best_score:
            best_score = s
            closest_word_item = item
            print(word, " ", str(item_text).lower())
            print("Score: " + str(s))
    if best_score < SIMILIARITY_RATIO:
        print(word + " not found in dictionary")
        return
    real_name = str(closest_word_item.text).lower()

    print("Downloading " + real_name + "...")
    closest_word_item.click()
    time.sleep(DOWNLOAD_WAIT)
    in_path = "/Users/user/Downloads/" +real_name + ".swf"
    out_path = SIGN_PATH + "/static/" + real_name + ".mp4"
    convert_file_format(in_path, out_path)
    browser.close()
    return real_name

def convert_file_format(in_path, out_path):
    # Converts .swf filw to .mp4 file and saves new file at out_path
    from ffmpy import FFmpeg

    ff = FFmpeg(
    inputs = {in_path: None},
    outputs = {out_path: None})
    ff.run()

def get_words_in_database():
    import os
    vids = os.listdir(SIGN_PATH+"/download")
    vid_names = [v[:-4] for v in vids]
    return vid_names

def process_text(text):
    # Split sentence into words

    for word in text.split():
        if word.lower() in contractions:
            text = text.replace(word, contractions[word.lower()])
    words=word_tokenize(text)
    # Remove all meaningless words
    usefull_words = [str(w).lower() for w in words if w.lower() not in set(stopword.words())]

    for i in range(len(usefull_words)):
        usefull_words[i]=wordnet_lemmatizer.lemmatize(usefull_words[i])
        if usefull_words[i].isnumeric():
            num=list(usefull_words[i])
            del usefull_words[i]
            for j in range(len(num)):
                usefull_words.insert(i+j, num[j])

    # TODO: Add stemming to words and change search accordingly. Ex: 'talking' will yield 'talk'.
    # from nltk.stem import PorterStemmer
    # ps = PorterStemmer()
    # usefull_stems = [ps.stem(word) for word in usefull_words]
    # print("Stems: " + str(usefull_stems))

    # TODO: Create Sytnax such that the words will be in ASL order as opposed to PSE.

    return usefull_words


def merge_signs(words):
    # Write a text file containing all the paths to each video
    with open("vidlist.txt", 'w') as f:
        for w in words:
            if w:
                f.write("file '" + SIGN_PATH + "/download/" + w + ".mp4'\n")
    command = "ffmpeg -f concat -safe 0 -i vidlist.txt -c copy output.mp4 -y"
    import shlex
    # Splits the command into pieces in order to feed the command line
    args = shlex.split(command)
    import subprocess
    process = subprocess.Popen(args)
    process.wait() # Block code until process is complete
    copyfile("output.mp4",SIGN_PATH + "/static/out.mp4") # copyfile(src, dst)
    # remove the temporary file (it used to ask me if it should override previous file).
    import os
    os.remove("output.mp4")

def in_database(w):
    db_list = get_words_in_database()
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    s = ps.stem(w)
    for word in db_list:
        if s == word[:len(s)]:
            return True
    return False


def similar(a, b):
    # Returns a decimal representing the similiarity between the two strings.
    return SequenceMatcher(None, a, b).ratio()

def find_in_db(w):
    best_score = -1.
    best_vid_name = None
    for v in get_words_in_database():
        s = similar(w, v)
        if best_score < s:
            best_score =  s
            best_vid_name = v
    if best_score > SIMILIARITY_RATIO:
        return best_vid_name

global gif_to_text

@app.route("/predicttext/",methods=['GET','POST'])
def predicttext():
    print('touchpoint')
    output_sentence = 'Alexa '
    for image in sorted(os.listdir('static/')):
        try:
            if image.endswith(".png"):
                temp_img = cv2.imread('static' + '/' + image)
                temp_img = cv2.resize(temp_img, size)
                test_image = np.expand_dims(temp_img, axis = 0)
                with graph.as_default():
                    result = model.predict(test_image)

                for letter, index in labels_dict.items():
                    if index == result.argmax():
                        if letter!='space':
                            output_sentence = output_sentence + letter
                        else:
                            output_sentence = output_sentence + ' '
        except Exception as e:
            pass
    output_sentence = output_sentence.replace('WHAT',"WHAT'S")
    print(output_sentence)
    gif_to_text = output_sentence
    return flask.render_template('index.html',your_prediction_appears_here=output_sentence)


@app.route("/texttospeech/",methods=['GET','POST'])
def texttospeech():
    from gtts import gTTS

    import pyglet

    import time, os



    lang = 'en'

    text  = "Alexa WHAT'S THE WEATHER"

    file = gTTS(text = text, lang = lang)
    try:
        filename = "static/temp.mp3"

        file.save(filename)



        music = pyglet.media.load(filename, streaming = False)

        music.play()



        time.sleep(music.duration)
        #
        os.remove(filename)

    except Exception:
        pass
    return flask.render_template('index.html')

app.run(host='127.0.0.1', port=5000,debug=True)
