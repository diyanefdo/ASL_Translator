#!/usr/bin/env python
import PySimpleGUI as sg
import cv2
import numpy as np
import os
import mediapipe as mp
from generate_csv import get_connections_list, get_distance
from tensorflow import keras
import pandas as pd
import time
import numpy as np
import textdistance
import re
from collections import Counter
import pickle5 as pickle
from openvino.inference_engine import IECore, IENetwork
import subprocess
import stanfordnlp
from operator import itemgetter, attrgetter, methodcaller
import speech_recognition as sr

model_xml = "tf_model.xml"
model_bin = "tf_model.bin"

ie = IECore()
net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
exec_net = ie.load_network(network=net, device_name='MYRIAD')
nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', treebank='en_ewt', use_gpu=False, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size
"""
Demo program that displays a webcam using OpenCV
"""


def main():

    sg.theme('Black')

    # define the window layout
    layout = [[sg.Text('ASL Translator', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='stock.png', key='stock pic', size=(327, 346))],  
              [sg.Image(filename='', key='image')],
              [sg.Text('', size=(40,1), justification='center', font='Helvetica 20', key='prediction')],
              [sg.Button('ASL->English', size=(10, 1), font='Helvetica 14'),
               sg.Button('English->ASL', size=(10, 1), font='Any 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ]]

    # create the window and show it without the plot
    window = sg.Window('ASL Translator application',
                       layout, size=(640, 480), element_justification="center", finalize=True)
    window.Maximize()	

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    #cap = cv2.VideoCapture(0)
    ASL_English = False
    English_ASL = False

    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'ASL->English':
            ASL_English = True
            English_ASL = False

        elif event == 'English->ASL':
            ASL_English = False
            English_ASL = True

        if ASL_English:
            #os.system('python3 real_time_prediction.py')
            # ret, frame = cap.read()
            # imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            # window['image'].update(data=imgbytes)
            window['prediction'].update('')
            window['stock pic'].update(data='')
            probs = {}
            with open('textProbs.pkl', 'rb') as f:
                probs = pickle.load(f)

            word_freq = {}
            with open('wordFreq.pkl', 'rb') as f:
                word_freq = pickle.load(f)

            V = {}
            with open('V.pkl', 'rb') as f:
                V = pickle.load(f)

            def my_autocorrect(input_word):
                input_word = input_word.lower()
                if len(input_word) == 1:
                    return input_word
                if input_word in V:
                    return input_word
                else:
                    sim = [1-(textdistance.Jaccard(qval=2).distance(v,input_word)) for v in word_freq.keys()]
                    df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
                    df = df.rename(columns={'index':'Word', 0:'Prob'})
                    df['Similarity'] = sim
                    output = df.sort_values(['Similarity', 'Prob'], ascending=False)
                    return output['Word'].iloc[0]


            def get_sign_list():
                # Function to get all the values in SIGN column
                df = pd.read_csv('connections.csv', index_col=0)
                return df['SIGN'].unique()

            sign_list = get_sign_list()
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands
            connections_dict = get_connections_list()

            # Initialize webcam
            # Default is zero, try changing value if it doesn't work
            cap = cv2.VideoCapture(0)

            with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
                last_letter = ''
                start_time = time.time()
                sentence = ''
                word = ''

                while cap.isOpened():
                    # Get image from webcam, change color channels and flip
                    ret, frame = cap.read()
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = cv2.flip(image, 1)

                    # Get result
                    results = hands.process(image)
                    if not results.multi_hand_landmarks:
                        # If no hand detected, then just display the webcam frame
                        height, width, layers = frame.shape
                        new_h = height / 1.5
                        new_w = width / 1.5
                        resize = cv2.resize(frame, (int(new_w), int(new_h)))
                        imgbytes = cv2.imencode('.png', resize)[1].tobytes()  # ditto
                        window['image'].update(data=imgbytes)
                        cv2.imshow(
                            'Sign Language Detection',
                            frame
                        )
                        if word != '':
                            #word = my_autocorrect(word)
                            if len(sentence) > 20:
                                sentence = ''
                            if sentence == '':
                                sentence = word
                            else:
                                sentence = sentence + ' ' + word
                            word = ''
                            print(sentence)
                            window['prediction'].update(sentence)
                    else:
                        # If hand detected, superimpose landmarks and default connections
                        mp_drawing.draw_landmarks(
                            image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                        )

                        # Get landmark coordinates & calculate length of connections
                        coordinates = results.multi_hand_landmarks[0].landmark
                        data = []
                        for _, values in connections_dict.items():
                            data.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
                        
                        # Scale data
                        data = np.array([data])
                        data[0] /= data[0].max()
                        
                        # Load model from h5 file
                        #model = keras.models.load_model('ann_model.h5')
                        res = exec_net.infer(inputs={input_blob: data})
                        # Get prediction
                        #pred = np.array(model(data))
                        pred = np.array(res[out_blob])
                        pred = sign_list[pred.argmax()]
                        pred = pred.lower()
                        if pred == last_letter:
                            if (time.time() - start_time) >= 1:
                                start_time = time.time()
                                if word == '':
                                    word = word + pred
                                    print(word)
                                elif word[-1] != pred:
                                    word = word + pred
                                    print(word)
                        else:
                            start_time = time.time()
                        last_letter = pred

                        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                        # Display text showing prediction
                        image = cv2.putText(
                            image, pred, (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, 
                            (255, 0, 0), 2
                        )

                        # Display final image
                        height, width, layers = image.shape
                        new_h = height / 1.5
                        new_w = width / 1.5
                        resize = cv2.resize(image, (int(new_w), int(new_h)))
                        imgbytes = cv2.imencode('.png', resize)[1].tobytes()  # ditto
                        window['image'].update(data=imgbytes)
                        cv2.imshow(
                            'Sign Language Detection',
                            image
                        )
                        
                    event, values = window.read(timeout=20)
                    if event == 'Exit' or event == sg.WIN_CLOSED:
                        return
                    elif event == 'English->ASL':
                        ASL_English = False
                        English_ASL = True
                        window['image'].update(data='')
                        break
                
                cap.release()
                cv2.destroyAllWindows()

        if English_ASL:

            window['stock pic'].update(data='')
            window.read(timeout=20)
            # Download models on first run
            # stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
            # Sets up a neural pipeline in English


            def getSpeech():
                r=sr.Recognizer()
                r.energy_threshold = 500
                text = ''
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source)
                    print("Say anything : ")
                    window['prediction'].update("Say anything")
                    event, values = window.read(timeout=20)
                    audio= r.listen(source)
                    try:
                        text = r.recognize_google(audio)
                        print("You said  :  "+text)
                        window['prediction'].update(text)
                    except:
                        print("sorry, could not recognise")
                result = text
                return result

            def parse(text):
                # Process text input
                doc = nlp(text) # Run the pipeline on text input

                for sentence in doc.sentences:
                
                    translation = translate(sentence)

                    result = []
                    for word in translation[0]:
                        result.append((word['text'].lower(), word['lemma'].lower()))
                    print("\nResult: ", result, "\n")
                    display(translation)

                return doc

            def wordToDictionary(word):
                dictionary = {
                    'index': word.index,
                    'governor': word.governor,
                    'text': word.text.lower(),
                    'lemma': word.lemma.lower(),
                    'upos': word.upos,
                    'xpos': word.xpos,
                    'dependency_relation': word.dependency_relation,
                    'feats': word.dependency_relation,
                    'children': []
                }
                return dictionary


            def getMeta(sentence):
                # sentence.print_dependencies()
                englishStruct = {}
                aslStruct = {
                    'rootElements':[],
                    'UPOS': {
                    'ADJ':[], 'ADP':[], 'ADV':[], 'AUX':[], 'CCONJ':[], 'DET':[], 'INTJ':[], 'NOUN':[], 'NUM':[], 'PART':[], 'PRON':[], 'PROPN':[], 'PUNCT':[], 'SCONJ':[], 'SYM':[], 'VERB':[], 'X':[]
                    }
                }
                reordered = []
                # Make a list of all tokenized words. This step might be unnecessary.
                words = []
                for token in sentence.tokens:
                    # print(token)
                    for word in token.words:
                    
                        print(word.index, word.governor, word.text, word.lemma, word.upos, word.dependency_relation) # , word.feats)
                    # # Insert as dict
                    # words.append(wordToDictionary(word))
                    # Insertion sort
                        j = len(words)
                        for i, w in enumerate(words):
                            if word.governor <= w['governor']:
                                continue
                            else:
                                j = i
                            break
                    # Convert to Python native structure when inserting.
                        words.insert(j, wordToDictionary(word))
                reordered = words
                return reordered

            def getLemmaSequence(meta):
                tone = ""
                translation = []
                for word in meta:
                    # Remove blacklisted words
                    if (word['text'].lower(), word['lemma'].lower()) not in (('is', 'be'), ('the', 'the'), ('of', 'the'), ('is', 'are'), ('by', 'by'), (',', ','), (';', ';'), (':'), (':')):
                    
                    # Get Tone: get the sentence's tone from the punctuation
                        if word['upos'] == 'PUNCT':
                            if word['lemma'] == "?":
                                tone = "?"
                            elif word['lemma'] == "!":
                                tone = "!"
                            else:
                                tone = ""
                            continue
                    
                    # Remove symbols and the unknown
                        elif word['upos'] == 'SYM' or word['upos'] == 'X':
                            continue
                    
                    # Remove particles
                        if word['upos'] == 'PART':
                            continue

                    # Convert proper nouns to finger spell
                        elif word['upos'] == 'PROPN':
                            fingerSpell = []
                            for letter in word['text'].lower():
                                print(letter)
                                spell = {}
                                spell['text'] = letter
                                spell['lemma'] = letter
                                # Add fingerspell as individual lemmas
                                fingerSpell.append(spell)
                            print(fingerSpell)
                            translation.extend(fingerSpell)
                            print(translation)

                    # Numbers
                        elif word['upos'] == 'NUM':
                            fingerSpell = []
                            for letter in word['text'].lower():
                                spell = {}
                        # Convert number to fingerspell
                                pass
                        # Add fingerspell as individual lemmas
                                fingerSpell.append(spell)

                    # Interjections usually use alternative or special set of signs
                        elif word['upos'] == 'CCONJ':
                            translation.append(word)
                    
                    # Interjections usually use alternative or special set of signs
                        elif word['upos'] == 'SCONJ':
                            if (word['text'].lower(), word['lemma'].lower() not in (('that', 'that'))):
                                translation.append(word)
                    
                    # Interjections usually use alternative or special set of signs
                        elif word['upos'] == 'INTJ':
                            translation.append(word)

                    # Adpositions could modify nouns
                        elif word['upos']=='ADP':
                        # translation.append(word)
                            pass

                    # Determinants modify noun intensity
                        elif word['upos']=='DET':
                            pass

                    # Adjectives modify nouns and verbs
                        elif word['upos']=='ADJ':
                            translation.append(word)
                        # pass

                    # Pronouns
                        elif word['upos'] == 'PRON' and word['dependency_relation'] not in ('nsubj'):
                            translation.append(word)

                    # Nouns
                        elif word['upos'] == 'NOUN':
                            translation.append(word)

                    # Adverbs modify verbs, leave for wh questions
                        elif word['upos']=='ADV':
                            translation.append(word)
                    
                        elif word['upos']=='AUX':
                            pass

                    # Verbs
                        elif word['upos']=='VERB':
                            translation.append(word)

                # translation = tree
                return (translation, tone)

            def translate(parse):
                meta = getMeta(parse)
                translation = getLemmaSequence(meta)
                return translation

            def display(translation):
                folder = os.getcwd()
                filePrefix = folder + "/videos/"
                # Alter ASL lemmas to match sign's file names.
                # In production, these paths would be stored at the dictionary's database.
                for word in translation[0]:
                    if os.path.exists(filePrefix + word['text'].lower() + ".mp4"):
                        capASL = cv2.VideoCapture(filePrefix + word['text'].lower() + ".mp4")
                        # Check if camera opened successfully
                        if (capASL.isOpened()== False): 
                            print("Error opening video  file")
                        while(capASL.isOpened()):      
                            # Capture frame-by-frame
                            ret, frame = capASL.read()
                            if ret == True:
                                height, width, layers = frame.shape
                                aspectRatio = width / height
                                new_h = 320
                                new_w = aspectRatio * 320
                                resize = cv2.resize(frame, (int(new_w), int(new_h)))
                                imgbytes = cv2.imencode('.png', resize)[1].tobytes()  # ditto
                                window['image'].update(data=imgbytes)
                                event, values = window.read(timeout=20)
                            else:
                                break

                # Run video sequence using the MLT Multimedia Framework


            flag = False

            while not flag:
                # Get text
                tests = [
                # "Where is the bathroom?",
                # "What is your name?",
                # "I'm Javier.",
                # "My name is Javier.",
                # "Bring your computer!",
                # "It's lunchtime!",
                # "Small dogs are cute",
                # "Chihuahuas are cute because they're small."
                ]

                if len(tests) == 0:
                    tests = tests + [ getSpeech() ]

                if len(tests[0]) == 0:
                    print("No speech detected... Reattempting.")
                else:
                    for text in tests:
                        print("Text to process: ", text, "\n")

                        parse(text)

                        flag = True
                    window['image'].update(data='')
                    window['stock pic'].update(filename='stock.png', size=(327, 346))
                    window['prediction'].update('')
                    ASL_English = False
                    English_ASL = False
                    event, values = window.read(timeout=20)
                    break   


main()
