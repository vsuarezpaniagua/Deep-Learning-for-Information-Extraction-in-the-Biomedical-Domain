import os
import sys
import re
from copy import deepcopy
import xml.etree.ElementTree as ET
from nltk import word_tokenize
from nltk import pos_tag
#word_tokenize = list # char-level
#from geniatagger import GeniaTagger
#tagger = GeniaTagger("../geniatagger-3.0.2/geniatagger")#tagger.parse(sentence)
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import wordnet as wn
import networkx as nx
import spacy
nlp = spacy.load("en_core_web_trf")
import numpy as np
from sklearn.metrics import confusion_matrix

# Global names for the entities
ENTITY1 = "1"
ENTITY2 = "2"
ENTITY0 = "0"
NUM = "#"

def clean_str(sentence):
    """
    Adapted from:
    https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # Keep characters, numbers and special characteres
    sentence = re.sub(r"[^A-Za-z0-9(),!?\']", " ", sentence)
    # Replace english contractions
    sentence = re.sub(r"\'m", " \'m", sentence)
    sentence = re.sub(r"\'s", " \'s", sentence)
    sentence = re.sub(r"\'re", " \'re", sentence)
    sentence = re.sub(r"\'ve", " \'ve", sentence)
    sentence = re.sub(r"\'d", " \'d", sentence)
    sentence = re.sub(r"\'ll", " \'ll", sentence)
    sentence = re.sub(r"n\'t", " n\'t", sentence)
    # Replace special characters
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    # Replace numbers
    sentence = re.sub(r"\d+", " " + NUM + " ", sentence)
    sentence = re.sub(r"(" + re.escape(NUM) + "(\s)*)+" + re.escape(NUM), " " + NUM + " ", sentence)
    #sentence = re.sub(r"(" + re.escape(NUM) + "(\s)*,(\s)*)+" + re.escape(NUM), " " + NUM + " ", sentence)
    # Replace the entities
    sentence = re.sub('entitya', ENTITY1, sentence)
    sentence = re.sub('entityb', ENTITY2, sentence)
    sentence = re.sub('entityx', ENTITY0, sentence)
    # Replace multiple spaces
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence.strip().lower() + " ."


def remove_repetition(sentence):
    """
    Remove all the "drug other" repetitions
    """
    # remove ENTITY0 repetitions
    sentence = re.sub(r"" + re.escape(ENTITY0) + " , ", ENTITY0 + " ", sentence)
    sentence = re.sub(r"(" + re.escape(ENTITY0) + " ){2,}", ENTITY0 + " ", sentence)
    sentence = re.sub(r"" + re.escape(ENTITY0) + " and " + re.escape(ENTITY0) + " ", ENTITY0 + " ", sentence)
    sentence = re.sub(r"" + re.escape(ENTITY0) + " or " + re.escape(ENTITY0) + " ", ENTITY0 + " ", sentence)
    # remove special cases
    sentence = re.sub(r"" + re.escape(ENTITY0) + " \( " + re.escape(NUM) + " \) ", ENTITY0 + " ", sentence)
    sentence = re.sub(r"e g , " + re.escape(ENTITY0) + " ", ENTITY0 + " ", sentence)
    sentence = re.sub(r"eg , " + re.escape(ENTITY0) + " ", ENTITY0 + " ", sentence)
    sentence = re.sub(r"\( " + re.escape(ENTITY0) + " \) ", ENTITY0 + " ", sentence)
    # remove ENTITY0 repetitions again
    sentence = re.sub(r"" + re.escape(ENTITY0) + " , ", ENTITY0 + " ", sentence)
    sentence = re.sub(r"(" + re.escape(ENTITY0) + " ){2,}", ENTITY0 + " ", sentence)
    sentence = re.sub(r"" + re.escape(ENTITY0) + " and " + re.escape(ENTITY0) + " ", ENTITY0 + " ", sentence)
    sentence = re.sub(r"" + re.escape(ENTITY0) + " or " + re.escape(ENTITY0) + " ", ENTITY0 + " ", sentence)
    # add a comma between ENTITIES
    sentence = re.sub(r"" + re.escape(ENTITY0) + " " + re.escape(ENTITY1) + " ", ENTITY0 + " , " + ENTITY1 + " ", sentence)
    sentence = re.sub(r"" + re.escape(ENTITY1) + " " + re.escape(ENTITY0) + " ", ENTITY1 + " , " + ENTITY0 + " ", sentence)
    sentence = re.sub(r"" + re.escape(ENTITY0) + " " + re.escape(ENTITY2) + " ", ENTITY0 + " , " + ENTITY2 + " ", sentence)
    sentence = re.sub(r"" + re.escape(ENTITY2) + " " + re.escape(ENTITY0) + " ", ENTITY2 + " , " + ENTITY0 + " ", sentence)
    sentence = re.sub(r"" + re.escape(ENTITY1) + " " + re.escape(ENTITY2) + " ", ENTITY1 + " , " + ENTITY2 + " ", sentence)
    sentence = re.sub(r"" + re.escape(ENTITY2) + " " + re.escape(ENTITY1) + " ", ENTITY2 + " , " + ENTITY1 + " ", sentence)    
    return sentence


def filter_neg(text, drug_name_e1, drug_name_e2):
    """
    Filtering negatives instances with a rule based.
    """
    filters = [
        # similar entities
        r"" + re.escape(ENTITY1) + " also known as " + re.escape(ENTITY2),
        r"" + re.escape(ENTITY1) + " also called " + re.escape(ENTITY2),
        r"" + re.escape(ENTITY1) + " \( e g ,( " + re.escape(ENTITY0) + " ,)* " + re.escape(ENTITY2),
        # coordinates
        r"" + re.escape(ENTITY1) + "( " + re.escape(ENTITY0) + ")+ " + re.escape(ENTITY2),
        #r"" + re.escape(ENTITY1) + "( " + re.escape(ENTITY0) + ")* " + re.escape(ENTITY2),
        r"" + re.escape(ENTITY1) + " ,( " + re.escape(ENTITY0) + " ,)* " + re.escape(ENTITY2),
        r"" + re.escape(ENTITY1) + " \(( " + re.escape(ENTITY0) + " ,)* " + re.escape(ENTITY2),
        # such as
        r"" + re.escape(ENTITY1) + " such as( " + re.escape(ENTITY0) + " ,)* " + re.escape(ENTITY2),
        r"" + re.escape(ENTITY1) + " such as( " + re.escape(ENTITY0) + " or)* " + re.escape(ENTITY2),
        r"" + re.escape(ENTITY1) + " such as( " + re.escape(ENTITY0) + " and)* " + re.escape(ENTITY2),
        # and
        #r"" + re.escape(ENTITY1) + " and " + re.escape(ENTITY2),
        r"" + re.escape(ENTITY1) + " ,( " + re.escape(ENTITY0) + " ,)* and " + re.escape(ENTITY2),
        r"" + re.escape(ENTITY1) + " ,( " + re.escape(ENTITY0) + " ,)* " + re.escape(ENTITY0) + " and " + re.escape(ENTITY2),
        # or
        r"" + re.escape(ENTITY1) + " or " + re.escape(ENTITY2),
        r"" + re.escape(ENTITY1) + " ,( " + re.escape(ENTITY0) + " ,)* or " + re.escape(ENTITY2),
        r"" + re.escape(ENTITY1) + " ,( " + re.escape(ENTITY0) + " ,)* " + re.escape(ENTITY0) + " or " + re.escape(ENTITY2),
        # and or
        r"" + re.escape(ENTITY1) + " and or " + re.escape(ENTITY2),
        r"" + re.escape(ENTITY1) + " ,( " + re.escape(ENTITY0) + " ,)* and or " + re.escape(ENTITY2),
        r"" + re.escape(ENTITY1) + " ,( " + re.escape(ENTITY0) + " ,)* " + re.escape(ENTITY0) + " and or " + re.escape(ENTITY2)
    ]
    return drug_name_e1.lower() == drug_name_e2.lower() or stemmer.stem(drug_name_e1) == stemmer.stem(drug_name_e2) or any([True for f in filters if re.search(f, text)])


def deoverlapping(overlapped, discarded = [], discarded_overlapped = []):
    """
    Get the all possible discarded overlapped in a list to have non-overlap
    overlapped = list of lists with overlapped [[overlap, [whom overlap]], ...]
    discarded = temporal discarded overlapped
    discarded_overlapped = complete discarded list to have a non-overlap
    """
    # all overlapped are None (No overlap) => keep the discarded
    if all(value[1] is None for value in overlapped):
        discarded.sort()
        if discarded not in discarded_overlapped:
            discarded_overlapped.append(discarded)
    # OVERLAP!
    else:
        for i in range(len(overlapped)):
            if not overlapped[i][1] == None:
                # Copy to pass-by-value
                overlapped_copy = deepcopy(overlapped)
                discarded_copy = deepcopy(discarded)
                for j in overlapped[i][1]:
                    if j not in discarded:
                        # Add to discarded the new whom overlap
                        discarded_copy.append(j)
                        # Find the index
                        for k in range(len(overlapped)):
                            if overlapped[k][0] == j:
                                break
                        # Remove the new whom overlap to continue
                        overlapped_copy[k][1] = None
                # Remove the overlap for the recursion
                overlapped_copy[i][1] = None
                discarded_overlapped = deoverlapping(overlapped_copy, discarded_copy, discarded_overlapped)
    return discarded_overlapped


def set_discon(sentence):
    """
    Set the discontinued entities to the non-overlapped offsets
    *It is assumed that the discontinued entities have an overlapped offsets with another entity
    sentence = contains the text and all the entities in the text
    """
    overlaps = []
    for e1 in sentence:
        if e1.tag=="entity":
            for e2 in sentence:
                if e2.tag=="entity":
                    if not e1.get("id") == e2.get("id"):
                        for o1 in e1.get("charOffset").split(";"):
                            o1a, o1b = list(map(int, o1.split("-")))
                            for o2 in e2.get("charOffset").split(";"):
                                o2a, o2b = list(map(int, o2.split("-")))
                                if o1a <= o2b and o2a <= o1b:
                                    if o1a < o2a:
                                        overlaps.append([o2a, o1b])
                                    elif o2b < o1b:
                                        overlaps.append([o1a, o2b])
                                    else:
                                        overlaps.append([o1a, o1b])
    for oa, ob in overlaps:
        for entity in sentence:
            if entity.tag=="entity":
                charOffset = entity.get("charOffset").split(";")
                for e in range(len(charOffset)):
                    ea, eb = list(map(int, charOffset[e].split("-")))
                    if ea <= ob and oa <= eb:
                        if ea < oa:
                            charOffset[e] = str(ea) + "-" + str(oa-1)
                        elif ob < eb:
                            charOffset[e] = str(ob+1) + "-" + str(eb)
                        else:
                            charOffset[e] = ""
                entity.set("charOffset", ";".join(c for c in charOffset if not c==""))
    return sentence


def load_data(data_path):
    """
    Load data, labels and distances from a path
    data_path = the path which contain the sentences files (in .xml format)
    """
    texts = []
    labels = []
    distances = []
    sdp = []
    pos = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.endswith(".xml"):
                for sentence in ET.parse(os.path.join(root, f)).getroot():
                    for pair in set_discon(sentence):
                        if pair.tag=="pair":
                            # Extract the text, labels and features
                            text, drug_e1, drug_e2 = replace_pair(sentence, pair)
                            texts.append(text)
                            labels.append([str(pair.get("type")), drug_e1[0], drug_e2[0], drug_e1[1], drug_e2[1], drug_e1[2], drug_e2[2], str(pair.get("id"))])
                            distances.append([word_tokenize(text).index(ENTITY1), word_tokenize(text).index(ENTITY2)])
                            if not word_tokenize == list:
                                # TODO: agregar más features lemma, steamming, chunking, dependency type, constituency type
                                pos.append(text)#(" ".join([p[1] if not (p[0] == ENTITY1 or p[0] == ENTITY2 or p[0] == ENTITY0) else 'NN' for p in pos_tag(word_tokenize(text))]))
                                sdp.append(text)#(SDP(text))
    return np.array(texts), np.array(labels), np.array(distances), np.array(pos), np.array(sdp)


def replace_pair(sentence, pair):
    """
    Load the text of a pair (e1-e2), replace all the entities and clean it
    sentence = contains the text and all the entities in the text
    pair = pair of the entities e1-e2
    """
    text = sentence.get("text")
    e1, e2 = pair.get("e1"), pair.get("e2")
    postOffset = 0
    for entity in sentence:
        if entity.tag=="entity":
            offset = [int(o) + postOffset for o in entity.get("charOffset").split("-")]
            # Drug blinding
            drug_id = entity.get("id")
            name = text[offset[0]:offset[1]+1]
            blind = (drug_id==e1)*" entitya " + (drug_id==e2)*" entityb " + (not drug_id==e1 and not drug_id==e2)*" entityx "#name
            text = text[:offset[0]] + blind + text[offset[1]+1:]
            # Keep the offsets for following replacements
            postOffset += len(blind) - len(name)
            # Drug names and types
            if drug_id==e1:
                drug_e1 = [name, entity.get("type"), drug_id]
            if drug_id==e2:
                drug_e2 = [name, entity.get("type"), drug_id]
    return remove_repetition(clean_str(text)), drug_e1, drug_e2


def SDP(sentence):
    """
    Reduce the sentence to the Shortest Dependency Path
    """
    try:
        sentence = re.sub(ENTITY1, 'entitya', sentence)
        sentence = re.sub(ENTITY2, 'entityb', sentence)
        sentence = re.sub(ENTITY0, 'entityx', sentence)
        text = nlp(sentence)
        edges = [(token.head.lower_, token.lower_) for token in text]
        #arcs = [token.dep_ for token in text]
        #pos = [token.tag_ for token in text] #(sustituye a POS de NLTK)
        #lemma = [token.lemma_ for token in text]
        #iob = [token.ent_iob_ for token in text]
        #chunks = [chunk for chunk in text.noun_chunks]
        '''
        chunk_iob = ['O']*len(text)
        for nc in text.noun_chunks:
            for w in range(len(nc)):
                if w == 0:
                    chunk_iob[nc[w].i] = 'B'
                else:
                    chunk_iob[nc[w].i] = 'I'
        '''
        sentence = " ".join(nx.shortest_path(nx.Graph(edges), source='entitya', target='entityb'))
        sentence = re.sub('entitya', ENTITY1, sentence)
        sentence = re.sub('entityb', ENTITY2, sentence)
        sentence = re.sub('entityx', ENTITY0, sentence)
    except nx.NetworkXNoPath:
        pass
    return sentence


def text_generation(sentences_data, sentences_label, sentences_dev):
    """
    Generate syntethic text sentences of a predefined class
    sentences_data = array of the sentences
    sentences_label = label of the data
    sentences_dev = array of the sentences to test the generation
    """
    texts = []
    labels = []
    distances = []
    pos = []
    sdp = []
    # Save data for a label
    sentences_file = "data_" + sentences_label + "_train.txt"
    with open(sentences_file, 'w+') as f:
        f.write("\n".join(sentences_data))
    # GAN Models
    gan_types = ["mle", "seqgan", "maligan", "textgan", "leakgan", "rankgan", "gsgan"]
    gan_types = ["seqgan", "leakgan"]
    for gan in gan_types:
        gen_file = "generation/" + gan + "_" + sentences_label + ".txt"
        print("Generating file: " + gen_file)
        if not os.path.isfile(gen_file):
            os.makedirs("save")
            argv = "-g " + gan + " -t real -d " + sentences_file
            parse_cmd(argv.split())
            os.rename("save/test_file.txt", gen_file)
            os.remove("save/eval_data.txt")
            os.remove("save/generator.txt")
            os.remove("save/oracle.txt")
            os.rmdir("save/")
        with open(gen_file, "r") as f:
            sentences = f.read().split('\n')
        sentences = [s for s in sentences if len(re.findall(ENTITY1,s))==1 and len(re.findall(ENTITY2,s))==1 and s.find(ENTITY1)<s.find(ENTITY2)]
        print("Valid generated sentences with " + gen_file + ": " + str(len(sentences)))
        for text in sentences:
            '''
            print(text)
            print("BLEU-1: {}".format(sentence_bleu(sentences_dev, text, weights=(1,0,0,0))))
            print("BLEU-2: {}".format(sentence_bleu(sentences_dev, text, weights=(0,1,0,0))))
            print("BLEU-3: {}".format(sentence_bleu(sentences_dev, text, weights=(0,0,1,0))))
            print("BLEU-4: {}".format(sentence_bleu(sentences_dev, text, weights=(0,0,0,1))))
            print("cumBLEU-1: {}".format(sentence_bleu(sentences_dev, text, weights=(1,0,0,0))))
            print("cumBLEU-2: {}".format(sentence_bleu(sentences_dev, text, weights=(0.5,0.5,0,0))))
            print("cumBLEU-3: {}".format(sentence_bleu(sentences_dev, text, weights=(0.33,0.33,0.33,0))))
            print("cumBLEU-4: {}".format(sentence_bleu(sentences_dev, text, weights=(0.25,0.25,0.25,0.25))))
            '''
            texts.append(text)
            labels.append([sentences_label, None, None, None, None])
            distances.append([word_tokenize(text).index(ENTITY1), word_tokenize(text).index(ENTITY2)])
            if not word_tokenize == list:
                pos.append(text)#(" ".join([p[1] if not (p[0] == ENTITY1 or p[0] == ENTITY2 or p[0] == ENTITY0) else 'NN' for p in pos_tag(word_tokenize(text))]))
                sdp.append(text)#(SDP(text))
    return np.array(texts), np.array(labels), np.array(distances), np.array(pos), np.array(sdp)


def print_results(y_true, y_pred, labelsnames, verbose = False):
    labels = labelsnames[:]
    idx = labels.index("None")
    labels[idx] = "Other"
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    eps = sys.float_info.epsilon
    # Classification Results
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    fmeasure = 2 * precision * recall / (precision + recall + eps)
    support = TP + FN
    # Micro-Average without Other-class
    m_TP = np.sum(TP) - TP[idx]
    m_FP = np.sum(FP) - FP[idx]
    m_FN = np.sum(FN) - FN[idx]
    m_precision = m_TP / (m_TP + m_FP + eps)
    m_recall = m_TP / (m_TP + m_FN + eps)
    m_fmeasure = 2 * m_precision * m_recall / (m_precision + m_recall + eps)
    m_support = np.sum(support) - support[idx]
    # Macro-Average without Other-class
    M_TP = (np.mean(TP) * len(TP) - TP[idx]) / (len(TP) - 1)
    M_FP = (np.mean(FP) * len(FP) - FP[idx]) / (len(FP) - 1)
    M_FN = (np.mean(FN) * len(FN) - FN[idx]) / (len(FN) - 1)
    M_precision = (np.mean(precision) * len(precision) - precision[idx]) / (len(precision) - 1)
    M_recall = (np.mean(recall) * len(recall) - recall[idx]) / (len(recall) - 1)
    M_fmeasure = 2 * M_precision * M_recall / (M_precision + M_recall + eps)
    M_support = (np.mean(support) * len(support) - support[idx]) / (len(support) - 1)
    if verbose:
        max_l = max([len(l) for l in labels])
        string = "CLASSIFICATION ANALYTICS\n"
        class_mat = np.zeros((len(labels)+3, 8), dtype="S" + str(max(max_l, 12)))
        class_mat[0,:] = ["Classes", "TP", "FP", "FN", "Precision", "Recall", "F-measure", "Support"]
        class_mat[1:,0] = labels + ["MicroAverage", "MacroAverage"]
        class_mat[1:,1] = np.concatenate([TP, [m_TP, str(np.around(M_TP, decimals=2))]])
        class_mat[1:,2] = np.concatenate([FP, [m_FP, str(np.around(M_FP, decimals=2))]])
        class_mat[1:,3] = np.concatenate([FN, [m_FN, str(np.around(M_FN, decimals=2))]])
        class_mat[1:,4] = np.around(100*np.concatenate([precision, [m_precision, M_precision]]), decimals=2)
        class_mat[1:,5] = np.around(100*np.concatenate([recall, [m_recall, M_recall]]), decimals=2)
        class_mat[1:,6] = np.around(100*np.concatenate([fmeasure, [m_fmeasure, M_fmeasure]]), decimals=2)
        class_mat[1:,7] = np.concatenate([support, [m_support, str(np.around(M_support, decimals=2))]])
        string += "\n".join(["".join([("%-" + str(max(max_l,12)+5)*(column==0)+str(len(class_mat[0][column])+5)*(column>0) + "s") % (class_mat[row][column].decode("utf-8")) for column in range(class_mat.shape[1])]) for row in range(class_mat.shape[0])])
        string += "\nCONFUSION MATRIX\n"
        con_mat = np.zeros((len(labels)+1, len(labels)+1), dtype="S" + str(max(max_l, 5)))
        con_mat[0,:] = ["Classes"] + labels
        con_mat[1:,0] = labels
        con_mat[1:len(labels)+1, 1:len(labels)+1] = cm
        string += "\n".join(["".join([("%-" + str(max(max_l,12)+5)*(column==0)+str(len(con_mat[0][column])+5)*(column>0) + "s") % (con_mat[row][column].decode("utf-8")) for column in range(con_mat.shape[1])]) for row in range(con_mat.shape[0])])
        print(string)
    return [["MicroAverage", 100*m_precision, 100*m_recall, 100*m_fmeasure], ["MacroAverage", 100*M_precision, 100*M_recall, 100*M_fmeasure]]