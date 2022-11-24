import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
import wn 

class WordnetOntology():
    def __init__(self, synset_mapping_filepath):
        self.wn_en = wn.Wordnet('omw-en')
        
        self.class_for_index = {}
        self.index_for_class = {}
        self.name_for_index = {}
        self.index_for_name = {}
        with open(synset_mapping_filepath) as file:
            for idx, line in enumerate(file, start=0):
                syn_id, name = line.split(',')[0].split(' ', maxsplit=1)
                self.class_for_index[idx] = syn_id # class n012932
                self.index_for_class[syn_id] = idx
                self.name_for_index[idx] = name # tinca
                self.index_for_name[name] = idx
    
    # from class to class lemma
    def class_name(self, synset_id):
        ss = self.wn_en.synset(f'omw-en-{synset_id[1:]}-n')
        return ss.lemmas()[0]

    # From class n21938 to hypernym class n983214
    def hypernym(self, synset_id):
        ss = self.wn_en.synset(f'omw-en-{synset_id[1:]}-n')
        return f"n{ss.hypernyms()[0].id.split('-')[2]}"
