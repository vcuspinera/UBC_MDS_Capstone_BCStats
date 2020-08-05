# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-05

import pandas as pd
import re
import spacy
import string
import en_core_web_sm
nlp = en_core_web_sm.load()

class Preprocessing:
    # Function adapted from `preprocess` function shared by Varada
    # in course 575 Advance Learning Machine.
    # 'Script with a class that preprocess the comments for using in models.'
    
    def general(self, text,
                min_token_len = 2,
                irrelevant_pos = ['PRON', 'SPACE', 'PUNCT', 'ADV', 
                                  'ADP', 'CCONJ', 'AUX', 'PRP'],
                avoid_entities = ['PERSON', 'ORG', 'LOC', 'GPE']):
        # note: Didn't use the following options in the `preprocess_comments`
        #    - 'PROPN', erase proper names, but also words as orange.
        #    - 'DET', removes the word 'no', which changes the meaning.
        """
        Function that identify sensible information, anonymize and transforms
        the data in a useful format for using with tokens.

        Parameters
        -------------
        text : (list)
            the list of text to be preprocessed
        irrelevant_pos : (list)
            a list of irrelevant 'pos' tags
        avoid_entities : (list)
            a list of entity labels to be avoided
        
        Returns
        -------------
        (list) list of preprocessed text
        
        Example
        -------------
        example = ["Hello, I'm George and I love swimming!",
                    "I am a really good cook; what about you?",
                    "Contact me at george23@gmail.com"]
        preprocess(example)
        (output:) ['hello love swimming', 'good cook', 'contact']
        """
        result = []
        others = ["'s", "the", "that", "this", "to", "-PRON-"]
        # comment: "-PRON-" is a lemma for "my", "your", etc.

        # tests
        # assert isinstance(text, list), "Error, you should pass a list of comments."

        # function
        for sent in text:
            sent = str(sent).lower()
            sent = re.sub(r"facebook", "social media", sent)
            sent = re.sub(r"twitter", "social media", sent)
            sent = re.sub(r"instagram", "social media", sent)
            sent = re.sub(r"whatsapp", "social media", sent)
            sent = re.sub(r"linkedin", "social media", sent)
            sent = re.sub(r"snapchat", "social media", sent)
            
            result_sent = []
            doc = nlp(sent)
            entities = [str(ent) for ent in doc.ents if ent.label_ in avoid_entities]
            # This helps to detect names of persons, organization and dates
            
            for token in doc:            
                if (token.like_email or
                    token.like_url or
                    token.pos_ in irrelevant_pos or
                    str(token) in entities or
                    str(token.lemma_) in others or
                    len(token) < min_token_len):
                    continue
                else:
                    result_sent.append(token.lemma_)
            result.append(" ".join(result_sent))
        return result


    def anonymize(self, text,
                irrelevant_pos = ['SPACE'],
                avoid_entities = ['PERSON', 'ORG', 'LOC', 'GPE']):
        # note: Didn't use the following options in the `preprocess_comments`
        #    - 'PROPN', erase proper names, but also words as orange.
        #    - 'DET', removes the word 'no', which changes the meaning.
        """
        Function that identify sensible information and anonymize the data.

        Parameters
        -------------
        text : (list)
            the list of text to be preprocessed
        irrelevant_pos : (list)
            a list of irrelevant 'pos' tags
        avoid_entities : (list)
            a list of entity labels to be avoided
        
        Returns
        -------------
        (list) list of preprocessed text
        
        Example
        -------------
        example = ["Hello, I'm George and I love swimming!",
                    "I am a really good cook; what about you?",
                    "Contact me at george23@gmail.com"]
        preprocess().anonymize(example)
        (output:) ["hello, i 'm and i love swimming!",
                'i am a really good cook; what about you?',
                'contact me at']
        """
        result = []
        
        # tests
        # assert isinstance(text, list), "Error, you should pass a list of comments."

        # function
        for sent in text:
            sent = str(sent).lower()
            sent = re.sub(r"facebook", "social media", sent)
            sent = re.sub(r"twitter", "social media", sent)
            sent = re.sub(r"instagram", "social media", sent)
            sent = re.sub(r"whatsapp", "social media", sent)
            sent = re.sub(r"linkedin", "social media", sent)
            sent = re.sub(r"snapchat", "social media", sent)
            
            result_sent = []
            doc = nlp(sent)
            entities = [str(ent) for ent in doc.ents if ent.label_ in avoid_entities]
            # This helps to detect names of persons, organization and dates
            
            for token in doc:            
                if (token.like_email or
                    token.like_url or
                    token.pos_ in irrelevant_pos or
                    str(token) in entities):
                    continue
                else:
                    if str(token) in string.punctuation:
                        try:
                            result_sent[-1] = str(result_sent[-1]) + str(token)
                        except:
                            result_sent.append(str(token))
                    else:
                        result_sent.append(str(token))
            result.append(" ".join(result_sent))
        return result
