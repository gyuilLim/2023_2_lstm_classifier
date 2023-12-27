import re
import pandas as pd

class Preprocessing() :
    def __init__(self, df_path):
        self.df_path = df_path
        self.toxicity_df = pd.read_csv(self.df_path, header=0)
        
    def preprocess_string(self, s):
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", '', s)
        # Replace all runs of whitespaces with a single space
        s = re.sub(r"\s+", ' ', s)
        # Replace digits with no space
        s = re.sub(r"\d", '', s)

        return s
    
    
    def preprocess_comment(self):
        self.toxicity_df['comment'] = self.toxicity_df['comment'].apply(self.preprocess_string)
        self.toxicity_df = self.toxicity_df.dropna()
        
        return self.toxicity_df
    
    
profenity = ['fuck', 'shit', 'suck', 'dick', 'nigga']#

def check_profanity(comment):
    count = 0
    for word in profenity:
        if word in comment:
            count += 1
            if count >= 2:
                return True
    return False