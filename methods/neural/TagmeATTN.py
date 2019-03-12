"""
pseudo code

1.) Prep data/features
    * Run Tagme (via Pipeline) on data
    * Generate features with DDR + MFD
2.) Write "get_batches" function that returns text, label, and ddr features
    * location: methods/neural/neural.py (get_batches)
    * alter the function get_batches, write a new function, or write a new
    class


3.) Copy and revise "ATTN_feat.py" to incorporate features

from methods.neural.nn import *  # includes run, run_pred

class TagmeAttn():
    def __init__(...):
        ...
    def build(self):
        ## declare TF variables
        # self.feat = tf.placeholder(...)
        # ...
        ## build embedding matrix for word2vec
        # self.embed = tf.nn.embedding_lookup(...)
        #...
        ## define network
        # attention:
        # gets hidden vectors, calculates weights, applies weights
        # gets feature vector, applies dropout to summary & features
        # applies both to fully connected layer
        # calculate loss and gets training op

    def run_model(self, ...):
        return run(self, ...)
    def predict_labels(self, ...):
        return run_pred(self, ...)

"""
