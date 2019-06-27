# MOVE TO ../NTAP/


from NTAP.data import Dataset
from NTAP.models import SVM, RNN
#from NTAP.viz show_alphas

data = Dataset("/home/brendan/Data/GabProject/full_gab_binary.pkl")
#data2 = Dataset(data)
data.clean("text")
data.set_params(vocab_size=5000)
#data.set_params(dictionary='mfd.json')
model = RNN("hate ~ seq(text)", data=data)#, optimizer='adagrad',
        #learning_rate=0.01, rnn_pooling=100, num_epochs=10)
#model.train(data)
#model.CV(data)
#model.print_vars()
alphas, predictions = model.predict(data, retrieve=["alphas", "predictions"])

#print(model.nodes)
#alphas, predictions = model.predict(data, retrieve=["alphas", "predictions"])

#model.CV(data)
#model.train(data)
data.set_params(num_topics=100, lda_max_iter=50)
model = SVM("hate ~ lda(text)", data=data, C=[0.1, 0.5, 0.8])
model.CV(data, num_folds=3)
model.CV(data, metric='precision')

model.summary(metrics=["accuracy", "precision"])
#fifth_model = model.load_model(cv_idx=5)
model.predict(data, model_path="...")
print(model.best_params)
print(model.best_score)

model.train(data)
#print(model.best_params)
#print(model.best_score)
y = model.predict(data)

