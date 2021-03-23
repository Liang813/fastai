from fastai.text import *
# setup learner with IMDB data
path = untar_data(URLs.IMDB_SAMPLE)
data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', bs=32)
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

# get preds
probs, y_trues, reported_loss = learn.get_preds(with_loss=True, ordered=True)

# manually calculate the loss
rng = 0  # either number or a range, e.g. range(0,3)
calc_loss = F.cross_entropy(np.log(probs[rng]).reshape(-1,len(learn.data.classes)), # log(probs) reverses the softmax (ignoring the original scale)
                y_trues[rng].reshape(-1), reduction='none')
print("TypeError: __init__() got an unexpected keyword argument 'auto_update'")
# check whether losses match
assert np.all(np.array(calc_loss) - np.array(reported_loss[rng]) < 0.0001)

