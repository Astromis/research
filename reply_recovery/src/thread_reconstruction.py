from collections import defaultdict
import numpy as np
from copy import deepcopy
from scipy.special import softmax
from dataset import TextsLabelsDataset

def extract_from_annot(dialog_):
    """Parse the LabelStudio json with dialogue annotation.

    See the data description for more details.

    Args:
        dialog_ (_type_): The one dialog from config list.

    Returns:
        _type_: The tuple of maps.
    """
    id2num = {}
    id_rel_id = defaultdict(list)
    for mes in dialog_:
        if mes["type"] == "paragraphlabels":
            if "value" in mes:
                id2num[mes["id"]] = int(mes["value"]["start"])
        if mes["type"] == "relation":
            if id2num[mes["to_id"]] > id2num[mes["from_id"]]:
                id_rel_id[mes["to_id"]].append(mes["from_id"])
            else:
                id_rel_id[mes["from_id"]].append( mes["to_id"])
        else:
            continue
    return id2num, id_rel_id

def get_edge_matrix_from_annot(dialog_list, mat_shape, ignore_self_connections=True):
    """Generate an edge graph matrix from dialogue annotation.

    Args:
        dialog_list (_type_): The list of annotated dialogue from Label Studio json `results` key.
        mat_shape (_type_): The shape of the matrix. Determined by the number of messages in dialogue.
        ignore_self_connections (bool, optional): Ignore the `self-connection` annotation. Defaults to True.

    Returns:
        _type_: An edge matrix.
    """
    id2num, id_rel_id = extract_from_annot(dialog_list)
    g = np.ones((mat_shape, mat_shape))
    edge_list = {}
    for i, j in id_rel_id.items():
        for k in j:
            edge_list[id2num[i]] = id2num[k]

    for i,j in edge_list.items():
        if i == j and ignore_self_connections:
            g[i,j] = 1
            continue
        g[i,j] = 0
    return g

def estimate_replies(dial, max_len, trainer, tokenizer):
    """Produce the reply probability matrix on dialog message list.

    Given the message list, we estimate how likely that previos messages
    up to max_len are replyes for a particular message. As we compute for 
    every message the result will be a probability matrix.

    Args:
        dial (List[str]): The message list.
        max_len (int): How many messages we consider for prediction.
        trainer (_type_): The crunch for a handle prediction aquisition. Should be replaced.
        tokenizer (_type_): A tokenizer for a messages.

    Returns:
        _type_: The reply probability matrix.
    """
    dialog_len = len(dial)
    data = []
    dial.reverse()
    for i, m in enumerate(dial[:-1], 0):
        p = []
        h = []
        right = dialog_len if i+max_len > dialog_len else i+max_len
        for m_ in dial[i+1:right]:
            p.append(m_)
            h.append(m)    
        data.append( (p, h) )
    dial.reverse()

    preds = []
    for premise, hypothesis in data:
        tokenized_texts = tokenizer(premise, hypothesis, return_tensors='pt',
                                    truncation=True, max_length=512, padding = 'max_length',)
        iter_data = TextsLabelsDataset(tokenized_texts, [0]*len(premise))
        iter_predictions = trainer.predict(iter_data)
        iter_predictions_softmax = softmax(iter_predictions[0], axis=1)[:, 0]
        preds.append(iter_predictions_softmax)

    gr = np.ones((dialog_len, dialog_len))

    for i,m in enumerate(preds, start=0):
        for j, p in enumerate(m):
            shift = i+1
            gr[i,shift + j] = p
    estimanted_matrix_proba = np.flip(gr)
    return estimanted_matrix_proba

def binarize(mat, treshold=.5):
    """Binarize the probability matrix.

    Args:
        mat (_type_): The probability matrix.
        treshold (float, optional): A binarization treshold. Defaults to .5.

    Returns:
        _type_: A binary matrix.
    """
    binary_mat = deepcopy(mat)
    rows, cols = np.where(binary_mat == 1.)
    binary_mat[rows, cols] = 0
    new_binary = np.ones(binary_mat.shape)
    new_binary[binary_mat >= treshold] = 0
    new_binary = new_binary.astype(np.int64)
    return new_binary

def greedify_binary(est_mat):
    """A thread reconstruction stategy that chose the reply
    
    by first classified reply. 

    Args:
        est_mat (_type_): A reply probability matrix. 

    Returns:
        _type_: The result reply matrix.
    """
    greed_matrix = deepcopy(est_mat)
    for i in range(est_mat.shape[0]):
        zeros = np.where(greed_matrix[i] == 0)[0]
        greed_matrix[i, zeros[:-1]] = 1
    return greed_matrix

def greedify_probas(est_max):
    """A thread reconstruction strategy that chose the reply
    by maximum probability in raw.

    Args:
        est_max (_type_): A reply probability matrix.

    Returns:
        _type_: The result reply meatrix.
    """
    probas_mat = deepcopy(est_max)
    rows, cols = np.where(probas_mat == 1.)
    probas_mat[rows, cols] = 0
    probas_mat[probas_mat < 0.5] = 0
    preds = probas_mat.argmax(axis=1)
    no_predicts_row_idx = np.where(probas_mat.sum(axis=1)==.0)[0]
    new_preds = np.ones(probas_mat.shape)
    for r,c in enumerate(preds):
        if r not in no_predicts_row_idx:
            new_preds[r,c] = 0
    return new_preds

def get_dummy(mat):
    """A thread reconstruction strategy that chose the reply
    by previous message.

    Args:
        mat (_type_): The reply probability matrix.

    Returns:
        _type_: _description_
    """
    dummy = np.ones(mat.shape)
    for i in range(1,100):
        dummy[i,i-1]=0
    return dummy

g_probas = lambda x: greedify_probas(x)
g_binary = lambda x: greedify_binary(binarize(x))