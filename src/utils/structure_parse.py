psl = '(ROOT\n  (S\n    (PP (IN At)\n      (NP\n        ' \
     '(NP (DT the) (NN end))\n        (PP (IN of)\n          ' \
     '(NP (DT the) (NN day)))))\n    (, ,)\n    (S\n      ' \
     '(ADVP (RB successfully))\n      (VP (VBG launching)\n        ' \
     '(NP (DT a) (JJ new) (NN product))))\n    (VP (VBZ means)\n      ' \
     '(S\n        (VP\n          (VP (VBG reaching)\n            ' \
     '(NP (DT the) (JJ right) (NN audience)))\n          ' \
     '(CC and)\n          (VP\n            (ADVP (RB consistently))\n            ' \
     '(VBG delivering)\n              (NP (DT a)\n              ' \
     '(ADJP (RB very) (JJ convincing))\n              (NN message))))))\n    (. .)))'
ps = '(ROOT\n  (S\n    (NP (PRP I))\n    (VP (VBP love)\n      (NP (PRP you)))\n    (. .)))'
# inp_to_tokens =['[CLS]', 'He', 'really', 'needs', 'to', 'fix', 'his', 'windscreen', '!', 'Lorry', 'driver', 'covers', 'wrecked', 'truck', 'with', 'tarpaulin', 'before', 'speeding', 'down', 'the', 'motorway', 'with', 'almost', 'ZERO', 'visibility', '.', 'A', 'mysterious', 'lorry', 'covered', 'in', 'plastic', 'wrap', 'was', 'spotted', 'on', 'a', 'motorway', 'in', 'Hubei', ',', 'central', 'China', ',', 'recently', '.', 'Almost', 'every', 'inch', 'of', 'the', 'driver', "'", 's', 'cabin', 'was', 'covered', 'with', 'tarpaulin', ',', 'leaving', 'only', 'a', 'tiny', 'square', 'to', 'allow', 'the', 'driver', 'to', 'look', 'out', ',', 'according', 'to', 'People', "'", 's', 'Daily', 'Online', '.', 'Incredibly', ',', 'the', 'vehicle', 'was', 'travelling', 'at', 'around', '50', 'miles', 'per', 'hour', 'on', 'the', 'road', ',', 'despite', 'reduced', 'visibility', '.', 'The', 'vehicle', 'was', 'spotted', 'by', 'the', 'local', 'police', 'on', 'a', 'live', 'video', 'feed', 'about', 'mid', '-', 'day', 'on', 'Christmas', 'Day', '.', 'Apart', 'from', 'the', 'wing', 'mirror', 'pointing', 'out', ',', 'it', 'was', 'almost', 'indistinguishable', 'as', 'a', 'lorry', 'from', 'the', 'front', '.', 'Police', 'were', 'able', 'to', 'stop', 'the', 'driver', 'at', 'a', 'toll', 'station', 'along', 'the', 'road', '.', 'The', 'driver', 'of', 'the', 'vehicle', ',', 'identified', 'by', 'his', 'surname', 'Cheng', ',', 'had', 'to', 'climb', 'out', 'of', 'the', 'window', 'in', 'order', 'to', 'be', 'questioned', '.', 'According', 'to', 'the', 'driver', ',', 'he', 'was', 'involved', 'in', 'an', 'accident', 'several', 'days', 'ago', 'in', 'Jingmen', ',', 'Hubei', ',', 'when', 'he', 'rear', '-', 'ended', 'another', 'vehicle', '.', 'As', 'he', 'was', 'tail', '-', 'gating', ',', 'he', 'had', 'to', 'pay', 'compensation', 'of', '3', ',', '000', 'Yuan', '-', 'LRB', '-', '#', '300', '-', 'RRB', '-', 'to', 'the', 'other', 'party', '-', 'almost', 'all', 'of', 'the', 'money', 'he', 'had', 'with', 'him', '.', 'In', 'order', 'to', 'save', 'money', 'on', 'repairs', ',', 'Cheng', 'decided', 'to', 'drive', 'his', 'lorry', 'to', 'a', 'garage', 'that', 'he', 'knows', 'about', '163', 'miles', 'away', 'in', 'Nan', '[SEP]']
# sub_words = ['[CLS]', 'He', 'really', 'needs', 'to', 'fix', 'his', 'winds', '##cre', '##en', '!', 'Lo', '##rry', 'driver', 'covers', 'wrecked', 'truck', 'with', 'ta', '##rp', '##aul', '##in', 'before', 'speeding', 'down', 'the', 'motorway', 'with', 'almost', 'Z', '##ER', '##O', 'visibility', '.', 'A', 'mysterious', 'lo', '##rry', 'covered', 'in', 'plastic', 'wrap', 'was', 'spotted', 'on', 'a', 'motorway', 'in', 'Hu', '##bei', ',', 'central', 'China', ',', 'recently', '.', 'Almost', 'every', 'inch', 'of', 'the', 'driver', "'", 's', 'cabin', 'was', 'covered', 'with', 'ta', '##rp', '##aul', '##in', ',', 'leaving', 'only', 'a', 'tiny', 'square', 'to', 'allow', 'the', 'driver', 'to', 'look', 'out', ',', 'according', 'to', 'People', "'", 's', 'Daily', 'Online', '.', 'Inc', '##red', '##ibly', ',', 'the', 'vehicle', 'was', 'travelling', 'at', 'around', '50', 'miles', 'per', 'hour', 'on', 'the', 'road', ',', 'despite', 'reduced', 'visibility', '.', 'The', 'vehicle', 'was', 'spotted', 'by', 'the', 'local', 'police', 'on', 'a', 'live', 'video', 'feed', 'about', 'mid', '-', 'day', 'on', 'Christmas', 'Day', '.', 'Apart', 'from', 'the', 'wing', 'mirror', 'pointing', 'out', ',', 'it', 'was', 'almost', 'in', '##dis', '##ting', '##ui', '##sha', '##ble', 'as', 'a', 'lo', '##rry', 'from', 'the', 'front', '.', 'Police', 'were', 'able', 'to', 'stop', 'the', 'driver', 'at', 'a', 'toll', 'station', 'along', 'the', 'road', '.', 'The', 'driver', 'of', 'the', 'vehicle', ',', 'identified', 'by', 'his', 'surname', 'Cheng', ',', 'had', 'to', 'climb', 'out', 'of', 'the', 'window', 'in', 'order', 'to', 'be', 'questioned', '.', 'According', 'to', 'the', 'driver', ',', 'he', 'was', 'involved', 'in', 'an', 'accident', 'several', 'days', 'ago', 'in', 'Jing', '##men', ',', 'Hu', '##bei', ',', 'when', 'he', 'rear', '-', 'ended', 'another', 'vehicle', '.', 'As', 'he', 'was', 'tail', '-', 'g', '##ating', ',', 'he', 'had', 'to', 'pay', 'compensation', 'of', '3', ',', '000', 'Yuan', '-', 'L', '##RB', '-', '#', '300', '-', 'R', '##RB', '-', 'to', 'the', 'other', 'party', '-', 'almost', 'all', 'of', 'the', 'money', 'he', 'had', 'with', 'him', '.', 'In', 'order', 'to', 'save', 'money', 'on', 'repairs', ',', 'Cheng', 'decided', 'to', 'drive', 'his', 'lo', '##rry', 'to', 'a', 'garage', 'that', 'he', 'knows', 'about', '163', 'miles', 'away', 'in', 'Nan', '[SEP]']
def nodes_info(parse_str,wid=0, sid=0):
    nodes_stack = []
    nodes_out = {}
    i = 0
    node_id = 0
    word_seq = []
    len_p = len(parse_str)
    while i < len_p:
        if parse_str[i] == '(':  # 压栈
            cur_r = ''
            i = i + 1
            while not (parse_str[i] == ' ' or parse_str[i] == '\n'):
                cur_r = cur_r + parse_str[i]
                i = i + 1
            nodes_stack.append(str(sid)+'-'+cur_r + '-' + str(node_id))
            node_id = node_id + 1
        elif parse_str[i] == ')':
            if parse_str[i - 1] != ')':  # 叶节点
                cur_r = ''
                j = -1
                while not (parse_str[i + j] == ' ' or parse_str[i + j] == '\n'):
                    cur_r = parse_str[i + j] + cur_r
                    j = j - 1
                # 出叶节点
                last_node = nodes_stack.pop()
                if len(nodes_stack) == 0:
                    break
                # nodes_out[last_node] = [0,str(wid)+'@'+cur_r]
                nodes_out[last_node] = [0,wid]
                word_seq.append(cur_r)
                wid = wid +1
                # 补充该叶节点的父节点信息
                llast_node = nodes_stack[-1]
                if llast_node in nodes_out:
                    level = max(1, nodes_out[llast_node][0])
                    nodes_out[llast_node][1].append(last_node)
                    nodes_out[llast_node][0]= level
                else:
                    nodes_out[llast_node] = [1,[last_node]]
                i = i + 1
            else:  # 非叶节点
                last_node = nodes_stack.pop()
                if len(nodes_stack) == 0:
                    break
                llast_node = nodes_stack[-1]
                if llast_node in nodes_out:
                    level = max(nodes_out[llast_node][0], nodes_out[last_node][0]+1)
                    nodes_out[llast_node][1].append(last_node)
                    nodes_out[llast_node][0]= level
                else:
                    nodes_out[llast_node] = [nodes_out[last_node][0]+1,[last_node]]
                i = i + 1
        else:
            i = i + 1
    return nodes_out,word_seq

def nodes_words_cover(nodes_out, MinLen=2, MaxLen=5):
    level = 0
    nodes_block = {}
    single_connect_keys = []
    single_values = []
    out_interval_keys = []
    while len(nodes_out) != len(nodes_block):
        # import pdb
        # pdb.set_trace()
        if level == 0:
            for k, v in nodes_out.items():
                if v[0] == 0:
                    nodes_block[k] = [v[1]]
                    single_values.append(k)
        else:
            for k, v in nodes_out.items():
                cur_ws = []
                if v[0] == level:
                    for vv in v[1]:
                        cur_ws = cur_ws + nodes_block[vv]
                    nodes_block[k] = cur_ws
                    if len(v[1]) == 1:
                        single_connect_keys.append(k)
                    elif len(cur_ws) > MaxLen or len(cur_ws)<MinLen:
                        out_interval_keys.append(k)

        level = level + 1
    for k in single_connect_keys:
        nodes_block.pop(k)
    for k in single_values:
        nodes_block.pop(k)
    for k in out_interval_keys:
        nodes_block.pop(k)
    return nodes_block


def sub2org(sub_words, org_words):
        cur_index = 0
        subi2orgi = []
        for iter, subw in enumerate(sub_words):
            if subw.startswith('##') and subw.replace('##','') in org_words[cur_index - 1]:
                subi2orgi.append(cur_index - 1)
                # print(subw,org_words[cur_index],iter,cur_index)
            elif subw in org_words[cur_index]:
                subi2orgi.append(cur_index)
                # print(subw,org_words[cur_index],iter,cur_index-1)
                cur_index = cur_index + 1
            elif subw in org_words[cur_index-1]:
                subi2orgi.append(cur_index - 1)
            else:
                if org_words[cur_index]+org_words[cur_index+1]==subw:
                    subi2orgi.append(cur_index)
                    cur_index = cur_index + 2
                else:
                    print("Empty subw")
        return subi2orgi
# nodes_out = nodes_info(psl)
# nodes_covers = nodes_words_cover(nodes_out)
# print("Parse result: ",nodes_out)
# print("Words Covers: ",nodes_covers)

# >>> tensor_0=torch.tensor([[1.2,3,1,0.5],[-1.0,0.7,1.4,3.3]])
# >>> index=torch.tensor([[0,0,1,2,2,3],[0,0,1,2,2,3]])
# >>> tensor_0.gather(1,index)
# tensor([[ 1.2000,  1.2000,  3.0000,  1.0000,  1.0000,  0.5000],
#         [-1.0000, -1.0000,  0.7000,  1.4000,  1.4000,  3.3000]])
# >>>

# >>> a=torch.zeros(2,10)
# >>> index1=torch.tensor([0,0])
# >>> index2=torch.tensor([2,3])
# >>> a[index1,index2]=1
# >>>
from stanfordcorenlp import StanfordCoreNLP
import torch

def to_phrase(nlp, inp_to_tokens, sub_words):
    sentences = inp_to_tokens[1:-1]
    sentences = " ".join(sentences)

    sents = []
    last_pos = 0
    for cid, c in enumerate(sentences):
        if c == '.' or c == '!' or c == '?':
            sents.append(sentences[last_pos:cid + 1])
            last_pos = cid + 1
    if len(sentences[last_pos:])>0:
        sents.append(sentences[last_pos:])
    # print(sentences)
    w_pos = 1
    word_seqs = [sub_words[0]]
    phrase_all = {}
    #1. to parse
    for sid, sent in enumerate(sents):
        # print(sid, sent)
        parse_str = nlp.parse(sent)
        nodes_out, word_seq = nodes_info(parse_str, w_pos, sid)
        # print(sent.strip().split(' '), word_seq)
        assert len(sent.strip().split(' '))<=len(word_seq),'sub_words longer than org_words'
        nodes_covers = nodes_words_cover(nodes_out)
        # print("Parse str: ", parse_str)
        # print("Parse result: ", nodes_out)
        # print("Words Covers: ", nodes_covers)
        w_pos = w_pos + len(word_seq)
        word_seqs = word_seqs + word_seq
        phrase_all.update(nodes_covers)
    word_seqs = word_seqs + [sub_words[-1]]
    subi2orgi = sub2org(sub_words, word_seqs)
    assert len(subi2orgi)==len(sub_words), 'len of subi2orgi ! = len of sub_words'
    len_org_words = len(word_seqs)

    #2. to tensor
    if len(phrase_all)==0:
        phrase_tensor_sub = torch.zeros(1, len(sub_words))
    else:
        phrase_tensor = torch.zeros(len(phrase_all), len_org_words)
        phrase_index1 = []
        phrase_index2 = []
        for kid, v in enumerate(phrase_all.values()):
            phrase_index1 = phrase_index1 + [kid for _ in range(len(v))]
            phrase_index2 = phrase_index2 + v
        phrase_index1 = torch.tensor(phrase_index1)
        phrase_index2 = torch.tensor(phrase_index2)
        # print("phrase index1: ", phrase_index1)
        # print("phrase index2: ", phrase_index2)
        phrase_tensor[phrase_index1, phrase_index2] = 1
        subi2orgi = torch.tensor(subi2orgi).view(1, -1)
        subi2orgi = subi2orgi.expand(len(phrase_all), len(sub_words))
        phrase_tensor_sub = phrase_tensor.gather(1, subi2orgi)
    return phrase_tensor_sub

if __name__=='__main__':
    nlp = StanfordCoreNLP('/home/meihuan2/download/stanford-corenlp-full-2018-02-27')
    phrase_tensor = to_phrase(nlp, inp_to_tokens, sub_words)
    print(phrase_tensor.shape)
    print(phrase_tensor[0])
    nlp.close()



