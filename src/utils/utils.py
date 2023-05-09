
def merge_sub_word(tokenizer, sentence):
    sub_words = tokenizer.convert_ids_to_tokens(sentence)
    words = []
    i= 0
    len_sub = len(sub_words)
    cur_word = ''
    while i <len_sub:
        if sub_words[i].startswith('##'):
            cur_word = cur_word+sub_words[i][2:]
        else:
            if len(cur_word)!=0:
                words.append(cur_word)
            cur_word = sub_words[i]
        i = i+1
    if len(cur_word)!=0:
        words.append(cur_word)
    return words, sub_words

def pos_subword(input_str, input_pos, compress_str):
    len_tgt = len(compress_str)
    compress_pos = [0 for i in range(len_tgt)]
    s1_ind = 0
    s2_ind = 0
    while s2_ind<len_tgt:
        if compress_str[s2_ind]==input_str[s1_ind]:
            compress_pos[s2_ind] = input_pos[s1_ind]
            s1_ind = s1_ind+1
            s2_ind = s2_ind+1
        else:
            shift = 0
            sub_w = ''
            while sub_w!=input_str[s1_ind]:
                shift = shift + 1
                assert s2_ind + shift + 1<=len_tgt, "error"
                sub_w = compress_str[s2_ind:s2_ind + shift + 1]
                sub_w = "".join(sub_w)
                sub_w = sub_w.replace('##', '')
            for j in range(shift+1):
                compress_pos[s2_ind+j] = input_pos[s1_ind]
            s1_ind = s1_ind+1
            s2_ind = s2_ind +shift+1
    return compress_pos
