import jieba
import  string
corpus='''我喜欢吃，苹果。你喜欢吃啥。苹果是一种水果。他爱吃梨吗？'''
def n_grame(n,corpus,test):
    #停用标点符号表
    stop_punctuation=['，','？','。','！','：','；']
    # test.translate(str.maketrans("", "", string.punctuation))
    # corpus.translate(str.maketrans("", "", string.punctuation))
    gen_corpus=jieba.cut(corpus)
    list_word=list(filter(lambda x:False if x in string.punctuation or x in stop_punctuation else True,gen_corpus))
    list_test_word=list(jieba.cut(test));
    list_test_word=list(filter(lambda x:False if x in string.punctuation or x in stop_punctuation else True,list_test_word))
    #制作词汇表  +1平滑
    word_dict=set(list_word);
    v=len(word_dict)
    list_test_word.insert(0,"BOS");
    test_word_number = len(list_test_word);
    list_test_word.insert(test_word_number,"EOS")
    n_gramp_word_map=dict()
    test_n_gram_list=[]
    for i in range(len(list_test_word)-n+1):
        n_tuple=tuple(list_test_word[i:i+n])
        test_n_gram_list.append(n_tuple)
    list_word.insert(0,'BOS')
    list_word.insert(len(list_word),'EOS')
    for i in range(len(list_word)-n+1):
        n_tuple=tuple(list_word[i:i+n])
        if n_tuple in n_gramp_word_map.keys():
            n_gramp_word_map[n_tuple]+=1;
        else:
            n_gramp_word_map[n_tuple]=1
    #输出n介马尔科夫链
    for value in test_n_gram_list:
        #平滑操作
        count_mol=1;
        count_den=v;
        condition_probality=1
        link_probality = 1;
        for k,v in n_gramp_word_map.items():
            if value==k:
                count_mol+=1;
            if value[0]==k[0]:
                count_den+=1
        condition_probality=count_mol/count_den
        link_probality*=condition_probality
    return link_probality
test1='''她喜欢吃水果吗'''
test2='''苹果喜欢吃你吗'''
print(n_grame(3,corpus,test1))
print(n_grame(3,corpus,test2))
