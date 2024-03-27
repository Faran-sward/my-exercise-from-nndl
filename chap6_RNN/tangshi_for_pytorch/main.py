import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn as rnn_lstm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start_token = 'G'
end_token = 'E'
batch_size = 64

def process_poems1(file_name):
    """

    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                # content = content.replace(' ', '').replace('，','').replace('。','')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                # print("error")
                pass

    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y  = row[1:]
            y.append(row[-1])
            y_data.append(y)
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
       
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():

    poems_vector, word_to_int, vocabularies = process_poems1('./poems.txt')

    print("finish  loadding data")
    BATCH_SIZE = 100

    torch.manual_seed(5)

    word_embedding = rnn_lstm.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 100)

    rnn_model = rnn_lstm.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)
    rnn_model.to(device)
    # optimizer = optim.Adam(rnn_model.parameters(), lr= 0.001)

    optimizer=optim.RMSprop(rnn_model.parameters(), lr=0.01)

    loss_fun = torch.nn.NLLLoss()

    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))  # if you have already trained your model you can load it by this line.

    for epoch in range(100):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch] # (batch , time_step)
            loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype = np.int64)
                y = np.array(batch_y[index], dtype = np.int64)
                x = Variable(torch.from_numpy(np.expand_dims(x,axis=1)))
                y = Variable(torch.from_numpy(y ))
                x, y = x.to(device), y.to(device)
                pre = rnn_model(x)
                loss += loss_fun(pre , y)
                if index == 0:
                    _, pre = torch.max(pre, dim=1)
                    print('prediction', pre.data.tolist())
                    print('b_y       ', y.data.tolist())
                    print('*' * 30)
            loss  = loss  / BATCH_SIZE
            print("epoch  ",epoch,'batch number',batch,"loss is: ", loss.data.tolist())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(rnn_model.parameters(), 1)
            optimizer.step()

        if epoch % 5 ==0:
            torch.save(rnn_model.state_dict(), './poem_generator_rnn')
            print("finish  save model")


def to_word(predict, vocabs):
    sample = np.argmax(predict)
    # print(sample)
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    # print(vocabs[sample])
    return vocabs[sample]

def pretty_print_poem(poem):
    shige=[]
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    # print(shige)
    poem_sentences = poem.split('。')
    print("".join(shige))
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')

def gen_poem(begin_word):
    # poems_vector, word_int_map, vocabularies = process_poems2('./tangshi.txt')  #  use the other dataset to train the network
    poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
    word_embedding = rnn_lstm.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
    rnn_model = rnn_lstm.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,
                                   embedding_dim=100, lstm_hidden_dim=128)

    rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))

    # 指定开始的字
    rnn_model.to(device)#GPU呗
    poem = begin_word
    word = begin_word
    while word != end_token:
        input = np.array([word_int_map[w] for w in poem],dtype= np.int64)
        input = Variable(torch.from_numpy(input)).to(device)
        output = rnn_model(input, is_test=True)
        word = to_word(output.data.tolist()[-1], vocabularies)
        poem += word

        if len(poem) > 30:
            return poem
    return poem
    
# run_training()  # 如果不是训练阶段 ，请注销这一行 。 网络训练时间很长。

pretty_print_poem(gen_poem("日"))
pretty_print_poem(gen_poem("红"))
pretty_print_poem(gen_poem("山"))
pretty_print_poem(gen_poem("夜"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("海"))
pretty_print_poem(gen_poem("月"))
pretty_print_poem(gen_poem("君"))
