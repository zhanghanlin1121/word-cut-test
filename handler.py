#coding=utf-8
import re
import time
import codecs
import sys
import io
import logging
import numpy as np
import tensorflow as tf
import xlrd
MAX_LEN  = 80

def strQ2B(string):
    """全角转半角"""
    rstring = u""
    for uchar in string:
        inside_code=ord(uchar)
        if inside_code == 12288:#全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

class Handler(object):
    def __init__(self,paras):
        self.paras = paras

    def init(self,): 
        saver = tf.train.import_meta_graph(self.paras['model_path']+".meta")
        self.sess =  tf.Session()
        with self.sess.as_default():
            saver.restore(self.sess, self.paras['model_path'])
        
        self.input_x = tf.get_default_graph().get_tensor_by_name('input_placeholder:0')
        self.unary_scores = tf.get_default_graph().get_tensor_by_name('Reshape_7:0')
        self.transitions = tf.get_default_graph().get_tensor_by_name('transitions:0')
        self.word2idx = self.load_word2idx(self.paras['char2vec_path'])
    
    def load_word2idx(self,word2idx_path):
        word2idx = {}
        f = codecs.open(word2idx_path,'r','utf-8')
        for line in f:
            line = line.strip()
            parts = line.split()
            if len(parts) != 2:
                continue
            word2idx[parts[0]] = int(parts[1])
        return word2idx

    def process(self,data):
        try:
            msg_id = '0'
            msg_id = data.get('msg_id','0')
            res = self._process(data) 
            return {'cut_msg':res}
        except Exception as e:
            logging.error('msg_id: %s %s',msg_id,e)
            return {'cut_msg':'','status':'error'}
    
    def _process(self,data):
        st = time.time()
        msg_id = data.get('msg_id','0')
        msg = data.get('msg','').strip()
        msg = strQ2B(msg)
        logging.info('get msg: %s',msg)
        sub_sentences = self.pre_process(msg)
        all_viterbi_sequences = []
        for ss in sub_sentences:
            input_x = []
            for word in ss:
                input_x.append(self.word2idx.get(word,1))
            origin_len = len(input_x) 
            if origin_len < MAX_LEN:
                for i in range(MAX_LEN - origin_len):
                    input_x.append(0)
            unary_scores, transition_matrix = self.sess.run([self.unary_scores,self.transitions],{self.input_x:np.array([input_x])})
            viterbi_sequence,_ = self.viterbi_decode(unary_scores[0][:origin_len],transition_matrix)
            
            all_viterbi_sequences.extend(viterbi_sequence)
        res = self.post_process(msg,all_viterbi_sequences) 
        et = time.time()
        logging.info('succ process %s cost %s ms %s',msg_id,(et-st)*1000,res)
        return res

    def pre_process(self,msg):
        st = time.time()
        res = []
        if len(msg) <= MAX_LEN:
            et = time.time()
            return [msg]
        else:
            left = 0
            while True:
                    for i in range(left,min((left + MAX_LEN),len(msg)))[::-1]:
                        if msg[i] in [u',',u';',u'”',u']',u'】',u')',u'、',u'>',u'》',u'。',u'?',u'、',u'~']:
                            res.append(msg[left:i+1])
                            left = i + 1
                            break
                    if i == left:
                        for i in range(left,min((left + MAX_LEN),len(msg)))[::-1]:
                            if msg[i] in [u' ',u'\t']:
                                res.append(msg[left:i+1])
                                left =  i + 1
                                break
                    if i == left:
                        res.append(msg[left:left+MAX_LEN])
                        left = left+MAX_LEN
                    if left >= len(msg):
                        break
                    if left + MAX_LEN >= len(msg):
                        res.append(msg[left:len(msg)])
                        break
            et = time.time()
            return res

    def post_process(self,msg,all_viterbi_sequences):
        '''
        wait to be done
        '''
        st = time.time()
        res = u''
        last_tag = 0
        for i,tag in enumerate(all_viterbi_sequences):
            tag = int(tag)
            if last_tag in [1,2] and msg[i] in [' ','\t']:
                res += ' '
            if msg[i] == ' ':
                continue
            
            #分数,小数,时分    
            if re.match('[0-9]',msg[i]) and re.match('[/\.:]',msg[min((i+1),len(msg)-1)]) and re.match('[0-9]',msg[min((i+2),len(msg)-1)]):#分数
                res += msg[i]
            #分数,小数,时分    
            elif re.match('[/\.:]',msg[i]) and re.match('[0-9]',msg[min((i+1),len(msg)-1)]) and re.match('[0-9]',msg[min((i-1),len(msg)-1)]):#分数
                res += msg[i]
            #f-35,email
            elif re.match('[0-9a-zA-Z]',msg[i]) and re.match('[0-9a-zA-Z\-@_+]',msg[min((i+1),len(msg)-1)]):#f35
                res += msg[i]
            elif re.match('[-]',msg[i]) and (re.match('[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}.*',msg[(i+1):]) or re.match('[0-9]{4}年.*',msg[(i+1):])) :#-2016-12-10
                res += ' '+ msg[i] + ' '
            elif re.match('[\-@_+]',msg[i]) and re.match('[0-9a-zA-Z\-@_]',msg[min((i+1),len(msg)-1)]) and re.match('[0-9a-zA-Z\-@_]',msg[max((i-1),0)]):#f35
                res += msg[i]
            #mail.com s.s(not s.. s.@) 
            elif re.match('[0-9a-zA-Z\-@_]',msg[i]) and re.match('[0-9a-zA-Z\-@_.]',msg[min((i+1),len(msg)-1)]) and re.match('[0-9a-zA-Z]',msg[min((i+2),len(msg)-1)]):#
                res += msg[i]
            #country. 中    
            elif re.match('[0-9a-zA-Z\-@_]',msg[i]) and re.match('[0-9a-zA-Z\-@_.]',msg[min((i+1),len(msg)-1)]) and not re.match('[0-9a-zA-Z]',msg[min((i+2),len(msg)-1)]):#
                res += msg[i]+' '
            elif re.match('[\-@_]',msg[i]) and not re.match('[0-9a-zA-Z]',msg[min((i+2),len(msg)-1)]):#
                res += msg[i]+' '
            #baidu/toutiao/
            elif re.match('[\D]',msg[i]) and re.match('[/]',msg[min((i+1),len(msg)-1)]):#
                res += msg[i] +' '
            #baidu/toutiao/
            elif re.match('[/]',msg[i]) and re.match('[\D]',msg[min((i+1),len(msg)-1)]):#
                res += msg[i] +' '
            elif re.match('[.]',msg[i]):
                if re.match('[0-9a-zA-Z]',msg[min((i+1),len(msg)-1)]):
                    #print i,msg[i],tag,5
                    res += msg[i]
                else:
                    #print i,msg[i],tag,6
                    res += msg[i] +' '
            
            elif tag == 0:
                res += msg[i]+' '
            elif  tag == 1:
                if msg[i] == u'[' or msg[i] == u']' or msg[i] == u'。':
                    res += msg[i] + ' '
                else:    
                    res += msg[i]
            elif tag == 2:
                if msg[i] == u'[' or msg[i] == u']' or msg[i] == u'。':
                    res += ' '+msg[i] + ' '
                else:    
                    res += msg[i]
            elif tag == 3:
                if msg[i] == u'[' or msg[i] == u']' or msg[i] == u'。':
                    res += ' '+msg[i] + ' '
                else:    
                    res += msg[i] + ' '
            last_tag = tag
        et = time.time()
        return ' '.join(res.split())

    def viterbi_decode(self,score, transition_params):
        trellis = np.zeros_like(score)
        backpointers = np.zeros_like(score, dtype=np.int32)
        trellis[0] = score[0]
        for t in range(1, score.shape[0]):
          v = np.expand_dims(trellis[t - 1], 1) + transition_params
          trellis[t] = score[t] + np.max(v, 0)
          backpointers[t] = np.argmax(v, 0)
        viterbi = [np.argmax(trellis[-1])]
        for bp in reversed(backpointers[1:]):
          viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = np.max(trellis[-1])
        return viterbi, viterbi_score

def listx(num):
    lis = []
    for j in range(num): lis.append(j)
    return lis

if __name__ == "__main__":
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')  # 改变标准输出的默认编码
    paras = {'model_path':'./model/model-1450','char2vec_path':'./model/basic_vocab.txt'}
    obj = Handler(paras)
    obj.init()
    data_set_fp = xlrd.open_workbook("word_set.xlsx",encoding_override='utf-8')
    all_question = data_set_fp.sheet_by_name(u'all_question')
    questions_rows = all_question.nrows
    all_questions = listx(questions_rows)
    print ("需要处理: "+str(questions_rows))
    for i in range(0, questions_rows):
        all_questions[i] = all_question.row_values(i)[0]


    sentence = '2015-2017-10-11 -20 -2015 -2015-10-12 -2015~10-12 ~2015-12-12，-2016年,;；www@hotmai.com您反馈的问题我已详细记录 , 稍后会反馈至后台专员跟进处理,次日会有95118号码与您取得联系,请您保持手机畅通,谢谢审核'
    output = open('training.txt', 'w+' ,encoding='utf-8')

    for i in range(0,questions_rows):
        print("需要处理: " + str(i))
        # gbk_str = all_questions[i].encode('gbk')
        data = {'msg':str(all_questions[i])}
        output.write(obj.process(data)['cut_msg'] + "\n")
    output.close()
