#coding:utf-8
import sys
sys.path.append('./gen-py')
import time
import json

import logging
logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler = logging.FileHandler("word_cut.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

from handler import Handler

from message_process.ttypes import *
from message_process import MessageProcess

from thrift.server import TServer
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server.TProcessPoolServer import TProcessPoolServer

class ThrftServer(object):
    def __init__(self,paras):
        self.handler = Handler(paras)
    
    def run(self,port):
        processor = MessageProcess.Processor(self.handler)
        transport = TSocket.TServerSocket('0.0.0.0',port)
        tfactory = TTransport.TBufferedTransportFactory()
        pfactory = TBinaryProtocol.TBinaryProtocolFactory()
        server = TProcessPoolServer(processor, transport, tfactory, pfactory)
        #server.setNumWorkers(TOTAL_WORKER_COUNT)
        server.setPostForkCallback(self._init_handler)

        logging.info("thrift_server start on %s ...",port)
        server.serve()

    def _init_handler(self,):
        self.handler.init()

if __name__ == '__main__':
    paras = {'model_path':'./model/model-1450','char2vec_path':'./model/basic_vocab.txt'}
    server = ThrftServer(paras)
    print 'cut_server ready'
    server.run(port=3366)
