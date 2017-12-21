#coding=utf-8
import sys
sys.path.append('../gen-py')
from message_process.ttypes import *
from message_process import MessageProcess

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
import traceback

class ThriftProcessor(object):
    def __init__(self, prop):
        self.host = prop['host']
        self.port = int(prop['port'])
        self.transport = TSocket.TSocket(self.host, self.port)
        self.transport = TTransport.TBufferedTransport(self.transport)

        self.protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
        self.client = MessageProcess.Client(self.protocol)

    def process(self, data):
        try:
            self.transport.open()
            rsp = self.client.process(data)
            return rsp
        except:
            print traceback.format_exc()
        finally:
            self.transport.close()

def test(data):
    obj = ThriftProcessor({'host':'0.0.0.0','port':3366})
    res = obj.process(data)
    print res['cut_msg']

if __name__ == "__main__":
    sent = '京东是一家伟大的公司'
    sent = '1 、已承保的配送方式为京东配送货到付款的(在线支付)订单;(2016年12月27日部分商家经自行选择后可实现货到付款+在线支付订单均投保,但仍'
    #sent = '-2017 2017-2018 2017 -2018 t+1 xxxwfa2434@163.com 12adshfkasdf9303q232@hotmail.com http://www.baidu.com/toutiao/index.html 5s... s.- s 3.5s京东。3.8 f35 su-35是 .一家iphone 6企业china is a great countrt. 2017年8月5日 14:30:40  china will develop well!. 你 .. 怎 ..么看...  。。。 2017-02-02 你好 哇哈哈哈哈哈哈 。。。50 %   80%'
    data = {'msg':sent}
    test(data)




