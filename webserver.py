#! -*- coding: utf-8 -*-
import os
import time
import json
from flask import views
from flask import Flask
from flask import request
from flask import render_template

import config
from data import Vocab
from predict import build_batch_by_article
from predict import BeamSearch

model_path = "./logs/weibo/train_20191030_005155/model/model_510000_20191030_014457"
vocab = Vocab(config.vocab_path, config.vocab_size)
beam_processor = BeamSearch(model_path, vocab)


app = Flask(__name__)     # 创建一个Flask对象，__name__传成其他字符串也行。

class AbstractView(views.MethodView):
    methods = ['GET', 'POST']
 
    def get(self):
        return render_template('summary.html')
 
    def post(self):
        ret = {"status":True, "error": None, "abstract":""}
        art = request.form.get("article")
        print("原文：", art)
        start = time.time()
        try:
            batch = build_batch_by_article(art, vocab)
            abstract = beam_processor.decode(batch)
            ret["abstract"] = abstract
        except Exception as e:
            ret["error"] = e
            ret["status"] = False
        
        end = time.time()
        print("time:", end-start)
        print("摘要：", abstract)
        return json.dumps(ret)
 
app.add_url_rule('/abstract.html', view_func=AbstractView.as_view(name='abstract'))

@app.route('/')
def hello_world():
    return 'Hello World!'
 
if __name__ == '__main__':
    host = '0.0.0.0'
    port = 6666
    app.run(host=host, port=port)
