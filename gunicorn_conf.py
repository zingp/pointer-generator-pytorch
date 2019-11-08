#import logging
#import logging.handlers
#from logging.handlers import WatchedFileHandler
#import os
#import multiprocessing
#bind = '127.0.0.1:6666'      # 绑定ip和端口号
bind = '0.0.0.0:6666'      # 绑定ip和端口号
backlog = 512                # 监听队列
chdir = './'  # gunicorn要切换到的目的工作目录
timeout = 30      # 超时
worker_class = 'gevent' # 使用gevent模式，还可以使用sync 模式，默认的是sync模式

workers = 10       # 进程数
threads = 2       # 指定每个进程开启的线程数
loglevel = 'info' # 日志级别，这个日志级别指的是错误日志的级别，而访问日志的级别无法设置

access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'    # 设置gunicorn访问日志格式，错误日志无法设置
accesslog = "./logs/gunicorn_access.log"      #访问日志文件
errorlog = "./logs/gunicorn_error.log"        #错误日志文件

