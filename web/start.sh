#!/bin/bash
nohup gunicorn -c ./conf/gunicorn_conf.py webserver:app >logs/nohup.log 2>&1 &

