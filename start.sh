#!/bin/bash
nohup gunicorn -c ./gunicorn_conf.py webserver:app >logs/nohup.log 2>&1 &
