#!/bin/bash

echo "Initializasing pipe..."

sh src/pre.sh
python src/main.py

echo "Pipe process completed"
