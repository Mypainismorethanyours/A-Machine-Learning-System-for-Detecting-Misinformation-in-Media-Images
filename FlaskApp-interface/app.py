#interface for handling user image upload and displaying the results of the model inference on that image
 from flask import Flask import, render_template,request, jsonify
 import os 
 import requests
 import uuid 
 from werkzeug.utils import secure_filename
 import time 
 import logging 
 from prometheus client import Counter,His
