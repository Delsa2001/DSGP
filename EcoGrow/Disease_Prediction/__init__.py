from flask import Blueprint

disease_identification_bp = Blueprint('disease_identification', __name__ ,template_folder='templates')
from .routes import *