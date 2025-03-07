
from flask import Blueprint

plant_prediction_bp = Blueprint('plant_prediction', __name__,
                                    template_folder='templates',
                                    static_folder='static')
from .routes import *