from . import routes
from flask import Blueprint

space_identification_bp = Blueprint('plant_prediction', __name__,
                                    template_folder='templates',
                                    static_folder='static')
