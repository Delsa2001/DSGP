from flask import Blueprint

space_identification_bp = Blueprint('space_identification', __name__)

from . import routes
