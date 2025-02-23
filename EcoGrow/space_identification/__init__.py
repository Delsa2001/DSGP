from flask import Blueprint

space_identification_bp = Blueprint('space_identification', __name__,
                                    template_folder='templates',
                                    static_folder='static'  )

from . import routes
