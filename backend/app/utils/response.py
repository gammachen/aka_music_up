from flask import jsonify

def make_response(data=None, message='success', code=200):
    return jsonify({
        'success': code == 200,
        'code': code,
        'message': message,
        'data': data
    })