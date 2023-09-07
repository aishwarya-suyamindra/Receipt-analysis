from flask import Flask, jsonify

class ApiError(Exception):
    pass

class BadRequest(ApiError):
    code = 400
    description = "Bad request"

class PageNotFound(ApiError):
    code = 404
    description = "Page Not Found"

class UnsupportedFileFormat(ApiError):
    code = 415
    description = "Unsupported Media Format"

class InternalServerError(ApiError):
    code = 500
    description = "Internal Server Error"

def handle_error(error):
    response = {"error": error.description, "message": ""}
    if len(error.args) > 0:
        response["message"] = error.args[0]
    return jsonify(response), error.code
