import threading
import requests
import json

from flask import Flask, request, Response, make_response
from werkzeug.serving import make_server


class FineServerThread(threading.Thread):
    def __init__(self, app):
        threading.Thread.__init__(self)
        self.server = make_server("0.0.0.0", 5000, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        print('starting server')
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()


class FineServer(Flask):
    def __init__(self):
        Flask.__init__(self, __name__)

        self.pose_callback = None
        self.canny_callback = None
        self.invoked_callback = None

        # make routes
        self.route("/", methods=["GET"])(self.root)
        self.route("/api/pose", methods=["GET"])(self.getPose)
        self.route("/api/canny", methods=["GET"])(self.getCanny)
        self.route('/api/invoked', methods=['POST'])(self.handleResult)

        self.server = FineServerThread(self)
        self.server.start()
        print('server started')

        self.invoke_thread = None

    def requestStop(self):
        self.server.shutdown()

    def root(self):
        return "Nothing here to see"

    def getPose(self):
        if self.pose_callback is not None:
            image_data = self.pose_callback()
        else:
            image_data = None

        response = make_response(image_data)

        response.headers.set('Content-Type', 'image/png')
        return response

    def getCanny(self):
        if self.canny_callback is not None:
            image_data = self.canny_callback()
        else:
            image_data = None

        response = make_response(image_data)

        response.headers.set('Content-Type', 'image/png')
        return response

    def handleResult(self):
        # TODO: validate request and handle exceptions

        data = request.files["image"].read()

        if self.invoked_callback is not None:
            self.invoked_callback(data)

        return Response(status=200)

    def invokeGeneration(self, host, batch_file):
        self.invoke_thread = threading.Thread(target=self.invokeGenerationFunction, kwargs={"host":host, "batch_file": batch_file})
        self.invoke_thread.start()

    def invokeGenerationFunction(self, host, batch_file):
        with open(batch_file) as fp:
            batch_data = json.load(fp)

        url = f"http://{host}/api/v1/queue/default/enqueue_batch"
        response = requests.post(url, json=batch_data)
        print(response)