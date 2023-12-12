import threading
import requests
import json
import uuid

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

    def invokeGeneration(self, host, workflow_file):
        self.invoke_thread = threading.Thread(target=self.invokeGenerationFunction, kwargs={"host":host, "workflow_file": workflow_file})
        self.invoke_thread.start()

    def invokeGenerationFunction(self, host, workflow_file):
        with open(workflow_file) as fp:
            workflow_data = json.load(fp)

        # create a graph and batch from the workflow

        batch_nodes = {}
        for node in workflow_data["nodes"]:
            node_data = node["data"]
            node_inputs = {}

            for input_key in node_data["inputs"]:
                input_data = node_data["inputs"][input_key]
                if "value" in input_data:
                    node_inputs[input_key] = input_data["value"]

            del node_data["inputs"]
            del node_data["outputs"]
            node_data.update(node_inputs)

            batch_nodes[node["id"]] = node_data

        batch_data = {
            "batch": {
                "graph": {
                    "id": str(uuid.uuid4()),
                    "nodes": batch_nodes,
                    "edges": [
                        {
                            "source": {
                                "node_id": edge["source"],
                                "field": edge["sourceHandle"],
                            },
                            "destination": {
                                "node_id": edge["target"],
                                "field": edge["targetHandle"],
                            },
                        }
                        for edge in workflow_data["edges"]
                        if edge["type"] == "default"
                    ],
                },
                "runs": 1,
            },
            "prepend": False,
        }

        url = f"http://{host}/api/v1/queue/default/enqueue_batch"
        response = requests.post(url, json=batch_data)
        print(response)