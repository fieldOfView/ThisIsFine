import threading
import requests
import json
import uuid

from flask import Flask, request, Response, make_response
from werkzeug.serving import make_server


class FineServerThread(threading.Thread):
    """
    A class representing a threaded server for the FineServer application.

    Args:
        app (Flask): The Flask application object.

    Attributes:
        server (WSGIServer): The WSGI server instance.
        ctx (AppContext): The application context.

    Methods:
        run(): Starts the server and serves requests indefinitely.
        shutdown(): Shuts down the server.

    """

    def __init__(self, app: Flask) -> None:
        threading.Thread.__init__(self)
        self.server = make_server("0.0.0.0", 5000, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self) -> None:
        """
        Starts the server and serves requests indefinitely.
        """
        print('starting server')
        self.server.serve_forever()

    def shutdown(self) -> None:
        """
        Shuts down the server.
        """
        self.server.shutdown()


class FineServer(Flask):
    """
    FineServer class extends Flask and represents a server for handling requests related to pose, canny, and result handling.

    Attributes:
        pose_callback (function): Callback function for retrieving pose image data.
        canny_callback (function): Callback function for retrieving canny image data.
        invoked_callback (function): Callback function for handling the result of an invocation.
        server (FineServerThread): Thread for running the server.
        invoke_thread (threading.Thread): Thread for invoking generation.

    Methods:
        __init__(): Initializes the FineServer instance.
        request_stop(): Requests the server to stop.
        root(): Handles the root route.
        get_pose(): Handles the /api/pose route.
        get_canny(): Handles the /api/canny route.
        handle_result(): Handles the /api/invoked route.
        invoke_generation(): Invokes generation by starting the invoke_thread.
        invoke_generation_function(): Function for invoking generation.
    """

    def __init__(self) -> None:
        Flask.__init__(self, __name__)

        self.pose_callback = None
        self.canny_callback = None
        self.invoked_callback = None

        # make routes
        self.route("/", methods=["GET"])(self.root)
        self.route("/api/pose", methods=["GET"])(self.get_pose)
        self.route("/api/canny", methods=["GET"])(self.get_canny)
        self.route('/api/invoked', methods=['POST'])(self.handle_result)

        self.server = FineServerThread(self)
        self.server.start()
        print('server started')

        self.invoke_thread = None

    def request_stop(self) -> None:
        """
        Requests the server to stop by shutting down the server thread.
        """
        self.server.shutdown()

    def root(self) -> str:
        """
        Handles the root route ("/") and returns a string response.
        """
        return "Nothing here to see"

    def get_pose(self):
        """
        Handles the /api/pose route and returns the pose image data as a response.
        """
        if self.pose_callback is not None:
            image_data = self.pose_callback()
        else:
            image_data = None

        response = make_response(image_data)

        response.headers.set('Content-Type', 'image/png')
        return response

    def get_canny(self):
        """
        Handles the /api/canny route and returns the canny image data as a response.
        """
        if self.canny_callback is not None:
            image_data = self.canny_callback()
        else:
            image_data = None

        response = make_response(image_data)

        response.headers.set('Content-Type', 'image/png')
        return response

    def handle_result(self):
        """
        Handles the /api/invoked route and processes the result of an invocation.
        """
        # TODO: validate request and handle exceptions

        data = request.files["image"].read()

        if self.invoked_callback is not None:
            self.invoked_callback(data)

        return Response(status=200)

    def invoke_generation(self, host, workflow_file):
        """
        Invokes generation without locking up the main thread by starting a new thread.

        Args:
            host (str): The host URL for the generation API.
            workflow_file (str): The path to the workflow file.

        """
        self.invoke_thread = threading.Thread(target=self.invoke_generation_function, kwargs={"host":host, "workflow_file": workflow_file})
        self.invoke_thread.start()

    def invoke_generation_function(self, host, workflow_file):
        """
        Function for invoking generation.

        Args:
            host (str): The host URL for the generation API.
            workflow_file (str): The path to the workflow file.

        """
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
