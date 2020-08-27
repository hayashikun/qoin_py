import fire
import grpc
import numpy as np
import pyautogui


class QoinPy:
    def __init__(self, host="localhost", port=50051):
        self._channel = grpc.insecure_channel(f'{host}:{port}')

    def run_hello(self):
        """
        Run hello
        :return:
        """
        from proto.qoin.proto import hello_pb2_grpc, hello_pb2
        stub = hello_pb2_grpc.GreeterStub(self._channel)
        response = stub.SayHello(hello_pb2.HelloRequest(name='you'))
        print("Greeter client received: ", response.message)
        response = stub.SayHelloAgain(hello_pb2.HelloRequest(name='you'))
        print("Greeter client received: ", response.message)
        response = stub.HelloStream(hello_pb2.HelloRequest(name='hayashi'))
        for res in response:
            print(res)

    def _face_mesh(self, receive_handler, post_handler):
        """
        Run face_mesh
        :return:
        """
        from proto.qoin.proto import face_mesh_pb2, face_mesh_pb2_grpc
        stub = face_mesh_pb2_grpc.FaceMeshStub(self._channel)
        response = stub.FaceMeshStream(face_mesh_pb2.FaceMeshRequest())
        try:
            for res in response:
                xyz = np.array([
                    [[lm.x, lm.y, lm.z] for lm in lml.landmark]
                    for lml in res.landmark_list])
                receive_handler(xyz)
        except Exception as e:
            print(e)
        post_handler()

    def run_face_mesh(self, landmark_save_path=None, suppress_dump=False):
        landmarks = list()

        def receive_handler(xyz):
            if not suppress_dump:
                print(xyz)
            if landmark_save_path is not None:
                landmarks.append(xyz)

        def post_handler():
            if landmark_save_path is not None:
                np.save(landmark_save_path, np.array(landmarks))

        self._face_mesh(receive_handler, post_handler)

    def right_or_left(self):
        def diff(lm):
            hist, x, y = np.histogram2d(lm[:, 0], lm[:, 1], bins=10)
            return hist[: len(x) // 2, :].sum() - hist[len(x) // 2:, :].sum()

        sampling_len = 20
        state = {
            "avg": 0,
            "sampling": np.empty(shape=sampling_len),
            "idx": 0,
            "direction": "center"
        }
        threshold = 300

        def receive_handler(xyz):
            d = diff(xyz[0])
            if state["idx"] < 20:
                state["sampling"][state["idx"]] = d
            elif state["idx"] == 20:
                state["avg"] = state["sampling"].mean()
                print("sampling done")
            else:
                d -= state["avg"]
                if state["direction"] == "center" and d < -threshold:
                    state["direction"] = "right"
                    print("right")
                elif state["direction"] == "center" and d > threshold:
                    state["direction"] = "left"
                    print("left")
                elif state["direction"] != "center" and np.abs(d) < threshold:
                    state["direction"] = "center"

            state["idx"] += 1

        def post_handler():
            pass

        self._face_mesh(receive_handler, post_handler)

    def _hand_tracking(self, receive_handler, post_handler):
        from proto.qoin.proto import hand_tracking_pb2, hand_tracking_pb2_grpc
        stub = hand_tracking_pb2_grpc.HandTrackingStub(self._channel)
        response = stub.HandTrackingStream(hand_tracking_pb2.HandTrackingRequest())
        try:
            for res in response:
                xyz = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark_list.landmark])
                receive_handler(xyz)
        except Exception as e:
            print(e)
        post_handler()

    def run_hand_tracking(self, landmark_save_path=None, suppress_dump=False):
        """
        Run hand_tracking
        :param landmark_save_path: If not None, landmark xyz data is saved at the path.
        :param suppress_dump: If True, landmark xyz is not dumped.
        :return:
        """
        landmarks = list()

        def receive_handler(xyz):
            if not suppress_dump:
                print(xyz)
            if landmark_save_path is not None:
                landmarks.append(xyz)

        def post_handler():
            if landmark_save_path is not None:
                np.save(landmark_save_path, np.array(landmarks))

        self._hand_tracking(receive_handler, post_handler)

    def move_cursor_with_hand(self):
        """
        Move mouse cursor with hand using hand_tracking
        :return:
        """
        screen_x, screen_y = pyautogui.size()
        pyautogui.FAILSAFE = False

        def receive_handler(pos):
            xy = pos.mean(axis=0)
            if (xy > 0.0).all():
                pyautogui.moveTo(screen_x * xy[0], screen_y * xy[1], duration=0)

        def post_handler():
            pass

        self._hand_tracking(receive_handler, post_handler)


if __name__ == '__main__':
    fire.Fire(QoinPy)
