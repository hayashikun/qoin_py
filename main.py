import grpc
import numpy as np
import pyautogui


def run_hello():
    from proto.qoin.proto import hello_pb2_grpc, hello_pb2
    channel = grpc.insecure_channel('localhost:50051')
    stub = hello_pb2_grpc.GreeterStub(channel)
    response = stub.SayHello(hello_pb2.HelloRequest(name='you'))
    print("Greeter client received: ", response.message)
    response = stub.SayHelloAgain(hello_pb2.HelloRequest(name='you'))
    print("Greeter client received: ", response.message)
    response = stub.HelloStream(hello_pb2.HelloRequest(name='hayashi'))
    for res in response:
        print(res)


def run_face_mesh():
    from proto.qoin.proto import face_mesh_pb2, face_mesh_pb2_grpc
    channel = grpc.insecure_channel('localhost:50051')
    stub = face_mesh_pb2_grpc.FaceMeshStub(channel)
    response = stub.FaceMeshStream(face_mesh_pb2.FaceMeshRequest())
    for res in response:
        print(res)


def _hand_tracking(receive_handler, post_handler):
    from proto.qoin.proto import hand_tracking_pb2, hand_tracking_pb2_grpc
    channel = grpc.insecure_channel('localhost:50051')
    stub = hand_tracking_pb2_grpc.HandTrackingStub(channel)
    response = stub.HandTrackingStream(hand_tracking_pb2.HandTrackingRequest())
    try:
        for res in response:
            pos = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark_list.landmark])
            receive_handler(pos)
    except Exception as e:
        print(e)
    post_handler()


def run_hand_tracking():
    positions = list()

    def receive_handler(pos):
        print(pos)
        positions.append(pos)

    def post_handler():
        pass
        # np.save("positions", np.array(positions))

    _hand_tracking(receive_handler, post_handler)


def move_cursor():
    screen_x, screen_y = pyautogui.size()
    pyautogui.FAILSAFE = False

    def receive_handler(pos):
        xy = pos.mean(axis=0)
        if (xy > 0.0).all():
            pyautogui.moveTo(screen_x * xy[0], screen_y * xy[1], duration=0)

    def post_handler():
        pass

    _hand_tracking(receive_handler, post_handler)


if __name__ == '__main__':
    run_hand_tracking()
