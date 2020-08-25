import grpc
import numpy as np
import pyautogui

from proto.qoin.proto import hello_pb2_grpc, hello_pb2, face_mesh_pb2, face_mesh_pb2_grpc, hand_tracking_pb2, \
    hand_tracking_pb2_grpc


def run_hello():
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
    channel = grpc.insecure_channel('localhost:50051')
    stub = face_mesh_pb2_grpc.FaceMeshStub(channel)
    response = stub.FaceMeshStream(face_mesh_pb2.FaceMeshRequest())
    for res in response:
        print(res)


def run_hand_tracking():
    screen_x, screen_y = pyautogui.size()
    pyautogui.FAILSAFE = False

    channel = grpc.insecure_channel('localhost:50051')
    stub = hand_tracking_pb2_grpc.HandTrackingStub(channel)
    response = stub.HandTrackingStream(hand_tracking_pb2.HandTrackingRequest())
    for res in response:
        pos = np.array([[lm.x, lm.y] for lm in res.landmark_list.landmark])
        xy = pos.mean(axis=0)
        if (xy > 0.0).all():
            pyautogui.moveTo(screen_x * xy[0], screen_y * xy[1], duration=0)


if __name__ == '__main__':
    run_hand_tracking()
