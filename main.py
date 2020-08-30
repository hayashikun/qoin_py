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

    def babiniku(self):
        """
        Be bisyoujo
        
        :return:
        """
        import cv2
        from face_mesh_parts import landmark_parts_index
        raw_img = cv2.imread("res/face_base.png", -1)

        img_size = (1000, 1000)
        base_face_center = (500, 590)
        base_face_size = (280, 260)

        raw_img = cv2.resize(raw_img, img_size)
        raw_img[raw_img[:, :, -1] == 0] = 220, 200, 255, 255

        eye_size = (100, 60)
        left_eye = cv2.imread("res/left_eye.png", -1)
        right_eye = cv2.imread("res/right_eye.png", -1)
        eyes_img = [cv2.resize(left_eye, eye_size), cv2.resize(right_eye, eye_size)]

        def receive_handler(xyz):
            x, y = xyz[0][:, :2].T
            x -= x.mean()
            y -= y.mean()
            img = raw_img.copy()

            lips_y = np.array([
                y[landmark_parts_index["upper_lip"]].mean(), y[landmark_parts_index["lower_lip"]].mean()
            ])

            l_idx = landmark_parts_index["left_eye"]
            r_idx = landmark_parts_index["right_eye"]
            eyes = np.array([
                [x[l_idx].mean(), y[l_idx].mean()],
                [x[r_idx].mean(), y[r_idx].mean()],
            ])
            face_angle = -np.arctan((eyes[1, 1] - eyes[0, 1]) / (eyes[1, 0] - eyes[0, 0]))

            f_idx = landmark_parts_index["face_oval"]
            face_center = np.array([x[f_idx].mean(), y[f_idx].mean()])
            mat = cv2.getRotationMatrix2D(tuple(face_center), np.rad2deg(- face_angle), 1)
            reversed_face_pts = mat @ np.array([
                x[f_idx],
                y[f_idx],
                np.ones_like(f_idx)
            ])
            x_scale = base_face_size[0] / (reversed_face_pts[0, :].max() - reversed_face_pts[0, :].min())
            y_scale = base_face_size[1] / (reversed_face_pts[1, :].max() - reversed_face_pts[1, :].min())

            for e, ei in zip(eyes, eyes_img):
                ex, ey = mat @ np.array([e[0], e[1], 1])
                ex = int(ex * x_scale + base_face_center[0])
                ey = int(ey * y_scale + base_face_center[1])
                sx = ex - eye_size[0] // 2, ex + eye_size[0] // 2
                sy = ey - eye_size[1] // 2, ey + eye_size[1] // 2
                tr = ei[:, :, -1] / 255
                for i in range(3):
                    img[sy[0]:sy[1], sx[0]:sx[1], i] = ei[:, :, i] * tr + img[sy[0]:sy[1], sx[0]:sx[1], i] * (1 - tr)

            lips_center = (base_face_center[0], (base_face_center[1] + lips_y[1] * y_scale).astype(np.int))
            lips_radius = np.abs((lips_y[1] - lips_y[0]) * y_scale).astype(np.int)
            img = cv2.circle(img, lips_center, lips_radius, color=(180, 120, 255), thickness=-1)
            img = cv2.circle(img, lips_center, lips_radius, color=(0, 0, 0), thickness=1)

            l_idx = landmark_parts_index["left_eyebrow"]
            r_idx = landmark_parts_index["right_eyebrow"]
            reversed_eyebrow_pts = [
                mat @ np.array([x[l_idx], y[l_idx], np.ones_like(l_idx)]),
                mat @ np.array([x[r_idx], y[r_idx], np.ones_like(r_idx)])
            ]

            for eb in reversed_eyebrow_pts:
                ebx = (eb[0] * x_scale + base_face_center[0])
                eby = (eb[1] * y_scale + base_face_center[1])
                eby = (eby.min() + (eby - eby.min()) * 0.25)
                img = cv2.fillConvexPoly(img, np.array([ebx.astype(np.int), eby.astype(np.int)]).T, color=(50, 50, 50))

            mat = cv2.getRotationMatrix2D((img_size[0] // 2, img_size[1]), np.rad2deg(face_angle) // 4, 1)
            img = cv2.warpAffine(img, mat, img_size)

            cv2.imshow("babiniku", img)
            cv2.waitKey(1)

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
