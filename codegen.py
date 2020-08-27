import os
import shutil
from glob import glob

from grpc_tools import protoc


def gen(qoin_path, mediapipe_path=None):
    if os.path.exists('proto'):
        shutil.rmtree("proto")
    os.makedirs("proto")

    include_path = "/usr/local/include/"
    qoin_proto_files = glob(os.path.join(qoin_path, 'qoin', 'proto', '**', '*.proto'), recursive=True)
    if mediapipe_path is None:
        mediapipe_path = os.path.join(qoin_path, 'bazel-qoin', 'external', 'mediapipe')
        mediapipe_path = os.path.abspath(mediapipe_path)

    mediapipe_proto_files = glob(os.path.join(mediapipe_path, 'mediapipe', 'framework', '**', '*.proto'),
                                 recursive=True)
    protoc.main(
        (
            '',
            f'-I={qoin_path}:{mediapipe_path}:{include_path}',
            '--python_out=./proto',
            '--grpc_python_out=./proto',
            *qoin_proto_files,
            *mediapipe_proto_files
        )
    )

    if not os.path.exists('proto/__init__.py'):
        with open('proto/__init__.py', 'w') as fp:
            fp.write("import sys\nfrom pathlib import Path\nsys.path.append(str(Path(__file__).parent))")


if __name__ == "__main__":
    gen("../qoin", )
