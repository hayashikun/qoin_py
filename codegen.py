import os
from glob import glob

from grpc_tools import protoc


def gen(qoin_path):
    if not os.path.exists('proto'):
        os.makedirs("proto")

    include_path = "/usr/local/include/"
    qoin_proto_files = glob(os.path.join(os.path.join(qoin_path, 'qoin', 'proto'), "**", "*.proto"), recursive=True)

    protoc.main(
        (
            '',
            f'-I={qoin_path}:{include_path}',
            '--python_out=./proto',
            '--grpc_python_out=./proto',
            *qoin_proto_files,
        )
    )

    if not os.path.exists('proto/__init__.py'):
        with open('proto/__init__.py', 'w') as fp:
            fp.write("import sys\nfrom pathlib import Path\nsys.path.append(str(Path(__file__).parent))")


if __name__ == "__main__":
    gen("../qoin", )
