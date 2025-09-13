import platform
import subprocess
import sys
import time
from pathlib import Path

import requests

CONTAINER_NAME = "qdrant_db"
IMAGE = "qdrant/qdrant:latest"
PORT_HTTP = 6333
PORT_GRPC = 6334
STORAGE_DIR = Path.cwd() / "qdrant_storage"
HEALTH_URL = f"http://localhost:{PORT_HTTP}/healthz"


def run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def docker_ok():
    return run(["docker", "version"]).returncode == 0


def container_exists(name):
    out = run(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"]
    )
    return name in out.stdout.splitlines()


def container_running(name):
    out = run(["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"])
    return name in out.stdout.splitlines()


def guess_platform():
    m = platform.machine().lower()
    return "linux/arm64" if ("arm" in m or "aarch" in m) else "linux/amd64"


def try_run_container(ports, with_volume=True):
    http_port, grpc_port = ports
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    volume_arg = f"{str(STORAGE_DIR)}:/qdrant/storage"
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "-p",
        f"{http_port}:6333",
        "-p",
        f"{grpc_port}:6334",
        "--platform",
        guess_platform(),
    ]
    if with_volume:
        cmd += ["-v", volume_arg]
    cmd += [IMAGE]

    res = run(cmd)
    if res.returncode == 0:
        cid = res.stdout.strip()
        print(
            f"Started container {CONTAINER_NAME} ({cid}) on ports {http_port}/{grpc_port}"
        )
        return True, ""
    else:
        return False, (res.stderr or res.stdout)


def main():
    if not docker_ok():
        print(
            "Docker isn’t available. Please start Docker Desktop/Engine and try again."
        )
        sys.exit(1)

    # Reuse or start existing container
    if container_exists(CONTAINER_NAME):
        if container_running(CONTAINER_NAME):
            print(f"Container '{CONTAINER_NAME}' is already running.")
        else:
            print(f"Starting existing container '{CONTAINER_NAME}'...")
            res = run(["docker", "start", CONTAINER_NAME])
            if res.returncode != 0:
                print("Failed to start existing container:\n", res.stderr or res.stdout)
                sys.exit(1)
        # health wait
        wait_health()
        return

    # Fresh run
    ok, err = try_run_container((PORT_HTTP, PORT_GRPC), with_volume=True)
    if not ok:
        msg = err.lower()

        # Name conflict (leftover)
        if (
            "conflict. the container name" in msg
            or "already in use by container" in msg
        ):
            print("A container with this name already exists. Removing it...")
            run(["docker", "rm", "-f", CONTAINER_NAME])
            ok, err = try_run_container((PORT_HTTP, PORT_GRPC), with_volume=True)

        # Port conflict -> try alternate ports
        if not ok and ("port is already allocated" in msg):
            alt_http, alt_grpc = PORT_HTTP + 2, PORT_GRPC + 2
            print(
                f"Ports {PORT_HTTP}/{PORT_GRPC} busy. Retrying on {alt_http}/{alt_grpc}..."
            )
            ok, err = try_run_container((alt_http, alt_grpc), with_volume=True)
            global HEALTH_URL
            HEALTH_URL = f"http://localhost:{alt_http}/healthz"

        # macOS mount denial -> retry without volume
        if not ok and (
            "mounts denied" in msg
            or "is not a shared mount" in msg
            or "operation not permitted" in msg
        ):
            print(
                "Volume mount was denied by Docker Desktop. Retrying without volume (data will be ephemeral)."
            )
            ok, err = try_run_container((PORT_HTTP, PORT_GRPC), with_volume=False)

        # Final failure
        if not ok:
            print("docker run failed. Full error:")
            print(err.strip())
            print("\nQuick fixes:")
            print(
                " • Check Docker Desktop > Settings > Resources > File sharing and add your project folder."
            )
            print(
                " • Stop other services using ports 6333/6334 or change PORT_HTTP/PORT_GRPC."
            )
            print(" • Remove stale container: docker rm -f", CONTAINER_NAME)
            print(" • On Apple Silicon, we already set --platform linux/arm64.")
            sys.exit(125)

    wait_health()


def wait_health():
    print("Waiting for Qdrant health:", HEALTH_URL)
    for _ in range(60):
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            if r.status_code == 200:
                base = HEALTH_URL.rsplit("/", 1)[0]
                print("Qdrant is up! REST:", base)
                print("gRPC: localhost:", base.replace("6333", "6334"))
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    print("Timed out waiting for Qdrant. See logs:\n  docker logs -f", CONTAINER_NAME)
    sys.exit(2)


if __name__ == "__main__":
    main()
