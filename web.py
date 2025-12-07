from flask import Flask, Response
import os, mmap, struct, time, fcntl

app = Flask(__name__)

# ===============================
# 共享内存参数
# ===============================
SHM_PATH = "/dev/shm/debug_frame"
SHM_SIZE = 2 * 1024 * 1024  # 2MB

mapfile = None
fd = None


def init_shm():
    global mapfile, fd
    try:
        fd = os.open(SHM_PATH, os.O_RDONLY)
        mapfile = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        print("共享内存打开成功")
        return True
    except Exception as e:
        print(f"共享内存初始化失败: {e}")
        return False


def mjpeg_stream():
    global mapfile
    while True:
        if mapfile:
            try:
                mapfile.seek(0)
                size_bytes = mapfile.read(4)
                if len(size_bytes) < 4:
                    continue

                jpg_size = struct.unpack("I", size_bytes)[0]
                if jpg_size <= 0 or jpg_size > SHM_SIZE - 4:
                    continue

                jpg_bytes = mapfile.read(jpg_size)
                if len(jpg_bytes) != jpg_size:
                    continue

                if jpg_bytes[0:3] != b"\xff\xd8\xff":
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    jpg_bytes +
                    b"\r\n"
                )
                time.sleep(0.03)

            except Exception:
                time.sleep(0.1)
                continue

        else:
            init_shm()
            time.sleep(0.5)


@app.route("/video")
def video_feed():
    return Response(mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("启动 MJPEG 视频流服务: http://0.0.0.0:5000/video")
    init_shm()
    app.run(host="0.0.0.0", port=5000, threaded=True)
