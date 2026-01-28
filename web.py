from flask import Flask, Response
import os, mmap, struct, time, fcntl
from PIL import Image
import io

app = Flask(__name__)

# ===============================
# SHM config
# ===============================
SHM_PATH = "/dev/shm/debug_frame"
SHM_SIZE = 8 * 1024 * 1024

SHM_HEADER_FMT = "6I"
SHM_HEADER_SIZE = struct.calcsize(SHM_HEADER_FMT)

MAGIC = 0x494D4756  # 'IMGV'
PIXEL_GRAY = 1
PIXEL_RGB  = 2

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
        if not mapfile:
            init_shm()
            time.sleep(0.5)
            continue

        try:
            mapfile.seek(0)
            header = mapfile.read(SHM_HEADER_SIZE)
            if len(header) != SHM_HEADER_SIZE:
                time.sleep(0.01)
                continue

            magic, w, h, stride, pf, data_size = struct.unpack(
                SHM_HEADER_FMT, header
            )

            if magic != MAGIC or w == 0 or h == 0 or data_size <= 0:
                time.sleep(0.01)
                continue

            raw = mapfile.read(data_size)
            if len(raw) != data_size:
                continue

            # ===============================
            # RAW → tightly packed buffer
            # ===============================
            if pf == PIXEL_GRAY:
                channels = 1
                mode = "L"
            elif pf == PIXEL_RGB:
                channels = 3
                mode = "RGB"
            else:
                continue

            row_bytes = w * channels
            tight = bytearray(row_bytes * h)

            for y in range(h):
                src_off = y * stride
                dst_off = y * row_bytes
                tight[dst_off:dst_off + row_bytes] = raw[
                    src_off:src_off + row_bytes
                ]

            img = Image.frombytes(mode, (w, h), bytes(tight))

            # ===============================
            # JPEG encode
            # ===============================
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70, optimize=True)
            jpg = buf.getvalue()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpg +
                b"\r\n"
            )

            time.sleep(0.03)

        except Exception as e:
            print("stream error:", e)
            time.sleep(0.1)


@app.route("/video")
def video_feed():
    return Response(
        mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    print("启动 Raw SHM → MJPEG 服务: http://0.0.0.0:5000/video")
    init_shm()
    app.run(host="0.0.0.0", port=5000, threaded=True)
