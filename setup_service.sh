#!/usr/bin/env bash
# 自动管理 systemd service 文件
# 工作目录 = 脚本所在路径

SERVICE_NAME="dart_vision"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
sudo usermod -aG video pico

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR"
LIB_DIR="$(dirname "$WORK_DIR")/arm_lib"

ACTION=$1

echo "📂  脚本路径: $SCRIPT_DIR"
echo "🏗  工作区路径: $WORK_DIR"
echo "📦  库路径: $LIB_DIR"
echo "🧾  目标 Service 文件: $SERVICE_FILE"
echo

uninstall_service() {

    if [[ ! -f "$SERVICE_FILE" ]]; then
        echo "⚠️  Service 文件不存在，无需卸载。"
        exit 0
    fi

    echo "🛑 停止并卸载 Service..."

    sudo systemctl stop "${SERVICE_NAME}.service" 2>/dev/null || true
    sudo systemctl disable "${SERVICE_NAME}.service" 2>/dev/null || true
    sudo rm -f "$SERVICE_FILE"

    sudo systemctl daemon-reload

    echo "✅ Service 已成功卸载。"
    exit 0
}

install_service() {

    if [[ -f "$SERVICE_FILE" ]]; then
        echo "⚠️  检测到已存在的 Service 文件：$SERVICE_FILE"
        read -p "是否覆盖？(y/N): " confirm

        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            echo "🚫 已取消安装。"
            exit 0
        fi

        echo "🧹 删除旧版本 Service..."
        sudo systemctl stop "${SERVICE_NAME}.service" 2>/dev/null || true
        sudo systemctl disable "${SERVICE_NAME}.service" 2>/dev/null || true
        sudo rm -f "$SERVICE_FILE"
    fi

    echo "✏️  正在生成新的 Service 文件..."

    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Dart Vision Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$WORK_DIR
Environment=LD_LIBRARY_PATH=$LIB_DIR
ExecStart=$WORK_DIR/vision $WORK_DIR/config.toml false
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    sudo chmod 644 "$SERVICE_FILE"

    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}.service"
    sudo systemctl restart "${SERVICE_NAME}.service"

    echo
    echo "✅ Service 已生成并启动成功！"
    echo "🔍 查看状态：sudo systemctl status ${SERVICE_NAME}.service"
    echo "📜 实时日志：journalctl -u ${SERVICE_NAME}.service -f"
}

if [[ -z "$ACTION" ]]; then
    echo "❌ 请输入操作：$0 [install|uninstall|start|stop|restart|status|journal]"
    exit 1
fi

case "$ACTION" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    start)
        sudo systemctl start "${SERVICE_NAME}.service"
        echo "✅ Service 已启动"
        ;;
    stop)
        sudo systemctl stop "${SERVICE_NAME}.service"
        echo "🛑 Service 已停止"
        ;;
    restart)
        sudo systemctl restart "${SERVICE_NAME}.service"
        echo "🔄 Service 已重启"
        ;;
    status)
        sudo systemctl status "${SERVICE_NAME}.service"
        ;;
    journal)
        sudo journalctl -u "${SERVICE_NAME}.service" -f
        ;;
    *)
        echo "❌ 参数错误: $0 [install|uninstall|start|stop|restart|status|journal]"
        exit 1
        ;;
esac
