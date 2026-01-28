format_files() {
    local dir="$1"
    [ -z "$dir" ] && dir="."

    find "$dir" \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) \
        -print0 | xargs -0 clang-format -i
}

format_files ./src
format_files ./test
