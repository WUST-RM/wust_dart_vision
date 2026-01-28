find ./src -name '*.cpp' -o -name '*.hpp'  -o -name '*.h'| xargs clang-format -i
find ./test -name '*.cpp' -o -name '*.hpp'  -o -name '*.h'| xargs clang-format -i