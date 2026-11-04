// Minimal JSON reader and writer helpers shared by banach-trace and banach-engine.
// Enough for line-delimited protocols and corpus files: objects, arrays, strings, numbers,
// true/false/null. Parsing is bounded (nesting depth 64) and throws on malformed input.
#pragma once

#include <cctype>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mini_json {

struct Value {
    enum Kind { NUL, BOOL, NUM, STR, ARR, OBJ } kind = NUL;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    const Value * get(const std::string & k) const {
        for (auto & kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

struct Parser {
    const std::string & s;
    size_t i = 0;
    explicit Parser(const std::string & src) : s(src) {}

    [[noreturn]] void fail(const char * what) {
        throw std::runtime_error(std::string("json: ") + what + " at offset " + std::to_string(i));
    }
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++; }
    bool lit(const char * w) {
        size_t n = strlen(w);
        if (s.compare(i, n, w) == 0) { i += n; return true; }
        return false;
    }
    static void put_utf8(std::string & o, uint32_t c) {
        if (c < 0x80) o += char(c);
        else if (c < 0x800) { o += char(0xC0 | (c >> 6)); o += char(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { o += char(0xE0 | (c >> 12)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
        else { o += char(0xF0 | (c >> 18)); o += char(0x80 | ((c >> 12) & 0x3F)); o += char(0x80 | ((c >> 6) & 0x3F)); o += char(0x80 | (c & 0x3F)); }
    }
    uint32_t hex4() {
        if (i + 4 > s.size()) fail("short \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) {
            char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad hex digit");
        }
        return v;
    }
    std::string string() {
        if (s[i] != '"') fail("expected string");
        i++;
        std::string o;
        while (true) {
            if (i >= s.size()) fail("unterminated string");
            char c = s[i++];
            if (c == '"') break;
            if (c != '\\') { o += c; continue; }
            if (i >= s.size()) fail("bad escape");
            char e = s[i++];
            switch (e) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    uint32_t c1 = hex4();
                    if (c1 >= 0xD800 && c1 < 0xDC00 && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        i += 2;
                        uint32_t c2 = hex4();
                        c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00);
                    }
                    put_utf8(o, c1);
                } break;
                default: fail("bad escape");
            }
        }
        return o;
    }
    Value value(int depth = 0) {
        if (depth > 64) fail("nesting too deep");
        ws();
        if (i >= s.size()) fail("unexpected end");
        Value v;
        char c = s[i];
        if (c == '{') {
            v.kind = Value::OBJ; i++; ws();
            if (i < s.size() && s[i] == '}') { i++; return v; }
            while (true) {
                ws(); std::string k = string(); ws();
                if (i >= s.size() || s[i] != ':') fail("expected ':'");
                i++;
                v.obj.emplace_back(k, value(depth + 1)); ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == '}') { i++; break; }
                fail("expected ',' or '}'");
            }
        } else if (c == '[') {
            v.kind = Value::ARR; i++; ws();
            if (i < s.size() && s[i] == ']') { i++; return v; }
            while (true) {
                v.arr.push_back(value(depth + 1)); ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == ']') { i++; break; }
                fail("expected ',' or ']'");
            }
        } else if (c == '"') {
            v.kind = Value::STR; v.str = string();
        } else if (lit("true")) { v.kind = Value::BOOL; v.b = true; }
        else if (lit("false")) { v.kind = Value::BOOL; v.b = false; }
        else if (lit("null")) { v.kind = Value::NUL; }
        else {
            size_t j = i;
            while (j < s.size() && (isdigit((unsigned char) s[j]) || strchr("+-.eE", s[j]))) j++;
            if (j == i) fail("unexpected character");
            v.kind = Value::NUM; v.num = std::stod(s.substr(i, j - i)); i = j;
        }
        return v;
    }
};

Value parse(const std::string & s) {
    Parser p(s);
    Value v = p.value();
    p.ws();
    if (p.i != s.size()) p.fail("trailing characters");
    return v;
}

std::string escape(const std::string & s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += char(c);
        }
    }
    return o + "\"";
}

} // namespace mini_json

namespace mini_json {

// Length of the longest prefix of s that ends on a complete UTF-8 sequence.
inline size_t utf8_complete_prefix(const std::string & s) {
    size_t n = s.size();
    size_t i = n;
    // walk back over at most 3 continuation bytes to the lead byte of the last sequence
    size_t back = 0;
    while (i > 0 && back < 4) {
        unsigned char c = (unsigned char) s[i - 1];
        if ((c & 0xC0) != 0x80) {
            size_t need = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
            return (n - (i - 1) >= need) ? n : i - 1;
        }
        i--;
        back++;
    }
    return n;
}

} // namespace mini_json
