//! Tool use (function calling) for the OpenAI and Anthropic APIs.
//!
//! The engine speaks OpenAI-format messages and tools: it renders them with the model's own
//! Jinja chat template and, for template families llama.cpp knows, constrains and parses tool
//! calls itself. This module converts Anthropic requests to that format and holds the
//! fallback extractor used when a template has no native tool-call syntax.

use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON text of the arguments object
    pub arguments: String,
}

pub fn text_of(content: &Value) -> Result<String, String> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p["type"].as_str() {
                    Some("text") | Some("input_text") | Some("output_text") => out.push_str(p["text"].as_str().unwrap_or("")),
                    Some(t) => return Err(format!("content parts of type {t:?} are not supported")),
                    None => return Err("content part without a type".into()),
                }
            }
            Ok(out)
        }
        _ => Err("content must be a string or an array of text parts".into()),
    }
}

fn args_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "{}".into(),
        other => other.to_string(),
    }
}

/// OpenAI chat messages, normalised for the engine (text content, tool calls, tool results).
pub fn openai_messages(msgs: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = vec![];
    for m in msgs {
        let role = m["role"].as_str().ok_or("message without a role")?;
        match role {
            "system" | "developer" => out.push(json!({"role": "system", "content": text_of(&m["content"])?})),
            "user" => out.push(json!({"role": "user", "content": text_of(&m["content"])?})),
            "assistant" => {
                let mut msg = json!({"role": "assistant", "content": text_of(&m["content"])?});
                if let Some(calls) = m["tool_calls"].as_array() {
                    let calls: Vec<Value> = calls
                        .iter()
                        .map(|c| {
                            json!({"id": c["id"], "type": "function", "function": {
                                "name": c["function"]["name"], "arguments": args_text(&c["function"]["arguments"])}})
                        })
                        .collect();
                    if !calls.is_empty() {
                        msg["tool_calls"] = Value::Array(calls);
                    }
                }
                out.push(msg);
            }
            "tool" | "function" => {
                let mut msg = json!({"role": "tool", "content": text_of(&m["content"])?});
                if let Some(id) = m["tool_call_id"].as_str() {
                    msg["tool_call_id"] = json!(id);
                }
                if let Some(n) = m["name"].as_str() {
                    msg["name"] = json!(n);
                }
                out.push(msg);
            }
            other => return Err(format!("messages with role {other:?} are not supported")),
        }
    }
    Ok(out)
}

/// OpenAI tools (and the legacy `functions` field), validated.
pub fn openai_tools(body: &Value) -> Result<Vec<Value>, String> {
    let mut out = vec![];
    if let Some(tools) = body["tools"].as_array() {
        for t in tools {
            if t["type"].as_str().unwrap_or("function") != "function" {
                return Err(format!("tools of type {:?} are not supported", t["type"]));
            }
            let f = &t["function"];
            if f["name"].as_str().map(|s| s.is_empty()).unwrap_or(true) {
                return Err("every tool needs a function name".into());
            }
            out.push(json!({"type": "function", "function": {
                "name": f["name"], "description": f["description"].as_str().unwrap_or(""),
                "parameters": if f["parameters"].is_object() { f["parameters"].clone() } else { json!({"type": "object", "properties": {}}) }}}));
        }
    }
    if let Some(fns) = body["functions"].as_array() {
        for f in fns {
            out.push(json!({"type": "function", "function": f}));
        }
    }
    Ok(out)
}

/// OpenAI tool_choice: "auto" | "none" | "required" | {"type":"function","function":{"name":..}}.
/// A named choice becomes "required" with the tool list narrowed to that tool.
pub fn openai_tool_choice(v: &Value, tools: &mut Vec<Value>) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(_) => {
            let name = v["function"]["name"].as_str().or(v["name"].as_str())?;
            tools.retain(|t| t["function"]["name"].as_str() == Some(name));
            Some("required".into())
        }
        _ => None,
    }
}

/// Anthropic messages request -> (OpenAI-format messages, tools, tool_choice).
/// Messages, tools and tool_choice of an Anthropic request, in OpenAI form.
pub type NormalisedRequest = (Vec<Value>, Vec<Value>, Option<String>);

pub fn anthropic_request(body: &Value) -> Result<NormalisedRequest, String> {
    let mut msgs = vec![];
    match &body["system"] {
        Value::Null => {}
        s => msgs.push(json!({"role": "system", "content": text_of(s)?})),
    }
    for m in body["messages"].as_array().ok_or("messages is required")? {
        let role = m["role"].as_str().ok_or("message without a role")?;
        let blocks: Vec<Value> = match &m["content"] {
            Value::String(s) => vec![json!({"type": "text", "text": s})],
            Value::Array(a) => a.clone(),
            Value::Null => vec![],
            _ => return Err("content must be a string or an array of blocks".into()),
        };
        match role {
            "user" => {
                let mut text = String::new();
                for b in &blocks {
                    match b["type"].as_str() {
                        Some("text") => text.push_str(b["text"].as_str().unwrap_or("")),
                        Some("tool_result") => {
                            // tool results become tool messages, in order
                            if !text.is_empty() {
                                msgs.push(json!({"role": "user", "content": std::mem::take(&mut text)}));
                            }
                            let mut result = text_of(&b["content"]).unwrap_or_default();
                            if b["is_error"].as_bool().unwrap_or(false) {
                                result = format!("Error: {result}");
                            }
                            msgs.push(json!({"role": "tool", "tool_call_id": b["tool_use_id"], "content": result}));
                        }
                        Some(t) => return Err(format!("content blocks of type {t:?} are not supported")),
                        None => return Err("content block without a type".into()),
                    }
                }
                if !text.is_empty() {
                    msgs.push(json!({"role": "user", "content": text}));
                }
            }
            "assistant" => {
                let mut text = String::new();
                let mut calls = vec![];
                for b in &blocks {
                    match b["type"].as_str() {
                        Some("text") => text.push_str(b["text"].as_str().unwrap_or("")),
                        Some("tool_use") => calls.push(json!({"id": b["id"], "type": "function", "function": {
                            "name": b["name"], "arguments": args_text(&b["input"])}})),
                        Some("thinking") | Some("redacted_thinking") => {}
                        Some(t) => return Err(format!("content blocks of type {t:?} are not supported")),
                        None => return Err("content block without a type".into()),
                    }
                }
                let mut msg = json!({"role": "assistant", "content": text});
                if !calls.is_empty() {
                    msg["tool_calls"] = Value::Array(calls);
                }
                msgs.push(msg);
            }
            other => return Err(format!("messages with role {other:?} are not supported")),
        }
    }
    let mut tools = vec![];
    if let Some(ts) = body["tools"].as_array() {
        for t in ts {
            let name = t["name"].as_str().ok_or("every tool needs a name")?;
            tools.push(json!({"type": "function", "function": {
                "name": name, "description": t["description"].as_str().unwrap_or(""),
                "parameters": if t["input_schema"].is_object() { t["input_schema"].clone() } else { json!({"type": "object", "properties": {}}) }}}));
        }
    }
    let choice = match body["tool_choice"]["type"].as_str() {
        Some("auto") => Some("auto".to_string()),
        Some("any") => Some("required".to_string()),
        Some("none") => Some("none".to_string()),
        Some("tool") => {
            let name = body["tool_choice"]["name"].as_str().unwrap_or("");
            tools.retain(|t| t["function"]["name"].as_str() == Some(name));
            Some("required".to_string())
        }
        _ => None,
    };
    Ok((msgs, tools, choice))
}

/// Fallback for templates without native tool-call syntax: find JSON tool calls in the text.
///
/// Accepts `{"name": .., "arguments"|"parameters": {..}}`, the same wrapped in
/// `{"function": {..}}`, and arrays of those, anywhere in the text (inside `<tool_call>` tags,
/// `<|tool_call|>`, code fences, or bare). Only names of offered tools count. Returns the text
/// with the calls and their wrappers removed, and the calls.
pub fn extract_tool_calls(text: &str, tool_names: &[String]) -> (String, Vec<ToolCall>) {
    let mut calls = vec![];
    let mut spans: Vec<(usize, usize)> = vec![];
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' || bytes[i] == b'[' {
            let mut stream = serde_json::Deserializer::from_str(&text[i..]).into_iter::<Value>();
            if let Some(Ok(v)) = stream.next() {
                let end = i + stream.byte_offset();
                let mut found = vec![];
                collect_calls(&v, tool_names, &mut found);
                if !found.is_empty() {
                    calls.extend(found);
                    spans.push((i, end));
                    i = end;
                    continue;
                }
            }
        }
        i += text[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
    }
    if calls.is_empty() {
        return (text.to_string(), calls);
    }
    let mut rest = String::new();
    let mut last = 0;
    for (s, e) in &spans {
        rest.push_str(&text[last..*s]);
        last = *e;
    }
    rest.push_str(&text[last..]);
    for tag in ["<tool_call>", "</tool_call>", "<|tool_call|>", "<tools>", "</tools>", "```json", "```"] {
        rest = rest.replace(tag, "");
    }
    for (k, c) in calls.iter_mut().enumerate() {
        if c.id.is_empty() {
            c.id = format!("call_{k}");
        }
    }
    (rest.trim().to_string(), calls)
}

fn collect_calls(v: &Value, names: &[String], out: &mut Vec<ToolCall>) {
    match v {
        Value::Array(items) => items.iter().for_each(|x| collect_calls(x, names, out)),
        Value::Object(_) => {
            let f = if v["function"].is_object() { &v["function"] } else { v };
            // a tool definition echoed back (description, JSON-schema parameters) is not a call
            let params = &f["parameters"];
            let looks_like_schema = f["arguments"].is_null()
                && (f.get("description").is_some() || (params["type"] == "object" && params.get("properties").is_some()));
            if looks_like_schema {
                return;
            }
            if let Some(name) = f["name"].as_str() {
                if names.iter().any(|n| n == name) {
                    let args = if !f["arguments"].is_null() { &f["arguments"] } else { &f["parameters"] };
                    out.push(ToolCall { id: v["id"].as_str().unwrap_or("").to_string(), name: name.to_string(), arguments: args_text(args) });
                }
            }
        }
        _ => {}
    }
}

pub fn tool_names(tools: &[Value]) -> Vec<String> {
    tools.iter().filter_map(|t| t["function"]["name"].as_str().map(|s| s.to_string())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["get_weather".into(), "read_file".into()]
    }

    #[test]
    fn extracts_granite_style_and_keeps_text() {
        let t = "Let me check.\n<tool_call>[{\"arguments\": {\"city\": \"Paris\"}, \"name\": \"get_weather\"}]";
        let (rest, calls) = extract_tool_calls(t, &names());
        assert_eq!(rest, "Let me check.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(serde_json::from_str::<Value>(&calls[0].arguments).unwrap(), json!({"city": "Paris"}));
    }

    #[test]
    fn extracts_hermes_fenced_and_wrapped_forms() {
        let t = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.py\"}}\n</tool_call>";
        assert_eq!(extract_tool_calls(t, &names()).1[0].name, "read_file");
        let t = "```json\n{\"function\": {\"name\": \"get_weather\", \"parameters\": {\"city\": \"Oslo\"}}}\n```";
        let (rest, calls) = extract_tool_calls(t, &names());
        assert_eq!(rest, "");
        assert_eq!(calls[0].arguments, "{\"city\":\"Oslo\"}");
        let t = "[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"A\"}},{\"name\":\"read_file\",\"arguments\":{\"path\":\"b\"}}]";
        assert_eq!(extract_tool_calls(t, &names()).1.len(), 2);
    }

    #[test]
    fn ignores_echoed_tool_definitions() {
        let t = "{\"name\": \"get_weather\", \"description\": \"w\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}}}";
        assert!(extract_tool_calls(t, &names()).1.is_empty());
        let t = "{\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}";
        assert!(extract_tool_calls(t, &names()).1.is_empty());
        // but parameters holding real values is a call
        let t = "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Oslo\"}}";
        assert_eq!(extract_tool_calls(t, &names()).1.len(), 1);
    }

    #[test]
    fn ignores_unknown_names_and_plain_json() {
        let t = "Here is JSON: {\"name\": \"rm_rf\", \"arguments\": {}} and {\"a\": 1}";
        let (rest, calls) = extract_tool_calls(t, &names());
        assert!(calls.is_empty());
        assert_eq!(rest, t);
    }

    #[test]
    fn anthropic_conversation_round_trip() {
        let body = json!({
            "system": "be helpful",
            "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
            "tool_choice": {"type": "any"},
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "content": [{"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18 C, sunny"}]}
            ]});
        let (msgs, tools, choice) = anthropic_request(&body).unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["arguments"], "{\"city\":\"Paris\"}");
        assert_eq!(msgs[3], json!({"role": "tool", "tool_call_id": "toolu_1", "content": "18 C, sunny"}));
        assert_eq!(tools[0]["function"]["parameters"]["properties"]["city"]["type"], "string");
        assert_eq!(choice.as_deref(), Some("required"));
    }

    #[test]
    fn openai_tool_messages_and_named_choice() {
        let msgs = openai_messages(&[
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
                   "function": {"name": "read_file", "arguments": {"path": "x"}}}]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "data"}]}),
        ])
        .unwrap();
        assert_eq!(msgs[0]["tool_calls"][0]["function"]["arguments"], "{\"path\":\"x\"}");
        assert_eq!(msgs[1], json!({"role": "tool", "content": "data", "tool_call_id": "c1"}));
        let body = json!({"tools": [
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "read_file"}}]});
        let mut tools = openai_tools(&body).unwrap();
        let c = openai_tool_choice(&json!({"type": "function", "function": {"name": "read_file"}}), &mut tools);
        assert_eq!(c.as_deref(), Some("required"));
        assert_eq!(tool_names(&tools), vec!["read_file".to_string()]);
    }
}
