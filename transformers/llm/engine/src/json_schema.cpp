#include "json_schema.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdio>

namespace MNN {
namespace Transformer {

// JsonValue implementation
bool JsonValue::has(const std::string& key) const {
    if (type != OBJECT) return false;
    for (const auto& kv : object_values) {
        if (kv.first == key) return true;
    }
    return false;
}

const JsonValue& JsonValue::get(const std::string& key) const {
    static JsonValue none_value;
    if (type != OBJECT) return none_value;
    for (const auto& kv : object_values) {
        if (kv.first == key) return kv.second;
    }
    return none_value;
}

std::string JsonValue::as_string() const {
    return type == STRING ? string_value : "";
}

double JsonValue::as_number() const {
    return type == NUMBER ? number_value : 0;
}

bool JsonValue::as_bool() const {
    return type == BOOL ? bool_value : false;
}

const std::vector<JsonValue>& JsonValue::as_array() const {
    static std::vector<JsonValue> empty_array;
    return type == ARRAY ? array_values : empty_array;
}

// Schema type conversion
SchemaType string_to_schema_type(const std::string& type_str) {
    if (type_str == "string") return SchemaType::STRING;
    if (type_str == "number") return SchemaType::NUMBER;
    if (type_str == "integer") return SchemaType::INTEGER;
    if (type_str == "boolean") return SchemaType::BOOLEAN;
    if (type_str == "object") return SchemaType::OBJECT;
    if (type_str == "array") return SchemaType::ARRAY;
    if (type_str == "null") return SchemaType::NULL_TYPE;
    return SchemaType::UNKNOWN;
}

std::string schema_type_to_string(SchemaType type) {
    switch (type) {
        case SchemaType::STRING: return "string";
        case SchemaType::NUMBER: return "number";
        case SchemaType::INTEGER: return "integer";
        case SchemaType::BOOLEAN: return "boolean";
        case SchemaType::OBJECT: return "object";
        case SchemaType::ARRAY: return "array";
        case SchemaType::NULL_TYPE: return "null";
        default: return "unknown";
    }
}

// JSON Parser implementation
void JsonSchemaNode::skip_whitespace(const std::string& json, size_t& pos) {
    while (pos < json.size() && std::isspace(json[pos])) {
        pos++;
    }
}

std::string JsonSchemaNode::parse_string(const std::string& json, size_t& pos) {
    if (pos >= json.size() || json[pos] != '"') {
        return "";
    }
    pos++; // skip opening quote
    
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                case '/': result += '/'; break;
                case 'b': result += '\b'; break;
                case 'f': result += '\f'; break;
                case 'n': result += '\n'; break;
                case 'r': result += '\r'; break;
                case 't': result += '\t'; break;
                case 'u': {
                    // Unicode escape - simplified handling
                    if (pos + 4 < json.size()) {
                        pos += 4;
                    }
                    break;
                }
                default: result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        pos++;
    }
    
    if (pos < json.size()) {
        pos++; // skip closing quote
    }
    return result;
}

double JsonSchemaNode::parse_number(const std::string& json, size_t& pos) {
    size_t start = pos;
    if (json[pos] == '-') pos++;
    
    while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '.' || 
           json[pos] == 'e' || json[pos] == 'E' || json[pos] == '+' || json[pos] == '-')) {
        if ((json[pos] == '+' || json[pos] == '-') && pos > start && 
            json[pos-1] != 'e' && json[pos-1] != 'E') {
            break;
        }
        pos++;
    }
    
    // Simple parsing without exceptions
    double result = 0.0;
    bool negative = false;
    bool has_decimal = false;
    double decimal_place = 1.0;
    size_t i = start;
    
    if (json[i] == '-') {
        negative = true;
        i++;
    }
    
    while (i < pos && json[i] != 'e' && json[i] != 'E') {
        if (json[i] == '.') {
            has_decimal = true;
        } else if (std::isdigit(json[i])) {
            if (has_decimal) {
                decimal_place *= 0.1;
                result += (json[i] - '0') * decimal_place;
            } else {
                result = result * 10 + (json[i] - '0');
            }
        }
        i++;
    }
    
    // Handle exponent (simplified)
    if (i < pos && (json[i] == 'e' || json[i] == 'E')) {
        i++;
        int exp = 0;
        bool exp_neg = false;
        if (i < pos && json[i] == '-') {
            exp_neg = true;
            i++;
        }
        while (i < pos && std::isdigit(json[i])) {
            exp = exp * 10 + (json[i] - '0');
            i++;
        }
        if (exp_neg) exp = -exp;
        result *= std::pow(10.0, exp);
    }
    
    return negative ? -result : result;
}

JsonValue JsonSchemaNode::parse_value(const std::string& json, size_t& pos) {
    skip_whitespace(json, pos);
    JsonValue result;
    
    if (pos >= json.size()) {
        return result;
    }
    
    char c = json[pos];
    
    if (c == '"') {
        result.type = JsonValue::STRING;
        result.string_value = parse_string(json, pos);
    }
    else if (c == '{') {
        result.type = JsonValue::OBJECT;
        pos++; // skip '{'
        skip_whitespace(json, pos);
        
        while (pos < json.size() && json[pos] != '}') {
            skip_whitespace(json, pos);
            std::string key = parse_string(json, pos);
            skip_whitespace(json, pos);
            
            if (pos < json.size() && json[pos] == ':') {
                pos++; // skip ':'
                JsonValue value = parse_value(json, pos);
                result.object_values.push_back({key, value});
            }
            
            skip_whitespace(json, pos);
            if (pos < json.size() && json[pos] == ',') {
                pos++;
            }
        }
        
        if (pos < json.size()) {
            pos++; // skip '}'
        }
    }
    else if (c == '[') {
        result.type = JsonValue::ARRAY;
        pos++; // skip '['
        skip_whitespace(json, pos);
        
        while (pos < json.size() && json[pos] != ']') {
            JsonValue value = parse_value(json, pos);
            result.array_values.push_back(value);
            
            skip_whitespace(json, pos);
            if (pos < json.size() && json[pos] == ',') {
                pos++;
            }
        }
        
        if (pos < json.size()) {
            pos++; // skip ']'
        }
    }
    else if (c == 't' && pos + 3 < json.size() && json.substr(pos, 4) == "true") {
        result.type = JsonValue::BOOL;
        result.bool_value = true;
        pos += 4;
    }
    else if (c == 'f' && pos + 4 < json.size() && json.substr(pos, 5) == "false") {
        result.type = JsonValue::BOOL;
        result.bool_value = false;
        pos += 5;
    }
    else if (c == 'n' && pos + 3 < json.size() && json.substr(pos, 4) == "null") {
        result.type = JsonValue::NONE;
        pos += 4;
    }
    else if (c == '-' || std::isdigit(c)) {
        result.type = JsonValue::NUMBER;
        result.number_value = parse_number(json, pos);
    }
    
    return result;
}

JsonValue JsonSchemaNode::parse_json(const std::string& json) {
    size_t pos = 0;
    return parse_value(json, pos);
}

bool JsonSchemaNode::parse(const std::string& schema_json) {
    if (schema_json.empty()) {
        return false;
    }
    
    JsonValue root = parse_json(schema_json);
    
    if (root.type != JsonValue::OBJECT) {
        return false;
    }
    
    // Parse properties
    if (root.has("properties")) {
        const JsonValue& props = root.get("properties");
        if (props.type == JsonValue::OBJECT) {
            for (const auto& kv : props.object_values) {
                SchemaProperty prop;
                prop.name = kv.first;
                
                const JsonValue& prop_schema = kv.second;
                
                // Parse type
                if (prop_schema.has("type")) {
                    std::string type_str = prop_schema.get("type").as_string();
                    prop.type = string_to_schema_type(type_str);
                }
                
                // Parse string constraints
                if (prop_schema.has("minLength")) {
                    prop.min_length = (int)prop_schema.get("minLength").as_number();
                }
                if (prop_schema.has("maxLength")) {
                    prop.max_length = (int)prop_schema.get("maxLength").as_number();
                }
                if (prop_schema.has("pattern")) {
                    prop.pattern = prop_schema.get("pattern").as_string();
                }
                
                // Parse number constraints
                if (prop_schema.has("minimum")) {
                    prop.min_value = prop_schema.get("minimum").as_number();
                }
                if (prop_schema.has("maximum")) {
                    prop.max_value = prop_schema.get("maximum").as_number();
                }
                
                // Parse enum
                if (prop_schema.has("enum")) {
                    const JsonValue& enum_val = prop_schema.get("enum");
                    if (enum_val.type == JsonValue::ARRAY) {
                        prop.enum_values = enum_val.array_values;
                    }
                }
                
                // Parse nested object schema
                if (prop.type == SchemaType::OBJECT) {
                    prop.object_schema = std::make_shared<JsonSchemaNode>();
                    // Convert property schema back to string for recursive parsing
                    // For simplicity, we'll just mark it as object without nested schema
                }
                
                // Parse array items schema
                if (prop.type == SchemaType::ARRAY) {
                    if (prop_schema.has("items")) {
                        prop.items_schema = std::make_shared<JsonSchemaNode>();
                    }
                    if (prop_schema.has("minItems")) {
                        prop.min_items = (int)prop_schema.get("minItems").as_number();
                    }
                    if (prop_schema.has("maxItems")) {
                        prop.max_items = (int)prop_schema.get("maxItems").as_number();
                    }
                }
                
                properties_.push_back(prop);
            }
        }
    }
    
    // Parse required array
    if (root.has("required")) {
        const JsonValue& req = root.get("required");
        if (req.type == JsonValue::ARRAY) {
            for (const auto& item : req.array_values) {
                std::string req_name = item.as_string();
                if (!req_name.empty()) {
                    required_.push_back(req_name);
                }
            }
        }
    }
    
    // Mark required properties
    for (auto& prop : properties_) {
        for (const auto& req : required_) {
            if (prop.name == req) {
                prop.required = true;
                break;
            }
        }
    }
    
    // Parse additionalProperties
    if (root.has("additionalProperties")) {
        additional_properties_ = root.get("additionalProperties").as_bool();
    }
    
    return true;
}

std::vector<std::string> JsonSchemaNode::required_properties() const {
    return required_;
}

const SchemaProperty* JsonSchemaNode::get_property(const std::string& name) const {
    for (const auto& prop : properties_) {
        if (prop.name == name) {
            return &prop;
        }
    }
    return nullptr;
}

bool JsonSchemaParser::initialize(const std::string& schema_json) {
    root_ = std::make_shared<JsonSchemaNode>();
    if (!root_->parse(schema_json)) {
        fprintf(stderr, "Failed to parse JSON schema\n");
        root_ = nullptr;
        return false;
    }
    return true;
}

} // namespace Transformer
} // namespace MNN
