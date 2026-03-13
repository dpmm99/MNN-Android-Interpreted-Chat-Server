#ifndef JSON_SCHEMA_HPP
#define JSON_SCHEMA_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace MNN {
namespace Transformer {

// JSON Schema property types
enum class SchemaType {
    UNKNOWN = 0,
    STRING = 1,
    NUMBER = 2,
    INTEGER = 3,
    BOOLEAN = 4,
    OBJECT = 5,
    ARRAY = 6,
    NULL_TYPE = 7
};

// Forward declaration
class JsonSchemaNode;

// Simple JSON value representation (no external dependencies)
struct JsonValue {
    enum Type { NONE, STRING, NUMBER, BOOL, OBJECT, ARRAY };
    Type type = NONE;
    std::string string_value;
    double number_value = 0;
    bool bool_value = false;
    std::vector<std::pair<std::string, JsonValue>> object_values;
    std::vector<JsonValue> array_values;
    
    bool has(const std::string& key) const;
    const JsonValue& get(const std::string& key) const;
    std::string as_string() const;
    double as_number() const;
    bool as_bool() const;
    const std::vector<JsonValue>& as_array() const;
};

// Schema property definition
struct SchemaProperty {
    std::string name;
    SchemaType type = SchemaType::UNKNOWN;
    bool required = false;
    std::string description;
    
    // For strings
    int min_length = 0;
    int max_length = -1;  // -1 means no limit
    std::string pattern;  // regex pattern
    
    // For numbers
    double min_value = -1e308;
    double max_value = 1e308;
    
    // For objects
    std::shared_ptr<JsonSchemaNode> object_schema;
    
    // For arrays
    std::shared_ptr<JsonSchemaNode> items_schema;
    int min_items = 0;
    int max_items = -1;
    
    // Enum values
    std::vector<JsonValue> enum_values;
};

// JSON Schema node (represents an object or array schema)
class JsonSchemaNode {
public:
    JsonSchemaNode() = default;
    
    // Parse schema from JSON string
    bool parse(const std::string& schema_json);
    
    // Get expected properties
    const std::vector<SchemaProperty>& properties() const { return properties_; }
    
    // Check if additional properties are allowed
    bool additional_properties() const { return additional_properties_; }
    
    // Get required property names
    std::vector<std::string> required_properties() const;
    
    // Get property by name
    const SchemaProperty* get_property(const std::string& name) const;
    
private:
    std::vector<SchemaProperty> properties_;
    bool additional_properties_ = true;
    std::vector<std::string> required_;
    
    // JSON parsing helpers
    JsonValue parse_json(const std::string& json);
    JsonValue parse_value(const std::string& json, size_t& pos);
    std::string parse_string(const std::string& json, size_t& pos);
    double parse_number(const std::string& json, size_t& pos);
    void skip_whitespace(const std::string& json, size_t& pos);
};

// JSON Schema parser and validator
class JsonSchemaParser {
public:
    JsonSchemaParser() = default;
    
    // Initialize from schema JSON string
    bool initialize(const std::string& schema_json);
    
    // Get root schema node
    const JsonSchemaNode* root() const { return root_.get(); }
    
    // Check if schema is loaded
    bool is_loaded() const { return root_ != nullptr; }
    
private:
    std::shared_ptr<JsonSchemaNode> root_;
};

// Convert string to SchemaType
SchemaType string_to_schema_type(const std::string& type_str);

// Convert SchemaType to string
std::string schema_type_to_string(SchemaType type);

} // namespace Transformer
} // namespace MNN

#endif // JSON_SCHEMA_HPP
