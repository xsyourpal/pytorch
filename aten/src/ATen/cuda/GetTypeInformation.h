#pragma once

#include <cuda.h>

struct BasicType {
  std::string name;
  size_t offset;
  size_t size;
  bool is_pointer;
};

struct ArrayType;

// StructType has no copy constructor because of the presence of unique_ptr
struct StructType {
  std::string name;
  std::unique_ptr<std::vector<std::variant<BasicType, StructType, ArrayType>>> members;
  // we need a size field because structs can have tail padding.
  size_t size;
};

struct ArrayType {
  std::string name;
  std::unique_ptr<std::variant<BasicType, StructType, ArrayType>> element_type;
  size_t num_elements;
};

using ArgumentInformation = StructType;

std::vector<ArgumentInformation>
getArgumentInformation(const char* linkageName, const std::string& elfPath);

std::vector<ArgumentInformation>
getArgumentInformation(const char* linkageName, void *buffer, size_t buffer_size);

std::unordered_map<std::string, std::vector<ArgumentInformation>>
get_argument_information(const std::vector<std::string> &function_names);

std::vector<ArgumentInformation>
get_argument_information(CUfunction func);

bool is_equal(void *arg1, void *arg2, ArgumentInformation info);

void prettyPrintArgumentInfo(const std::vector<ArgumentInformation>& args);
