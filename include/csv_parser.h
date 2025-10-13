#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "types.h"

namespace splendor {

class CSVParser {
public:
    static std::vector<Card> ParseCardsCSV(const std::string& filename);
    
private:
    static Color StringToColor(const std::string& color_str);
    static std::array<uint8_t, DEV_COLORS> ParseCost(const std::vector<std::string>& cost_fields);
};

} 