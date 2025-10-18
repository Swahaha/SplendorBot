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
    static std::vector<Card> ParseCardsCSV(const std::string& filename); // used to parse cards.csv
};

} 