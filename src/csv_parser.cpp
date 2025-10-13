#include "csv_parser.h"
#include <iostream>
#include <algorithm>

namespace splendor {

std::vector<Card> CSVParser::ParseCardsCSV(const std::string& filename) {
    std::vector<Card> cards;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        try {
            std::stringstream ss(line);
            std::string field;
            std::vector<std::string> fields;
            
            while (std::getline(ss, field, ',')) {
                fields.push_back(field);
            }
            
            if (fields.size() < 8) continue;
            
            // Parse with default values on any error
            int tier = 1;
            try { tier = std::stoi(fields[0]); } catch (...) {}
            
            std::string color_str = fields[1];
            int prestige = 0;
            try { prestige = std::stoi(fields[2]); } catch (...) {}
            
            Color color = WHITE;
            if (color_str == "blue") color = BLUE;
            else if (color_str == "green") color = GREEN;
            else if (color_str == "red") color = RED;
            else if (color_str == "black") color = BLACK;
            else if (color_str == "gold") color = GOLD;
            
            std::array<uint8_t, DEV_COLORS> cost = {0, 0, 0, 0, 0};
            for (int i = 0; i < 5; i++) {
                try {
                    if (!fields[3 + i].empty()) {
                        cost[i] = std::stoi(fields[3 + i]);
                    }
                } catch (...) {
                    cost[i] = 0; // Default to 0 on any error
                }
            }
            
            cards.emplace_back(prestige, color, cost, tier);
        }
        catch (const std::exception& e) {
            std::cout << "Error processing line, skipping: " << line << " - " << e.what() << std::endl;
        }
    }
    
    std::cout << "Loaded " << cards.size() << " cards from CSV" << std::endl;
    return cards;
}

} // namespace splendor