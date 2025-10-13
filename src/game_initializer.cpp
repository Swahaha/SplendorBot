#include "game_initializer.h"
#include "csv_parser.h"
#include <algorithm>
#include <stdexcept>

namespace splendor {

void GameInitializer::InitializeGame(SplendorGame& game) {
    InitializeBank(game);
    InitializeDecks(game);
    FillMarket(game);
    InitializeNobles(game);
    InitializePlayers(game);
}

void GameInitializer::InitializeBank(SplendorGame& game) {
    int token_count = (game.num_players_ == 2) ? 4 : 
                     (game.num_players_ == 3) ? 5 : 7;
    
    for (int i = 0; i < DEV_COLORS; ++i) game.bank_[i] = token_count;
    game.bank_[GOLD] = 5;
}

void GameInitializer::InitializeDecks(SplendorGame& game) {
    auto all_cards = CSVParser::ParseCardsCSV("data/cards.csv");
    for (const auto& card : all_cards) game.decks_[card.tier - 1].push_back(card);
    for (auto& deck : game.decks_) std::shuffle(deck.begin(), deck.end(), game.rng_); // shuffle decks
}

void GameInitializer::FillMarket(SplendorGame& game) {
    for (int t = 0; t < TIERS; ++t) {
        for (int s = 0; s < SLOTS_PER_TIER; ++s) {
            DrawIntoSlot(game, t, s);
        }
    }
}

void GameInitializer::InitializeNobles(SplendorGame& game) {
    game.nobles_ = {
        Noble{{3,3,3,0,0}, 3},
        Noble{{0,3,3,3,0}, 3},
        Noble{{0,3,3,3,0}, 3},
        Noble{{0,3,3,3,0}, 3}
    };
    std::shuffle(game.nobles_.begin(), game.nobles_.end(), game.rng_);
    game.nobles_.resize(game.num_players_ + 1);
}

void GameInitializer::InitializePlayers(SplendorGame& game) {
    game.players_.resize(game.num_players_);
}

void GameInitializer::DrawIntoSlot(SplendorGame& game, int tier, int slot) {
    auto& deck = game.decks_[tier];
    if (deck.empty()) {
        game.market_[tier][slot] = Card(); // Invalid card
    } else {
        game.market_[tier][slot] = deck.back();
        deck.pop_back();
    }
}

}