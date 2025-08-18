"""Pokemon Red/Blue specific event tracking system."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import time
from dataclasses import dataclass, field


# Pokemon Red character encoding mapping
POKEMON_CHARSET = {
    0x80: 'A', 0x81: 'B', 0x82: 'C', 0x83: 'D', 0x84: 'E', 0x85: 'F', 0x86: 'G', 0x87: 'H',
    0x88: 'I', 0x89: 'J', 0x8A: 'K', 0x8B: 'L', 0x8C: 'M', 0x8D: 'N', 0x8E: 'O', 0x8F: 'P',
    0x90: 'Q', 0x91: 'R', 0x92: 'S', 0x93: 'T', 0x94: 'U', 0x95: 'V', 0x96: 'W', 0x97: 'X',
    0x98: 'Y', 0x99: 'Z', 0x9A: '(', 0x9B: ')', 0x9C: ':', 0x9D: ';', 0x9E: '[', 0x9F: ']',
    0xA0: 'a', 0xA1: 'b', 0xA2: 'c', 0xA3: 'd', 0xA4: 'e', 0xA5: 'f', 0xA6: 'g', 0xA7: 'h',
    0xA8: 'i', 0xA9: 'j', 0xAA: 'k', 0xAB: 'l', 0xAC: 'm', 0xAD: 'n', 0xAE: 'o', 0xAF: 'p',
    0xB0: 'q', 0xB1: 'r', 0xB2: 's', 0xB3: 't', 0xB4: 'u', 0xB5: 'v', 0xB6: 'w', 0xB7: 'x',
    0xB8: 'y', 0xB9: 'z', 0xE1: 'PK', 0xE2: 'MN', 0xE6: '?', 0xE7: '!', 0xE8: '.', 0xEF: '♂',
    0xF5: '♀', 0xF0: '₽', 0xF1: '.', 0xF2: ',', 0xF3: "'", 0xF4: "'", 0xF6: '…', 0xF7: '♂',
    0xF8: '♀', 0xF9: '-', 0xFA: '?', 0xFB: '!', 0xFC: '.', 0xFD: '&', 0xFE: 'é', 0x50: '', 0x00: ''
}

# Pokemon names (first 151)
POKEMON_NAMES = [
    "", "Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon", "Charizard",
    "Squirtle", "Wartortle", "Blastoise", "Caterpie", "Metapod", "Butterfree",
    "Weedle", "Kakuna", "Beedrill", "Pidgey", "Pidgeotto", "Pidgeot", "Rattata",
    "Raticate", "Spearow", "Fearow", "Ekans", "Arbok", "Pikachu", "Raichu",
    "Sandshrew", "Sandslash", "Nidoran♀", "Nidorina", "Nidoqueen", "Nidoran♂",
    "Nidorino", "Nidoking", "Clefairy", "Clefable", "Vulpix", "Ninetales",
    "Jigglypuff", "Wigglytuff", "Zubat", "Golbat", "Oddish", "Gloom", "Vileplume",
    "Paras", "Parasect", "Venonat", "Venomoth", "Diglett", "Dugtrio", "Meowth",
    "Persian", "Psyduck", "Golduck", "Mankey", "Primeape", "Growlithe", "Arcanine",
    "Poliwag", "Poliwhirl", "Poliwrath", "Abra", "Kadabra", "Alakazam", "Machop",
    "Machoke", "Machamp", "Bellsprout", "Weepinbell", "Victreebel", "Tentacool",
    "Tentacruel", "Geodude", "Graveler", "Golem", "Ponyta", "Rapidash", "Slowpoke",
    "Slowbro", "Magnemite", "Magneton", "Farfetch'd", "Doduo", "Dodrio", "Seel",
    "Dewgong", "Grimer", "Muk", "Shellder", "Cloyster", "Gastly", "Haunter",
    "Gengar", "Onix", "Drowzee", "Hypno", "Krabby", "Kingler", "Voltorb",
    "Electrode", "Exeggcute", "Exeggutor", "Cubone", "Marowak", "Hitmonlee",
    "Hitmonchan", "Lickitung", "Koffing", "Weezing", "Rhyhorn", "Rhydon",
    "Chansey", "Tangela", "Kangaskhan", "Horsea", "Seadra", "Goldeen", "Seaking",
    "Staryu", "Starmie", "Mr. Mime", "Scyther", "Jynx", "Electabuzz", "Magmar",
    "Pinsir", "Tauros", "Magikarp", "Gyarados", "Lapras", "Ditto", "Eevee",
    "Vaporeon", "Jolteon", "Flareon", "Porygon", "Omanyte", "Omastar", "Kabuto",
    "Kabutops", "Aerodactyl", "Snorlax", "Articuno", "Zapdos", "Moltres",
    "Dratini", "Dragonair", "Dragonite", "Mewtwo", "Mew"
]

# Badge names
BADGE_NAMES = ["Boulder", "Cascade", "Thunder", "Rainbow", "Soul", "Marsh", "Volcano", "Earth"]


@dataclass
class PokemonEvent:
    """Represents a tracked Pokemon game event."""
    env_id: int
    timestamp: float
    event_type: str
    data: Dict[str, Any]


@dataclass
class GameState:
    """Snapshot of current Pokemon game state."""
    player_name: str = ""
    rival_name: str = ""
    party_count: int = 0
    party_pokemon: List[str] = field(default_factory=list)
    badges: int = 0
    badge_names: List[str] = field(default_factory=list)
    money: int = 0
    current_map: int = 0
    player_x: int = 0
    player_y: int = 0
    pokedex_owned: int = 0
    pokedex_seen: int = 0


class PokemonTracker:
    """Tracks Pokemon Red/Blue game events with minimal performance impact."""
    
    def __init__(self, env_id: int, track_interval: int = 100):
        """
        Initialize Pokemon event tracker.
        
        Args:
            env_id: Environment ID for tracking
            track_interval: Steps between tracking updates (higher = less overhead)
        """
        self.env_id = env_id
        self.track_interval = track_interval
        self.step_count = 0
        
        # Current game state cache
        self.current_state = GameState()
        self.previous_state = GameState()
        
        # Event history
        self.events: List[PokemonEvent] = []
        
        # Performance tracking
        self.last_deep_check = 0
        self.deep_check_interval = 1000  # Check text/names every 1000 steps
        
    def update(self, pyboy) -> List[PokemonEvent]:
        """
        Update tracking state and return new events.
        
        Args:
            pyboy: PyBoy emulator instance
            
        Returns:
            List of new events since last update
        """
        self.step_count += 1
        new_events = []
        
        # Quick checks every track_interval steps
        if self.step_count % self.track_interval == 0:
            new_events.extend(self._quick_check(pyboy))
            
        # Deep checks (text decoding) less frequently  
        if self.step_count % self.deep_check_interval == 0:
            new_events.extend(self._deep_check(pyboy))
            
        return new_events
    
    def _quick_check(self, pyboy) -> List[PokemonEvent]:
        """Perform quick numeric checks (party, badges, money)."""
        events = []
        current_time = time.time()
        
        try:
            # Store previous state
            self.previous_state = GameState(
                party_count=self.current_state.party_count,
                badges=self.current_state.badges,
                money=self.current_state.money,
                current_map=self.current_state.current_map,
                player_x=self.current_state.player_x,
                player_y=self.current_state.player_y
            )
            
            # Read current values
            self.current_state.party_count = pyboy.memory[0xD163]
            self.current_state.badges = pyboy.memory[0xD356]
            self.current_state.current_map = pyboy.memory[0xD35E] 
            self.current_state.player_x = pyboy.memory[0xD362]
            self.current_state.player_y = pyboy.memory[0xD361]
            
            # Read money (3 bytes, BCD format)
            money_bytes = [pyboy.memory[addr] for addr in range(0xD347, 0xD34A)]
            self.current_state.money = self._decode_bcd(money_bytes)
            
            # Check for events
            if self.current_state.party_count > self.previous_state.party_count:
                events.append(PokemonEvent(
                    env_id=self.env_id,
                    timestamp=current_time,
                    event_type="pokemon_caught", 
                    data={"party_count": self.current_state.party_count}
                ))
                
            if self.current_state.badges != self.previous_state.badges:
                new_badges = self._get_new_badges(self.previous_state.badges, self.current_state.badges)
                for badge in new_badges:
                    events.append(PokemonEvent(
                        env_id=self.env_id,
                        timestamp=current_time,
                        event_type="badge_earned",
                        data={"badge_name": badge, "total_badges": bin(self.current_state.badges).count('1')}
                    ))
                    
            if self.current_state.current_map != self.previous_state.current_map:
                events.append(PokemonEvent(
                    env_id=self.env_id,
                    timestamp=current_time,
                    event_type="location_changed",
                    data={
                        "old_map": self.previous_state.current_map,
                        "new_map": self.current_state.current_map,
                        "position": (self.current_state.player_x, self.current_state.player_y)
                    }
                ))
                
        except Exception as e:
            # Silently continue if memory read fails
            pass
            
        return events
    
    def _deep_check(self, pyboy) -> List[PokemonEvent]:
        """Perform expensive checks (text decoding, party composition)."""
        events = []
        current_time = time.time()
        
        try:
            # Read and decode player name
            name_bytes = [pyboy.memory[addr] for addr in range(0xD158, 0xD163)]
            new_player_name = self._decode_text(name_bytes)
            
            if new_player_name and new_player_name != self.current_state.player_name:
                self.current_state.player_name = new_player_name
                events.append(PokemonEvent(
                    env_id=self.env_id,
                    timestamp=current_time,
                    event_type="player_named",
                    data={"name": new_player_name}
                ))
                
            # Read and decode rival name 
            rival_bytes = [pyboy.memory[addr] for addr in range(0xD34A, 0xD352)]
            new_rival_name = self._decode_text(rival_bytes)
            
            if new_rival_name and new_rival_name != self.current_state.rival_name:
                self.current_state.rival_name = new_rival_name
                events.append(PokemonEvent(
                    env_id=self.env_id,
                    timestamp=current_time,
                    event_type="rival_named", 
                    data={"name": new_rival_name}
                ))
                
            # Read party Pokemon
            if self.current_state.party_count > 0:
                party_ids = [pyboy.memory[addr] for addr in range(0xD164, 0xD164 + self.current_state.party_count)]
                new_party = [self._get_pokemon_name(pid) for pid in party_ids]
                
                if new_party != self.current_state.party_pokemon:
                    self.current_state.party_pokemon = new_party
                    events.append(PokemonEvent(
                        env_id=self.env_id,
                        timestamp=current_time,
                        event_type="party_updated",
                        data={"party": new_party}
                    ))
                    
        except Exception as e:
            # Silently continue if memory read fails
            pass
            
        return events
    
    def _decode_text(self, byte_array: List[int]) -> str:
        """Decode Pokemon text from byte array."""
        text = ""
        for byte in byte_array:
            if byte == 0x50 or byte == 0x00:  # String terminator
                break
            char = POKEMON_CHARSET.get(byte, f"\\x{byte:02X}")
            text += char
        return text.strip()
    
    def _decode_bcd(self, byte_array: List[int]) -> int:
        """Decode BCD (Binary Coded Decimal) format used for money."""
        result = 0
        for byte in byte_array:
            high = (byte >> 4) & 0x0F
            low = byte & 0x0F
            result = result * 100 + high * 10 + low
        return result
    
    def _get_pokemon_name(self, pokemon_id: int) -> str:
        """Get Pokemon name from ID."""
        if 0 < pokemon_id < len(POKEMON_NAMES):
            return POKEMON_NAMES[pokemon_id]
        return f"Unknown#{pokemon_id}"
    
    def _get_new_badges(self, old_badges: int, new_badges: int) -> List[str]:
        """Get list of newly earned badge names."""
        diff = new_badges ^ old_badges  # XOR to find differences
        new_earned = diff & new_badges   # AND with new to get only newly set bits
        
        badges = []
        for i in range(8):
            if new_earned & (1 << i):
                badges.append(BADGE_NAMES[i])
        return badges
    
    def get_current_state(self) -> GameState:
        """Get current game state snapshot."""
        return GameState(
            player_name=self.current_state.player_name,
            rival_name=self.current_state.rival_name,
            party_count=self.current_state.party_count,
            party_pokemon=self.current_state.party_pokemon.copy(),
            badges=self.current_state.badges,
            badge_names=[BADGE_NAMES[i] for i in range(8) if self.current_state.badges & (1 << i)],
            money=self.current_state.money,
            current_map=self.current_state.current_map,
            player_x=self.current_state.player_x,
            player_y=self.current_state.player_y
        )
    
    def get_events_since(self, timestamp: float) -> List[PokemonEvent]:
        """Get all events since given timestamp."""
        return [event for event in self.events if event.timestamp >= timestamp]
    
    def clear_events(self) -> None:
        """Clear event history to save memory."""
        self.events.clear()